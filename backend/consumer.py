import json
import time
import pika
from datetime import datetime
from collections import deque
from river import cluster, metrics
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import sys

# Import custom modules
sys.path.append('backend')
from modelDB import DatabaseManager
from modelEvaluasi import ClusterEvaluator
from modelStorageFootprint import StorageTracker
from modelLogging import setup_logging, get_logger

# Setup logging
setup_logging(level='INFO', log_file='logs/consumer.log')
logger = get_logger(__name__)


class ModelConsumer:
    """
    Main consumer class untuk online clustering
    Modular design dengan separation of concerns
    """
    
    def __init__(self, queue='data_stream', host='localhost', buffer_size=1000):
        self.queue = queue
        self.host = host
        self.buffer_size = buffer_size
        
        # Components
        self.db_manager = DatabaseManager("clustering_datastreaming")
        self.evaluator = ClusterEvaluator(self.db_manager)
        self.storage_tracker = StorageTracker(self.db_manager)
        
        # Model & data
        self.model = None
        self.data_buffer = deque(maxlen=buffer_size)
        self.counter = 0
        self.metric = metrics.rand.AdjustedRand()
        
        # Restore state
        self.restore_full_state()
        
        logger.info(f" ModelConsumer initialized | Processed: {self.counter} messages")
    
    def restore_full_state(self):
        """Restore consumer state dari database"""
        state = self.db_manager.restore_consumer_state()
        
        if state:
            self.counter = state.get("processed_count", 0)
            
            # Restore model
            model = self.db_manager.restore_model_snapshot()
            if model:
                self.model = model
                logger.info(" Full model state restored from snapshot")
            else:
                logger.warning(" No model snapshot, initializing fresh")
                self.load_or_initialize_model()
            
            # Restore buffer
            buffer_data = self.db_manager.restore_buffer_data(self.buffer_size)
            for item in buffer_data:
                self.data_buffer.append(item)
            
            logger.info(f" State restored: {self.counter} processed, {len(self.data_buffer)} in buffer")
        else:
            logger.info("Starting fresh")
            self.load_or_initialize_model()
    
    def save_full_state(self):
        """Save consumer state ke database"""
        # Save model snapshot
        self.db_manager.save_model_snapshot(self.model, self.counter)
        
        # Save consumer state
        model_params = {
            "clustering_threshold": self.model.clustering_threshold,
            "fading_factor": self.model.fading_factor
        }
        self.db_manager.save_consumer_state(
            self.counter, 
            len(self.data_buffer), 
            model_params
        )
    
    def load_or_initialize_model(self):
        """Initialize DBSTREAM model"""
        params = self.db_manager.get_model_params()
        
        self.model = cluster.DBSTREAM(
            clustering_threshold=params["clustering_threshold"],
            fading_factor=params["fading_factor"],
            cleanup_interval=50,
            intersection_factor=0.3,
            minimum_weight=1.0
        )

    
    def process_datapoint(self, message):
        try:
            original_id = message.get('metadata', {}).get('original_id')

            # Skip duplicates (idempotency)
            if original_id and self.db_manager.is_data_processed(original_id):
                logger.debug(f"Skipping duplicate: {original_id}")
                return
            
            # Extract features
            features = message['features']
            x = {f'x{i}': val for i, val in enumerate(features)}

            # Learn & predict
            self.model.learn_one(x)
            cluster_id = self.model.predict_one(x)

            if cluster_id is None:
                cluster_id = 0
                logger.warning("Cluster ID was None, assigned to 0")

            # Add to buffer
            true_label = message.get("label", None)
            self.data_buffer.append((features, cluster_id, true_label))
            self.counter += 1

            # Save to database
            metadata = message.get('metadata', {})
            raw_data = message.get('raw_data', {})
            
            cluster_data = {
                "features": features,
                "raw_data": raw_data,
                "cluster_id": int(cluster_id),
                "true_label": true_label,
                "timestamp": datetime.utcnow(),
                "User ID": metadata.get('User ID'),
                "Item ID": metadata.get('Item ID'),
                "Category ID": metadata.get('Category ID'),
                "Behavior type": metadata.get('behavior_type'),
                "original_id": original_id,
                "processing_counter": self.counter
            }
            
            self.db_manager.save_cluster_data(cluster_data)

            # Periodic operations (every 100 messages)
            if self.counter % 100 == 0:
                self._periodic_operations()

        except Exception as e:
            logger.error(f"Error processing datapoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _periodic_operations(self):
        """Operasi periodik: evaluate, merge, track, save"""
        # 1. Evaluate SEBELUM merge (memory only)
        self.evaluator.pre_merge_metrics = self.evaluator.evaluate_internal_metrics(
            self.data_buffer, 
            self.counter, 
            self.model,
            save_to_db=False,
            label_suffix="pre_merge"
        )
        
        # 2. Merge clusters
        merge_happened = self.merge_similar_clusters()
        
        # 3. Evaluate SETELAH merge (save to DB)
        if merge_happened:
            self.evaluator.evaluate_internal_metrics(
                self.data_buffer,
                self.counter,
                self.model,
                save_to_db=True,
                label_suffix="post_merge"
            )
        else:
            # No merge, save pre-merge metrics
            if self.evaluator.pre_merge_metrics:
                self.db_manager.db.metrics.replace_one(
                    {"_id": "latest"},
                    {**{"_id": "latest"}, **self.evaluator.pre_merge_metrics},
                    upsert=True
                )
                archive_doc = {
                    **self.evaluator.pre_merge_metrics,
                    "clustering_threshold": self.model.clustering_threshold,
                    "fading_factor": self.model.fading_factor,
                    "stage": "no_merge"
                }
                self.db_manager.db.metrics_archive.insert_one(archive_doc)
        
        # 4. Track storage
        self.storage_tracker.track_storage_footprint(
            self.model, 
            self.data_buffer, 
            self.counter
        )
        
        # 5. Save full state
        self.save_full_state()
        
        # Clear temporary metrics
        self.evaluator.pre_merge_metrics = None
        
        logger.info(f"Checkpoint saved at {self.counter} messages")
    
    def merge_similar_clusters(self, threshold=0.7):
        """
        Merge similar clusters menggunakan HAC
        
        Returns:
            bool: True jika ada merge, False jika tidak
        """
        if len(self.data_buffer) < 5:
            return False

        try:
            features = np.array([f[:3] for f, _, _ in self.data_buffer])
            labels = np.array([cid for _, cid, _ in self.data_buffer])
            unique_clusters = np.unique(labels)
            
            if len(unique_clusters) < 2:
                return False

            # Compute centroids
            centroids = np.array([
                np.mean(features[labels == cid], axis=0) 
                for cid in unique_clusters
            ])
            if centroids.shape[0] < 2:
                return False

            # HAC linkage
            Z = linkage(centroids, method='average')
            
            # Adaptive threshold untuk over-segmentation
            over_seg_ratio = len(unique_clusters) / len(self.data_buffer)
            if over_seg_ratio > 0.1:
                adaptive_threshold = threshold * 1.5
                logger.info(f" Over-segmentation detected, using adaptive threshold: {adaptive_threshold:.2f}")
            else:
                adaptive_threshold = threshold
            
            # Cluster grouping
            group_labels = fcluster(Z, t=adaptive_threshold, criterion='distance')

            # Build merge map
            merged_map = {}
            for grp in np.unique(group_labels):
                members = unique_clusters[group_labels == grp]
                rep = int(np.min(members))
                for m in members:
                    if int(m) != rep:
                        merged_map[int(m)] = rep

            if not merged_map:
                return False

            # Update buffer
            new_buffer = deque(maxlen=self.data_buffer.maxlen)
            for f, cid, lbl in self.data_buffer:
                new_cid = merged_map.get(int(cid), int(cid))
                new_buffer.append((f, new_cid, lbl))
            self.data_buffer = new_buffer

            # Update database
            self.db_manager.update_merged_clusters(merged_map, adaptive_threshold)

            logger.info(f"HAC merge complete: {len(merged_map)} clusters merged")
            return True

        except Exception as e:
            logger.error(f" Error in merge: {e}")
            return False
    
    def connect(self):
        """Connect to RabbitMQ"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.host,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue=self.queue, durable=True)
            channel.basic_qos(prefetch_count=1)
            return connection, channel
        except Exception as e:
            logger.warning(f" Cannot connect to RabbitMQ: {e}")
            return None, None
    
    def run(self):
        """Main consumer loop"""
        logger.info(" Starting consumer...")
        
        while True:
            connection, channel = self.connect()
            if connection is None or channel is None:
                logger.info("Waiting for RabbitMQ...")
                time.sleep(5)
                continue

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    self.process_datapoint(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logger.error(f" Error in callback: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

            channel.basic_consume(
                queue=self.queue,
                on_message_callback=callback,
                auto_ack=False
            )

            logger.info(f"Listening to queue '{self.queue}'...")
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                logger.info("  Interrupted by user. Saving state...")
                self.save_full_state()
                break
            except Exception as e:
                logger.warning(f"  Connection error: {e}")
                self.save_full_state()
                time.sleep(5)

if __name__ == "__main__":
    consumer = ModelConsumer()
    consumer.run()
