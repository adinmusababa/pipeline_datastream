import json
import time
import logging
import pika
import numpy as np
import pickle
import base64
from datetime import datetime, timedelta
from collections import deque
from pymongo import MongoClient
from river import cluster, metrics
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
import sys

try:
    import psutil
except Exception:
    psutil = None

try:
    from pympler import asizeof
except Exception:
    asizeof = None

logging.basicConfig(level=logging.INFO)

class modelConsumer:
    
    def __init__(self, queue='data_stream', host='localhost', buffer_size=1000):
        self.queue = queue
        self.host = host
        self.model = None
        self.data_buffer = deque(maxlen=buffer_size)
        self.counter = 0
        self.storage_tracking_interval = 1000

        # MongoDB
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["clustering_datastreaming"]
        
        self.setup_indexes()
        self.metric = metrics.rand.AdjustedRand()
        
        # CRITICAL: Restore full state including model
        self.restore_full_state()
        
        logging.info(f"✅ Consumer initialized | Processed: {self.counter} messages")

    def setup_indexes(self):
        self.db.clusters.create_index("timestamp")
        self.db.clusters.create_index("original_id", unique=True)
        self.db.merge_history.create_index([
            ("old_cluster_id", 1),
            ("new_cluster_id", 1),
            ("merge_timestamp", -1)
        ])
        self.db.StorageFootprint.create_index([("timestamp", -1)])
        self.db.consumer_state.create_index("_id")
        self.db.model_snapshots.create_index([("snapshot_at_counter", -1)])

    def restore_full_state(self):

        state = self.db.consumer_state.find_one({"_id": "current_state"})
        
        if state:
            self.counter = state.get("processed_count", 0)
            
            # CRITICAL: Restore full model state
            model_restored = self.restore_model_state()
            
            if model_restored:
                logging.info(f"Full model state restored from snapshot")
            else:
                logging.warning(" No model snapshot found, initializing fresh")
                self.load_or_initialize_model()
            
            # Restore buffer
            recent_docs = list(self.db.clusters.find()
                             .sort("timestamp", -1)
                             .limit(self.data_buffer.maxlen))
            
            for doc in reversed(recent_docs):
                self.data_buffer.append((
                    doc['features'],
                    doc['cluster_id'],
                    doc.get('true_label')
                ))
            
            logging.info(f"Restored: {self.counter} processed, {len(self.data_buffer)} in buffer")
        else:
            logging.info("Starting fresh")
            self.load_or_initialize_model()

    def save_full_state(self):

        # Save model snapshot
        self.save_model_snapshot()
        
        # Save consumer state
        self.db.consumer_state.update_one(
            {"_id": "current_state"},
            {
                "$set": {
                    "processed_count": self.counter,
                    "last_update": datetime.utcnow(),
                    "buffer_size": len(self.data_buffer),
                    "model_snapshot_counter": self.counter,
                    "model_params": {
                        "clustering_threshold": self.model.clustering_threshold,
                        "fading_factor": self.model.fading_factor
                    }
                }
            },
            upsert=True
        )

    def save_model_snapshot(self):

        try:
            # Serialize model menggunakan pickle
            model_bytes = pickle.dumps(self.model)
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Save to database
            self.db.model_snapshots.insert_one({
                "snapshot_at_counter": self.counter,
                "timestamp": datetime.utcnow(),
                "model_state": model_b64,
                "model_type": "DBSTREAM",
                "parameters": {
                    "clustering_threshold": self.model.clustering_threshold,
                    "fading_factor": self.model.fading_factor
                }
            })
            
            # Keep only last 5 snapshots
            snapshots = list(self.db.model_snapshots.find()
                           .sort("snapshot_at_counter", -1)
                           .skip(5))
            if snapshots:
                self.db.model_snapshots.delete_many({
                    "_id": {"$in": [s["_id"] for s in snapshots]}
                })
            
            logging.info(f"Model snapshot saved at counter={self.counter}")
            
        except Exception as e:
            logging.error(f"Failed to save model snapshot: {e}")

    def restore_model_state(self):
        """
        Restore complete model state dari snapshot
        """
        try:
            # Get latest snapshot
            snapshot = self.db.model_snapshots.find_one(
                sort=[("snapshot_at_counter", -1)]
            )
            
            if not snapshot:
                return False
            
            # Deserialize model
            model_b64 = snapshot.get("model_state")
            model_bytes = base64.b64decode(model_b64)
            self.model = pickle.loads(model_bytes)
            
            logging.info(f"Model restored from snapshot at counter={snapshot['snapshot_at_counter']}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore model snapshot: {e}")
            return False

    def load_or_initialize_model(self):

        params = self.db.model_params.find_one({"_id": "best_dbstream_config"})
        
        if params:
            clustering_threshold = params.get("clustering_threshold", 0.8)
            fading_factor = params.get("fading_factor", 0.0005)
        else:

            clustering_threshold = 0.8  # Lebih tinggi dari 0.5
            fading_factor = 0.0001      # Lebih kecil = fade lebih lambat
            
            logging.warning("Using conservative default parameters")
        
        self.model = cluster.DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=50,
            intersection_factor=0.3,
            minimum_weight=1.0
        )
        # eps terlalu kecil relatif terhadap variasi data.
        # decay terlalu lemah (cluster lama tetap hidup walau sudah tidak relevan).
        
        logging.info(f"Model initialized: CT={clustering_threshold}, FF={fading_factor}")

    def is_already_processed(self, original_id):
        return self.db.clusters.find_one({"original_id": original_id}) is not None

    def process_datapoint(self, message):
        try:
            original_id = message.get('metadata', {}).get('original_id')

            # Skip duplicates
            if original_id and self.is_already_processed(original_id):
                logging.debug(f"Skipping duplicate: {original_id}")
                return
            
            features = message['features']
            x = {f'x{i}': val for i, val in enumerate(features)}

            # Learn & predict
            self.model.learn_one(x)
            cluster_id = self.model.predict_one(x)


            if cluster_id is None:
                cluster_id = 0
                logging.warning("luster ID was None, assigned to 0")

            true_label = message.get("label", None)
            self.data_buffer.append((features, cluster_id, true_label))
            self.counter += 1

            # Save to MongoDB
            metadata = message.get('metadata', {})
            raw_data = message.get('raw_data', {})
            
            self.db.clusters.insert_one({
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
                "processing_counter": self.counter  # Track order
            })

            # Periodic operations
            if self.counter % 1000 == 0:
                self.evaluate_internal_metrics()
                self.merge_similar_clusters()
                self.track_storage_footprint()
                self.save_full_state()  # Save FULL state termasuk model
                logging.info(f"Full checkpoint saved at {self.counter} messages")

            if self.counter % 100 == 0:
                num_microclusters = len(getattr(self.model, 'centers', []))
                logging.info(f"Processed: {self.counter} | Cluster: {cluster_id} | Micro-clusters: {num_microclusters}")

        except Exception as e:
            logging.error(f"Error processing datapoint: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def evaluate_internal_metrics(self):
        if len(self.data_buffer) < 5:
            return

        try:
            features = np.array([f[:3] for f, _, _ in self.data_buffer])
            labels = np.array([cid for _, cid, _ in self.data_buffer])
            unique_clusters = np.unique(labels)

            if len(unique_clusters) < 2:
                return

            try:
                silhouette = silhouette_score(features, labels)
                db_index = davies_bouldin_score(features, labels)
            except:
                silhouette, db_index = -1, -1

            dunn = self.compute_dunn_index(features, labels)
            intra, inter = self.compute_cluster_distances(features, labels)

            # SOLUSI 5: Track over-segmentation
            over_segmentation_ratio = len(unique_clusters) / len(self.data_buffer)

            self.db.metrics.replace_one(
                {"_id": "latest"},
                {
                    "_id": "latest",
                    "total_data": self.counter,
                    "active_clusters": len(unique_clusters),
                    "silhouette": float(silhouette),
                    "davies_bouldin": float(db_index),
                    "dunn_index": float(dunn),
                    "intra_distance": float(intra),
                    "inter_distance": float(inter),
                    "over_segmentation_ratio": float(over_segmentation_ratio),
                    "timestamp": datetime.utcnow()
                },
                upsert=True
            )

            self.db.metrics_archive.insert_one({
                "timestamp": datetime.utcnow(),
                "silhouette": float(silhouette),
                "davies_bouldin": float(db_index),
                "dunn_index": float(dunn),
                "intra_distance": float(intra),
                "inter_distance": float(inter),
                "total_data": self.counter,
                "active_clusters": len(unique_clusters),
                "over_segmentation_ratio": float(over_segmentation_ratio),
                "clustering_threshold": getattr(self.model, "clustering_threshold", None),
                "fading_factor": getattr(self.model, "fading_factor", None)
            })
            
            # Warning jika over-segmentation
            if over_segmentation_ratio > 0.1:  # Lebih dari 10% data jadi cluster sendiri
                logging.warning(f"OVER-SEGMENTATION DETECTED: {len(unique_clusters)} clusters for {len(self.data_buffer)} points ({over_segmentation_ratio:.2%})")
            
            logging.info(f"Metrics | Silhouette: {silhouette:.3f} | DBI: {db_index:.3f} | Clusters: {len(unique_clusters)}")

        except Exception as e:
            logging.error(f"Error in evaluation: {e}")

    def merge_similar_clusters(self, threshold=0.7):

        if len(self.data_buffer) < 5:
            return

        try:
            features = np.array([f[:3] for f, _, _ in self.data_buffer])
            labels = np.array([cid for _, cid, _ in self.data_buffer])
            unique_clusters = np.unique(labels)
            
            if len(unique_clusters) < 2:
                return

            # Hitung centroid
            centroids = np.array([np.mean(features[labels == cid], axis=0) for cid in unique_clusters])
            if centroids.shape[0] < 2:
                return

            # HAC pada centroid dengan threshold lebih aggressive
            Z = linkage(centroids, method='average')
            

            over_seg_ratio = len(unique_clusters) / len(self.data_buffer)
            if over_seg_ratio > 0.1:
                # Jika terlalu banyak cluster, merge lebih agresif
                adaptive_threshold = threshold * 1.5
                logging.info(f"Over-segmentation detected, using adaptive threshold: {adaptive_threshold:.2f}")
            else:
                adaptive_threshold = threshold
            
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
                return

            # Update buffer
            new_buffer = deque(maxlen=self.data_buffer.maxlen)
            for f, cid, lbl in self.data_buffer:
                new_cid = merged_map.get(int(cid), int(cid))
                new_buffer.append((f, new_cid, lbl))
            self.data_buffer = new_buffer

            # Update database
            current_time = datetime.utcnow()
            for old_cid, new_cid in merged_map.items():
                self.db.clusters.update_many(
                    {"cluster_id": int(old_cid)},
                    {
                        "$set": {
                            "cluster_id": int(new_cid),
                            "last_updated": current_time
                        }
                    }
                )
                self.db.merge_history.insert_one({
                    "old_cluster_id": int(old_cid),
                    "new_cluster_id": int(new_cid),
                    "merge_timestamp": current_time,
                    "threshold_used": adaptive_threshold,
                    "method": "hac_average_centroid_adaptive"
                })

            # Cleanup
            remaining_olds = list(self.db.clusters.distinct("cluster_id", {"cluster_id": {"$in": [int(k) for k in merged_map.keys()]}}))
            if remaining_olds:
                self.db.clusters.delete_many({"cluster_id": {"$in": remaining_olds}})

            self.evaluate_internal_metrics()
            logging.info(f"HAC merge complete: {len(merged_map)} clusters merged")

        except Exception as e:
            logging.error(f"Error in merge: {e}")

    def compute_dunn_index(self, X, labels):
        try:
            clusters = [X[labels == cid] for cid in set(labels)]
            if len(clusters) < 2:
                return 0
            min_inter = np.min([
                np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))
                for i, c1 in enumerate(clusters)
                for j, c2 in enumerate(clusters) if i < j
            ])
            max_intra = np.max([
                np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
                for c in clusters if len(c) > 1
            ])
            return min_inter / max_intra if max_intra > 0 else 0
        except:
            return 0

    def compute_cluster_distances(self, X, labels):
        try:
            clusters = [X[labels == cid] for cid in set(labels)]
            intra = np.mean([
                np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
                for c in clusters if len(c) > 1
            ])
            centroids = [np.mean(c, axis=0) for c in clusters if len(c) > 0]
            inter = np.mean([
                np.linalg.norm(c1 - c2)
                for i, c1 in enumerate(centroids)
                for j, c2 in enumerate(centroids) if i < j
            ]) if len(centroids) > 1 else 0
            return intra, inter
        except:
            return 0, 0

    def track_storage_footprint(self):
        """Track storage metrics"""
        storage_metrics = {
            "timestamp": datetime.utcnow(),
            "model_metrics": {
                "model_size_bytes": None,
                "buffer_size_bytes": None,
                "total_points_processed": self.counter,
                "buffer_length": len(self.data_buffer)
            },
            "database_metrics": {},
            "model_params": {}
        }

        try:
            if asizeof:
                storage_metrics["model_metrics"]["model_size_bytes"] = asizeof.asizeof(self.model)
                storage_metrics["model_metrics"]["buffer_size_bytes"] = asizeof.asizeof(self.data_buffer)
            else:
                storage_metrics["model_metrics"]["model_size_bytes"] = sys.getsizeof(self.model)
                storage_metrics["model_metrics"]["buffer_size_bytes"] = sys.getsizeof(self.data_buffer)
        except Exception as e:
            logging.debug(f"Model size probe failed: {e}")

        try:
            if psutil:
                p = psutil.Process()
                storage_metrics["process_memory"] = {
                    "rss_mb": p.memory_info().rss / (1024 * 1024),
                    "vms_mb": p.memory_info().vms / (1024 * 1024)
                }
        except:
            pass

        for coll in ["clusters", "metrics", "metrics_archive", "merge_history"]:
            try:
                stats = self.db.command("collstats", coll)
                storage_metrics["database_metrics"][coll] = {
                    "count": stats.get("count", 0),
                    "size_bytes": stats.get("size", 0),
                    "storage_bytes": stats.get("storageSize", 0)
                }
            except Exception:
                try:
                    cnt = self.db[coll].count_documents({})
                    storage_metrics["database_metrics"][coll] = {"count": cnt}
                except:
                    storage_metrics["database_metrics"][coll] = {}

        try:
            storage_metrics["model_params"] = {
                "clustering_threshold": self.model.clustering_threshold,
                "fading_factor": self.model.fading_factor
            }
        except:
            storage_metrics["model_params"] = {}

        try:
            self.db.StorageFootprint.insert_one(storage_metrics)
        except Exception as e:
            logging.error(f"Failed to insert StorageFootprint: {e}")

    def connect(self):
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
            logging.warning(f"Cannot connect to RabbitMQ: {e}")
            return None, None

    def run(self):
        while True:
            connection, channel = self.connect()
            if connection is None or channel is None:
                logging.info("Waiting for RabbitMQ...")
                time.sleep(5)
                continue

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    self.process_datapoint(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logging.error(f"Error in callback: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

            channel.basic_consume(
                queue=self.queue,
                on_message_callback=callback,
                auto_ack=False
            )

            logging.info(f"Listening to queue '{self.queue}'...")
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                logging.info("Interrupted by user. Saving state...")
                self.save_full_state()
                break
            except Exception as e:
                logging.warning(f"Connection error: {e}")
                self.save_full_state()
                time.sleep(5)


if __name__ == "__main__":
    consumer = modelConsumer()
    consumer.run()