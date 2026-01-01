import pickle
import base64
import logging
from datetime import datetime
from pymongo import MongoClient

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_name="clustering_datastreaming"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.setup_indexes()
        logger.info(f"DatabaseManager initialized for '{db_name}'")
    
    def setup_indexes(self):
        """Setup all required indexes"""
        try:
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
            logger.info("Database indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def save_consumer_state(self, counter, buffer_size, model_params):
        """Save current consumer state"""
        try:
            self.db.consumer_state.update_one(
                {"_id": "current_state"},
                {
                    "$set": {
                        "processed_count": counter,
                        "last_update": datetime.utcnow(),
                        "buffer_size": buffer_size,
                        "model_snapshot_counter": counter,
                        "model_params": model_params
                    }
                },
                upsert=True
            )
            logger.debug(f"Consumer state saved at counter={counter}")
            return True
        except Exception as e:
            logger.error(f"Failed to save consumer state: {e}")
            return False
    
    def restore_consumer_state(self):
        """Restore consumer state dari database"""
        try:
            state = self.db.consumer_state.find_one({"_id": "current_state"})
            if state:
                logger.info(f" Consumer state found: {state.get('processed_count', 0)} processed")
                return state
            else:
                logger.info("No previous consumer state found")
                return None
        except Exception as e:
            logger.error(f"Error restoring consumer state: {e}")
            return None
    
    def save_model_snapshot(self, model, counter):
        """Save model snapshot ke database"""
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            # Save to database
            self.db.model_snapshots.insert_one({
                "snapshot_at_counter": counter,
                "timestamp": datetime.utcnow(),
                "model_state": model_b64,
                "model_type": "DBSTREAM",
                "parameters": {
                    "clustering_threshold": model.clustering_threshold,
                    "fading_factor": model.fading_factor
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
            
            logger.info(f"Model snapshot saved at counter={counter}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model snapshot: {e}")
            return False
    
    def restore_model_snapshot(self):
        """Restore model snapshot dari database"""
        try:
            snapshot = self.db.model_snapshots.find_one(
                sort=[("snapshot_at_counter", -1)]
            )
            
            if not snapshot:
                logger.warning("No model snapshot found")
                return None
            
            # Deserialize model
            model_b64 = snapshot.get("model_state")
            model_bytes = base64.b64decode(model_b64)
            model = pickle.loads(model_bytes)
            
            logger.info(f"Model restored from snapshot at counter={snapshot['snapshot_at_counter']}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to restore model snapshot: {e}")
            return None
    
    def restore_buffer_data(self, buffer_maxlen):
        """Restore buffer data dari database"""
        try:
            recent_docs = list(self.db.clusters.find()
                             .sort("timestamp", -1)
                             .limit(buffer_maxlen))
            
            buffer_data = []
            for doc in reversed(recent_docs):
                # Restore buffer as (features, cluster_id) — no true labels
                buffer_data.append((
                    doc['features'],
                    doc['cluster_id']
                ))
            
            logger.info(f"Buffer restored: {len(buffer_data)} items")
            return buffer_data
            
        except Exception as e:
            logger.error(f"Error restoring buffer: {e}")
            return []
    
    def is_data_processed(self, original_id):
        """Check if data sudah pernah diproses"""
        return self.db.clusters.find_one({"original_id": original_id}) is not None
    
    def save_cluster_data(self, data):
        """Save cluster result ke database"""
        try:
            self.db.clusters.insert_one(data)
            return True
        except Exception as e:
            logger.error(f"Error saving cluster data: {e}")
            return False
    
    def save_merge_history(self, old_cid, new_cid, threshold, method="hac_average_centroid"):
        """Save merge history"""
        try:
            self.db.merge_history.insert_one({
                "old_cluster_id": int(old_cid),
                "new_cluster_id": int(new_cid),
                "merge_timestamp": datetime.utcnow(),
                "threshold_used": threshold,
                "method": method
            })
            return True
        except Exception as e:
            logger.error(f"Error saving merge history: {e}")
            return False
    
    def update_merged_clusters(self, merged_map, threshold):
        """Update cluster IDs setelah merge"""
        try:
            current_time = datetime.utcnow()
            
            for old_cid, new_cid in merged_map.items():
                # Update cluster documents
                self.db.clusters.update_many(
                    {"cluster_id": int(old_cid)},
                    {"$set": {"cluster_id": int(new_cid), "last_updated": current_time}}
                )
                
                # Save merge history
                self.save_merge_history(old_cid, new_cid, threshold)
            
            # Cleanup old cluster IDs
            remaining_olds = list(self.db.clusters.distinct(
                "cluster_id", 
                {"cluster_id": {"$in": [int(k) for k in merged_map.keys()]}}
            ))
            if remaining_olds:
                self.db.clusters.delete_many({"cluster_id": {"$in": remaining_olds}})
            
            logger.info(f"Updated {len(merged_map)} merged clusters in database")
            return True
            
        except Exception as e:
            logger.error(f"Error updating merged clusters: {e}")
            return False
    
    def get_model_params(self):
        """Get best model parameters from database"""
        try:
            params = self.db.model_params.find_one({"_id": "best_dbstream_config"})
            if params:
                logger.info("Model parameters loaded from database")
                return {
                    "clustering_threshold": params.get("clustering_threshold", 0.8),
                    "fading_factor": params.get("fading_factor", 0.0005)
                }
            else:
                logger.warning(" No model parameters found, using defaults")
                return {
                    "clustering_threshold": 0.8,
                    "fading_factor": 0.0005
                }
        except Exception as e:
            logger.error(f"Error getting model params: {e}")
            return {
                "clustering_threshold": 0.8,
                "fading_factor": 0.0005
            }
    
    def close(self):
        """Close database connection"""
        try:
            self.client.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")