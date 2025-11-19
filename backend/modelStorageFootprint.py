"""
Storage footprint tracking
Monitors memory, disk usage, dan model size
"""

import sys
import logging
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

try:
    from pympler import asizeof
except Exception:
    asizeof = None

logger = logging.getLogger(__name__)


class StorageTracker:
    """
    Track storage metrics untuk monitoring resource usage
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        logger.info("StorageTracker initialized")
    
    def track_storage_footprint(self, model, data_buffer, counter):
        """
        Track dan save storage metrics
        
        Args:
            model: DBSTREAM model instance
            data_buffer: deque buffer
            counter: jumlah data processed
        """
        storage_metrics = {
            "timestamp": datetime.utcnow(),
            "model_metrics": {
                "model_size_bytes": None,
                "buffer_size_bytes": None,
                "total_points_processed": counter,
                "buffer_length": len(data_buffer)
            },
            "database_metrics": {},
            "model_params": {}
        }

        # Model & buffer sizes
        try:
            if asizeof:
                storage_metrics["model_metrics"]["model_size_bytes"] = asizeof.asizeof(model)
                storage_metrics["model_metrics"]["buffer_size_bytes"] = asizeof.asizeof(data_buffer)
            else:
                storage_metrics["model_metrics"]["model_size_bytes"] = sys.getsizeof(model)
                storage_metrics["model_metrics"]["buffer_size_bytes"] = sys.getsizeof(data_buffer)
        except Exception as e:
            logger.debug(f"Model size probe failed: {e}")

        # Process memory
        try:
            if psutil:
                p = psutil.Process()
                storage_metrics["process_memory"] = {
                    "rss_mb": p.memory_info().rss / (1024 * 1024),
                    "vms_mb": p.memory_info().vms / (1024 * 1024)
                }
        except Exception as e:
            logger.debug(f"Process memory probe failed: {e}")

        # Database collection stats
        for coll in ["clusters", "metrics", "metrics_archive", "merge_history"]:
            try:
                stats = self.db.db.command("collstats", coll)
                storage_metrics["database_metrics"][coll] = {
                    "count": stats.get("count", 0),
                    "size_bytes": stats.get("size", 0),
                    "storage_bytes": stats.get("storageSize", 0)
                }
            except Exception:
                try:
                    cnt = self.db.db[coll].count_documents({})
                    storage_metrics["database_metrics"][coll] = {"count": cnt}
                except:
                    storage_metrics["database_metrics"][coll] = {}

        # Model parameters
        try:
            storage_metrics["model_params"] = {
                "clustering_threshold": model.clustering_threshold,
                "fading_factor": model.fading_factor
            }
        except:
            storage_metrics["model_params"] = {}

        # Save to database
        try:
            self.db.db.StorageFootprint.insert_one(storage_metrics)
            logger.debug("Storage footprint saved")
        except Exception as e:
            logger.error(f"Failed to save storage footprint: {e}")