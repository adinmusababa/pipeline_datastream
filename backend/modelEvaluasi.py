"""
Evaluation metrics untuk clustering
Handles: Silhouette, DBI, Dunn Index, distances
"""

import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Evaluasi kualitas clustering dengan berbagai metrik
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.pre_merge_metrics = None
        logger.info(" ClusterEvaluator initialized")
    
    def evaluate_internal_metrics(self, data_buffer, counter, model, 
                                  save_to_db=True, label_suffix=""):
        """
        Evaluasi metrik internal clustering
        
        Args:
            data_buffer: deque of (features, cluster_id, true_label)
            counter: jumlah data yang telah diproses
            model: DBSTREAM model instance
            save_to_db: simpan ke database atau tidak
            label_suffix: label untuk stage (pre_merge/post_merge)
        
        Returns:
            dict: metrics data atau None
        """
        if len(data_buffer) < 5:
            return None

        try:
            # Extract features dan labels dari buffer
            features = np.array([f[:3] for f, _, _ in data_buffer])
            labels = np.array([cid for _, cid, _ in data_buffer])
            unique_clusters = np.unique(labels)

            if len(unique_clusters) < 2:
                logger.warning("  Less than 2 clusters, skipping evaluation")
                return None

            # Compute metrics
            try:
                silhouette = silhouette_score(features, labels)
                db_index = davies_bouldin_score(features, labels)
            except Exception as e:
                logger.warning(f"  Silhouette/DBI computation failed: {e}")
                silhouette, db_index = -1, -1

            dunn = self.compute_dunn_index(features, labels)
            intra, inter = self.compute_cluster_distances(features, labels)

            # Build metrics data
            metrics_data = {
                "total_data": counter,
                "active_clusters": len(unique_clusters),
                "silhouette": float(silhouette),
                "davies_bouldin": float(db_index),
                "dunn_index": float(dunn),
                "intra_distance": float(intra),
                "inter_distance": float(inter),
                "timestamp": datetime.utcnow()
            }

            # Save to database jika diminta
            if save_to_db:
                self._save_metrics_to_db(metrics_data, model, label_suffix)
                logger.info(f"Metrics | Silhouette: {silhouette:.3f} | "
                          f"DBI: {db_index:.3f} | Clusters: {len(unique_clusters)}")
            
            return metrics_data

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_metrics_to_db(self, metrics_data, model, label_suffix):
        """Internal method untuk save metrics ke database"""
        try:
            # Update latest metrics
            self.db.db.metrics.replace_one(
                {"_id": "latest"},
                {**{"_id": "latest"}, **metrics_data},
                upsert=True
            )

            # Archive metrics
            archive_doc = {
                **metrics_data,
                "clustering_threshold": getattr(model, "clustering_threshold", None),
                "fading_factor": getattr(model, "fading_factor", None),
                "stage": "post_merge" if not label_suffix else label_suffix
            }
            
            # Add comparison jika ada pre-merge metrics
            if self.pre_merge_metrics:
                archive_doc["pre_merge_clusters"] = self.pre_merge_metrics["active_clusters"]
                archive_doc["cluster_reduction"] = (
                    self.pre_merge_metrics["active_clusters"] - 
                    metrics_data["active_clusters"]
                )
                archive_doc["silhouette_improvement"] = (
                    metrics_data["silhouette"] - 
                    self.pre_merge_metrics["silhouette"]
                )
            
            self.db.db.metrics_archive.insert_one(archive_doc)
            
        except Exception as e:
            logger.error(f"Error saving metrics to DB: {e}")
    
    def compute_dunn_index(self, X, labels):
        """
        Compute Dunn Index
        Higher is better (better separation)
        """
        try:
            clusters = [X[labels == cid] for cid in set(labels)]
            if len(clusters) < 2:
                return 0
            
            # Minimum inter-cluster distance
            min_inter = np.min([
                np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))
                for i, c1 in enumerate(clusters)
                for j, c2 in enumerate(clusters) if i < j
            ])
            
            # Maximum intra-cluster distance
            max_intra = np.max([
                np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
                for c in clusters if len(c) > 1
            ])
            
            return min_inter / max_intra if max_intra > 0 else 0
            
        except Exception as e:
            logger.error(f" Error computing Dunn index: {e}")
            return 0
    
    def compute_cluster_distances(self, X, labels):
        """
        Compute intra-cluster dan inter-cluster distances
        
        Returns:
            tuple: (intra_distance, inter_distance)
        """
        try:
            clusters = [X[labels == cid] for cid in set(labels)]
            
            # Intra-cluster distance (average within cluster)
            intra = np.mean([
                np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
                for c in clusters if len(c) > 1
            ])
            
            # Inter-cluster distance (average between clusters)
            centroids = [np.mean(c, axis=0) for c in clusters if len(c) > 0]
            inter = np.mean([
                np.linalg.norm(c1 - c2)
                for i, c1 in enumerate(centroids)
                for j, c2 in enumerate(centroids) if i < j
            ]) if len(centroids) > 1 else 0
            
            return intra, inter
            
        except Exception as e:
            logger.error(f" Error computing cluster distances: {e}")
            return 0, 0