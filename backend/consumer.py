# 16

import logging
import numpy as np
from datetime import datetime
import pika
from scipy.cluster.hierarchy import linkage, fcluster
from collections import deque
from pymongo import MongoClient
from river import metrics, cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time
import json
from datetime import timedelta
import sys



class modelConsumer:
    def __init__(self, queue='data_stream', host='localhost', buffer_size=1000):
        self.queue = queue
        self.host = host
        self.buffer_size = deque(maxlen=buffer_size)
        self.model = None
        self.counter = 0
        self.storage_tracking_interval = 1000

        # monggoDB setup
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['data_points_db']

        # setup index
        self.setup_indexes()

        # metrix evaluasi
        self.metric = metrics.rand.AdjustedRand()

        # lanjutkan dari state terakhir jika ada
        self.lanjutkan_dari_state_terakhir()

        # inisialisasi model
        self.init_model()

        # logging
        logging.info(f"inisialisasi consumer, memproses: {self.counter} message.")

    # fungsi utama memproses setiap kali data diterima
    def proses_data_consumer(self, message):
        # pastikan data memiliki fitur yang sesuai untuk memroses
        try:
            # orriginal_id = message.get['metadata', {}].get['original_id']
            orriginal_id = message.get('metadata', {}).get('original_id')

            if orriginal_id is None:
                logging.warning("Data point tanpa original_id, dilewati.")
                return
            features = message['features']
            x = {f'x{i}': val for i, val in enumerate(features)
                 }
            # model OML update
            self.model.learn_one(x)
            cluster_id = self.model.predict_one(x)
            # add to buffer
            self.buffer_size.append((features, cluster_id))

            self.counter += 1

            # # simpan ke database setiap interval tertentu
            # metadata = message.get('metadata', {})
            # raw_data = message.get('raw_data', {})

            # self.db.clusters.insert_one({
            #     "features": features,
            #     "raw_data": raw_data,
            #     # "cluster_id": int(cluster_id),
            #     "cluster_id": int(cluster_id) if cluster_id is not None and not np.isnan(cluster_id) else -1,
            #     "timestamp": datetime.utcnow(),
            #     "User ID": metadata.get('User ID'),
            #     "Item ID": metadata.get('Item ID'),
            #     "Category ID": metadata.get('Category ID'),
            #     "Behavior type": metadata.get('behavior_type'),
            #     "original_id": orriginal_id
            # })
            
            # evaluasi secara periodik
            if self.counter % 100 == 0:
                self.evaluasi_cluster()
                self.gabung_cluster_HAC()
                self.track_storage_footprint()
                self.save_state()  # Save checkpoint
                logging.info(f"Checkpoint saved at {self.counter} messages")

            if self.counter % 100 == 0:
                logging.info(f"Processed: {self.counter} | Cluster: {cluster_id} | Buffer: {len(self.buffer_size)}")

        except Exception as e:
            logging.error(f"Error processing datapoint: {e}")
    
    # muat dan inisialisasi model
    def init_model(self):
        self.model = cluster.DBSTREAM(
            fading_factor=0.001,
            clustering_threshold=0.5,
            cleanup_interval=2,
            intersection_factor=0.5,
            minimum_weight=1.0
        )
        logging.info("Model DBSTREAM diinisialisasi.")

    # fungsi evaluassi Silhouette dan Adjusted Rand Index
    def evaluasi_cluster(self):
        if len(self.buffer_size) < 5:
            return
        
        # features = np.array([f[:3] for f, _, _ in self.buffer_size]) 
        # labels = np.array([cid for _, cid, _ in self.buffer_size])
        features = np.array([f for f, _ in self.buffer_size])
        labels = np.array([cid for _, cid in self.buffer_size])
        
        unique_clusters = np.unique(labels)
        # unique_clusters = set(labels)

        if len(unique_clusters) < 2: 
            return 

        try: # Menghitung metrik silhouette dan Davies-Bouldin Index
            silhouette = silhouette_score(features, labels) 
            db_index = davies_bouldin_score(features, labels)
        except Exception:
            silhouette, db_index = -1, -1

        dunn = self.compute_index_dunn(features, labels) # Menghitung indeks Dunn untuk mengevaluasi pemisahan dan kepadatan klaster
        intra, inter = self.jarak_cluster(features, labels) # Menghitung jarak intra-klaster dan inter-klaster

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
            "clustering_threshold": getattr(self.model, "clustering_threshold", None),
            "fading_factor": getattr(self.model, "fading_factor", None)
        })
        
        logging.info(f"Metrics | Silhouette: {silhouette:.3f} | DBI: {db_index:.3f} | Dunn: {dunn:.3f}")

    # fungsi HAC untuk menggabungkan cluster yang mirip
    def gabung_cluster_HAC(self, threshold=0.7):
        # features = np.array([f[:3] for f, _, _ in self.buffer_size]) 
        # labels = np.array([cid for _, cid, _ in self.buffer_size])
        features = np.array([f for f, _ in self.buffer_size])
        labels = np.array([cid for _, cid in self.buffer_size])
        unique_clusters = np.unique(labels)

        if len(self.buffer_size) < 5: 
            return
        
        # hitung centroid setiap cluster yang diurutkan berdasarkan unique cluster
        centroids = np.array([np.mean(features[labels == cid], axis=0) for cid in unique_clusters])
        if len(centroids) < 2:
            return
        
        # terapkan HAC pada centroid
        hac =  linkage(centroids, method='average', metric='cosine')

        cluster_groups = fcluster(hac, threshold, criterion='distance')

        # peta cluster lama ke cluster baru
        cluster_map = {}
        for grp in np.unique(cluster_groups):
            anggota = unique_clusters[cluster_groups == grp]
            rep = int(np.min(anggota))

            for m in anggota:
                if m != rep:
                    cluster_map[int(m)] = rep

        if not cluster_map:
            return
        
        new_buffer = deque(maxlen=self.buffer_size)
        for features, cid, true_label in self.buffer_size:
            new_cid = cluster_map.get(cid, cid)
            new_buffer.append((features, new_cid, true_label))
        self.buffer_size = new_buffer

        # update database
        waktu_proses = datetime.utcnow()
        for old_cid, new_cid, in cluster_map.items():
            self.db.clusters.update_many(
                {"cluster_id": old_cid},
                {
                    "$set": {
                        "cluster_id": new_cid,
                        "merged_at": waktu_proses
                    }
                }
            )

            # simpan riwayat penggabungan
            self.db.cluster_merges.insert_one({
                "old_cluster_id": int(old_cid),
                "new_cluster_id": int(new_cid),
                "merge_timestamp": waktu_proses,
                "threshold_used": threshold,
                "method": "hac_average_centroid"
            })

        reaming_olds = list(self.db.clusters.distinct("cluster_id", {
            {"cluster_id": {"$in": [int(k) for k in cluster_map.keys()]}}
        }))

        if reaming_olds:
            self.db.clusters.delete_many({
                "cluster_id": {"$in": reaming_olds}
            })
        
        self.evaluasi_cluster()
        logging.info(f"HAC Merge | Clusters merged: {len(cluster_map)} | Remaining clusters: {len(unique_clusters) - len(cluster_map)}")

    # menghitung pemisahan antar cluster
    def compute_index_dunn(self, features, labels):
        cluster = [features[labels == cid] for cid in set(labels)]

        if len(cluster) < 2:
            return 0

        min_inter = np.min([
            np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))
            for i, c1 in enumerate(cluster) 
            for j, c2 in enumerate(cluster) if i < j
        ])

        max_intra = np.max([
            np.mean(np.linalg.norm( c - np.mean(c, axis=0), axis=1))
            for c in cluster if len(c) > 1
        ])

        return min_inter/max_intra if max_intra > 0 else 0
    
    # htung jarak intra dan inter cluster
    def jarak_cluster(self, features, labels):
        cluster = [features[labels == cid] for cid in set(labels)]

        intra = np.mean([
            np.mean(np.linalg.norm(c - np.mean(c, axis=0), axis=1))
            for c in cluster if len(c) > 1
        ])
        inter = np.mean([
            np.linalg.norm(np.mean(c1, axis=0) - np.mean(c2, axis=0))
            for i, c1 in enumerate(cluster) 
            for j, c2 in enumerate(cluster) if i < j
        ]) if len(cluster) > 1 else 0
        return intra, inter

    # fungsi setup index pada mongoDB
    def setup_indexes(self):
        self.db.clusters.create_index("timestamp")
        self.db.clusters.create_index("original_id", unique=True)
        self.db.merge_history.create_index([
            ("old_cluster_id", 1),
            ("new_cluster_id", 1),
            ("merge_timestamp", -1)
        ])
        self.db.StorageFootprint.create_index([
            ("timestamp", -1)
        ])
        self.db.consumer_state.create_index("_id") # Untuk menyimpan status konsumen, _id adalah kunci utama dokumen

    def lanjutkan_dari_state_terakhir(self):
        state = self.db.consumer_state.find_one({"_id": "checkpoint"})
        if state:
            self.counter = state.get("counter", 0)
            logging.info(f"Melanjutkan dari checkpoint: {self.counter} pesan telah diproses.")
        else:
            logging.info("Tidak ada checkpoint ditemukan, memulai dari awal.")
    
    # bersihkan hail eval lama
    def clear_old_evaluations(self, days=7):
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = self.db.metrics_archive.delete_many({"timestamp": {"$lt": cutoff}})
        logging.info(f"Old evaluations cleared: {result.deleted_count} records older than {days} days removed.")

    # fugnsi untuk menyimpan status konsumen ke database untuk melanjutkan pemrosesan jika sistem dihentikan
    def save_state(self): 
        self.db.consumer_state.update_one(  
            {"_id": "current_state"}, # Filter untuk menemukan dokumen dengan _id "current_state"
            {
                "$set": {
                    "processed_count": self.counter,
                    "last_update": datetime.utcnow(),
                    "buffer_size": len(self.buffer_size),
                    "model_params": {
                        "clustering_threshold": getattr(self.model, "clustering_threshold", None),
                        "fading_factor": getattr(self.model, "fading_factor", None)
                    }
                }
            },
            upsert=True # Jika dokumen tidak ada, buat dokumen baru
        )
    # fungsi untuk menyimpan footprint penyimpanan
    def track_storage_footprint(self):
        from pympler import asizeof
        store_metrics = {
            "timestamp": datetime.utcnow(),
            "matrics": {
                "total_points_processed": self.counter,
                "buffer_length": len(self.buffer_size)
            },
            "database_matrics":{},
            "model_params": {}
        }

        try:
            if asizeof:
                store_metrics["model_metrics"]["model_size_bytes"] = asizeof.asizeof(self.model)
                store_metrics["model_metrics"]["buffer_size_bytes"] = asizeof.asizeof(self.buffer_size)
            else:
                store_metrics["model_metrics"]["model_size_bytes"] = sys.getsizeof(self.model)
                store_metrics["model_metrics"]["buffer_size_bytes"] = sys.getsizeof(self.buffer_size)
        except Exception as e:
            asizeof = None
            logging.debug(f"ukuran model error: {e}")
        
    
    # koneksi ke rabbitmq dan membuat channel
    def connection(self):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host = self.host,
                    blocked_connection_timeout=300,
                    heartbeat=600 #heartbeat digunakan untuk menjaga koneksi tetap hidup
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue=self.queue, durable=True)
            channel.basic_qos(prefetch_count=1) # memastikan satu pesan diproses pada satu waktu
            return connection, channel
        except Exception as e:
            logging.error(f"Error connecting to RabbitMQ: {e}")
            raise e

    # menjalankan semua fungsi untuk clustering secara real-time
    def run_consumer(self):
        while True:
            connection, channel = self.connection()
            if connection is None or channel is None:
                logging.error("menunggu koneksi ulang ke RabbitMQ...")
                time.sleep(5)
                continue

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    self.proses_data_consumer(message)  # fungsi kamu tetap menerima 1 argumen
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logging.error(f"Error in callback processing: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(
                queue=self.queue,
                on_message_callback=callback,
                auto_ack=False
            )

            logging.info(f"Menunggu pesan di RabbitMQ '{self.queue}'...")
            try:
                channel.start_consuming()
            except Exception as e:
                logging.error(f"koneksi error: {e}")
            finally:
                if connection and not connection.is_closed:
                    connection.close()
                    logging.info("Koneksi RabbitMQ ditutup, mencoba koneksi ulang...")
                self.save_state()
                time.sleep(5)


    def run_consumer(self):
        while True:
            connection, channel = self.connection()
            if connection is None or channel is None:
                logging.error("menunggu koneksi ulang ke RabbitMQ...")
                time.sleep(5)
                continue

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    self.proses_data_consumer(message)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    logging.error(f"Error in callback processing: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=self.queue, on_message_callback=callback, auto_ack=False)


            # def callback(ch, method, properties, body):
            #     try:
            #         message = json.loads(body)
            #         self.proses_data_consumer(message)
            #         ch.basic_ack(delivery_tag=method.delivery_tag)
            #     except Exception as e:
            #         logging.error(f"Error in callback processing: {e}")
            #         ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)\
            
            # channel.basic_consume(
            #     queue=self.queue,
            #     on_message_callback=callback,
            #     auto_ack=False
            # )
            # channel.basic_consume(
            #     queue=self.queue,
            #     on_message_callback=lambda ch, method, properties, body: self.proses_data_consumer(json.loads(body)),
            #     auto_ack=False)

            logging.info(f"Menunggu pesan di RabbitMQ '{self.queue}'...")
            try:
                channel.start_consuming()
            except Exception as e:
                logging.error(f"koneksi error: {e}")
            finally:
                if connection and not connection.is_closed:
                    connection.close()
                    logging.info("Koneksi RabbitMQ ditutup, mencoba koneksi ulang...")
                self.save_state()
                time.sleep(5)

if __name__ == "__main__":
    consumer = modelConsumer(
    )
    consumer.run_consumer()
                