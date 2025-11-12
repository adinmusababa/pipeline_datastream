import pika
import time
import pandas as pd
import numpy as np
from pymongo import MongoClient
import logging
import json
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler


class modelProducer:
    def __init__(self, bootstrap_servers="localhost", topic="data_stream"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic

        # collections untuk cekpoint
        self.checkpoint_collection = "checkpoint_producer"

        # logging
        # configure logging with timestamp and level
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def publish_data_rabitmq(self,
                             watch_mode=True):
        # setup koneksi ke RabitMQ
        try:
            self.logger.info("Menghubungkan ke RabitMQ...")
            client = MongoClient("mongodb://localhost:27017/")
            db = client["ecommerce_db"]
            collection = db["data_perilaku_pengguna_ecommerce"]

            # check last update dari checkpoint
            last_processed_id, prosses_count = self.get_checkpoint_update(db)

            # setup koneksi ke RabitMQ
            connection, channel = self.setup_rabitmq(
                host='localhost',
                queue=self.topic
            )
            if connection is None or channel is None:
                self.logger.error("Koneksi ke RabitMQ gagal. Menghentikan proses.")
                client.close()
                return

            batch_size = 1000
            total_published = prosses_count

            while True:
                # query resume dari last_processed_id
                if last_processed_id:
                    query = {
                        "_id": {"$gt": last_processed_id}
                    }
                    self.logger.info(f"Mengambil data dengan query: {query}")
                else:
                    query = {}
                    self.logger.info("Mengambil data dari awal.")
            
                # hitung data yang telah diproses
                remaining_count = collection.count_documents(query)
                if remaining_count == 0:
                    if not watch_mode:
                        self.logger.info("Tidak ada data baru untuk diproses. Menghentikan proses.")
                        self.mark_complete(db)
                        break
                    else:
                        self.logger.info("Tidak ada data baru. Menunggu data baru...")
                        time.sleep(10)
                        continue
                
                self.logger.info(f"Sisa data untuk diproses: {remaining_count}")

                # proses data dalam batch
                cursor = collection.find(query).sort("_id", 1)
                batch = []

                for document in cursor:
                    batch.append(document)
                    if len(batch) >= batch_size:
                        bublised_data = self.prosess_batch(batch, channel, db)
                        total_published += bublised_data

                        # update checkpoint
                        last_processed_id = batch[-1]["_id"]
                        self.save_checkpoint(db, last_processed_id, total_published)
                        
                        batch = []
                
                # proses sisa data dalam batch
                if batch:
                    publised = self.prosess_batch(batch, channel, db)
                    total_published += publised
                    last_processed_id = batch[-1]["_id"]
                    self.save_checkpoint(db, last_processed_id, total_published)
                    self.logger.info(f"Published {publised} messages | Total: {total_published}")
                
                # jika tidak dalam mode watch, hentikan proses setelah satu putaran
                if not watch_mode:
                    self.mark_complete(db)
                    break
                else:
                    time.sleep(10)
                
                connection.close()
                client.close()
                self.logger.info("Proses publish data selesai.")

        except KeyboardInterrupt:
            self.logger.error(f"proses dihentikan oleh user.")
        finally:
            try:
                connection.close()
            except:
                pass
            try:
                client.close()
            except:
                pass


    # fungsi untuk mendapatkan checkpoint terakhir
    def checkpoint_update(self,db):
        checkpoint = db[self.collections_point].find_one({"_id","last_processed"})
        if checkpoint:
            self.logger.info("Checkpoint ditemukan : {checkpoint['last_id]}")
            return checkpoint["last_id"], checkpoint["count"]
        else:
            self.logger.info("Tidak ada checkpoint ditemukan. Memulai dari awal.")
            return None, 0
        
    # fungsi untuk mengatur koneksi ke RabitMQ, 
    def setup_rabitmq(self, host="localhost", queue="data_stream"):
        # sekema kesalahan yang ditangani
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=host,
            ))
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)
            self.logger.info("kneksi ke RabitMQ berhasil.")
            return connection, channel
        except Exception as e:
            self.logger.error(f"Gagal menghubungkan ke RabitMQ: {e}")
            return None, None

    # fungsi proses batch data
    def prosess_batch(self, batch, channel, db):
        df = pd.DataFrame(batch)
        if df.empty:
            return 0

        try:
            published_count = 0

            # ENCODING behavior
            behavior_order = [['pv', 'fav', 'cart', 'buy']]
            ordinal_encoder = OrdinalEncoder(categories=behavior_order)
            df['behavior_encoded'] = ordinal_encoder.fit_transform(df[['Behavior type']])

            # SCALING fitur numerik
            id_features = ['User ID', 'Item ID', 'Category ID']
            standar_scaler = StandardScaler()
            df[id_features] = standar_scaler.fit_transform(df[id_features])

            minmax_scaler = MinMaxScaler()
            df['timestamp_scaled'] = minmax_scaler.fit_transform(df[['Timestamp']])

            # fitur untuk clustering
            features_for_clustering = [
                'User ID',
                'Item ID',
                'Category ID',
                'behavior_encoded',
                'timestamp_scaled'
            ]
            
            self.logger.info(f"Batch siap diproses: {len(df)} baris, kolom: {list(df.columns)}")
            
            self.logger.info(df[features_for_clustering].head(3).to_dict(orient="records"))            
            self.logger.info(f"Batch siap diproses: {len(df)} baris, kolom: {list(df.columns)}")
            self.logger.info(df[features_for_clustering].head(3).to_dict(orient="records"))
            
            # kirim ke RabbitMQ
            for _, row in df.iterrows():
                features = row[features_for_clustering].tolist()
                message = {
                    'features': features,
                    'timestamp': int(row['Timestamp']),
                    'metadata': {
                        'original_id': str(row.get('_id', '')),
                        'behavior_type': row.get('Behavior type', ''),
                        'User ID': int(row['User ID']),
                        'Item ID': int(row['Item ID']),
                        'Category ID': int(row['Category ID']),
                        'Timestamp': int(row['Timestamp'])
                    },
                    'raw_data': {
                        'User ID': int(row['User ID']),
                        'Item ID': int(row['Item ID']),
                        'Category ID': int(row['Category ID']),
                        'Behavior type': row.get('Behavior type', ''),
                        'Timestamp': int(row['Timestamp'])
                    }
                }
                if published_count < 3:  # hanya log 3 pesan pertama biar gak banjir log
                    self.logger.info(f"Pesan #{published_count+1} -> {json.dumps(message, indent=2)}")
                    

                channel.basic_publish(
                    exchange='',
                    routing_key=self.topic,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )

                published_count += 1
                time.sleep(0.01)

            self.logger.info(f"Total pesan terkirim dari batch ini: {published_count}")
            return published_count

        except Exception as e:
            self.logger.error(f"Gagal memproses batch: {e}")
            return 0

    
    def save_checkpoint(self, db, last_id, count):
        db[self.checkpoint_collection].update_one(
            {"_id": "last_processed"},
            {
                "$set": {
                    "last_id": last_id,
                    "count": count,
                    "timestamp": int(time.time()),
                    "status": "in_progress"
            }
            },
            upsert=True
        )

    def mark_complete(self, db):
        db[self.checkpoint_collection].update_one(
            {"_id": "last_processed"},
            {
                "$set": {
                    "status": "complete",
                    "completed_at": int(time.time())
                }
            }
        )
    
    def get_checkpoint_update(self, db):
        checkpoint = db[self.checkpoint_collection].find_one({"_id": "last_processed"})
        if checkpoint:
            self.logger.info(f"Checkpoint ditemukan: {checkpoint['last_id']}")
            return checkpoint["last_id"], checkpoint["count"]
        else:
            self.logger.info("Tidak ada checkpoint ditemukan. Memulai dari awal.")
            return None, 0
        
if __name__ == "__main__":
    producer = modelProducer(
        bootstrap_servers="localhost",
        topic="data_stream"
    )
    producer.publish_data_rabitmq(
        watch_mode=True
    )