import pika
import time
import pandas as pd
import numpy as np
from pymongo import MongoClient
import logging
import json
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
import sys

from modelPreprocessing import StreamPreprocessor

class modelProducer:
    def __init__(self, bootstrap_servers="localhost", topic="data_stream"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.preprocessor = StreamPreprocessor.load("models/preprocessor.pkl")
    
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

                # # proses data dalam batch
                # cursor = collection.find(query).sort("_id", 1)
                # batch = []

                # for document in cursor:
                #     batch.append(document)
                #     if len(batch) >= batch_size:
                #         bublised_data = self.prosess_batch(batch, channel, db)
                #         total_published += bublised_data

                #         # update checkpoint
                #         last_processed_id = batch[-1]["_id"]
                #         self.save_checkpoint(db, last_processed_id, total_published)
                        
                #         batch = []
                cursor = collection.find(query).sort("_id", 1)
                batch = []

                for document in cursor:
                    batch.append(document)
                    
                    if len(batch) >= batch_size:
                        # Process batch dengan reconnection handling
                        published_data = self.prosess_batch(batch, channel, db)
                        total_published += published_data
                        
                        # Check if connection closed during processing
                        if published_data < len(batch):
                            self.logger.warning(f"⚠️  Incomplete batch ({published_data}/{len(batch)}), reconnecting...")
                            
                            # Close old connection
                            try:
                                connection.close()
                            except:
                                pass
                            
                            # Reconnect
                            connection, channel = self.setup_rabitmq(host='localhost', queue=self.topic)
                            if connection is None or channel is None:
                                self.logger.error("Reconnection failed!")
                                time.sleep(5)
                                continue
                            
                            self.logger.info("Reconnected to RabbitMQ")
                        
                        # Update checkpoint
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
            self.cleanup()    
            connection.close()
            client.close()
            self.logger.info("Proses publish data selesai.")

        except KeyboardInterrupt:
            self.logger.error(f"proses dihentikan oleh user.")
        finally:
            self.cleanup()
            try:
                connection.close()
            except:
                pass
            try:
                client.close()
            except:
                pass


    # fungsi untuk mendapatkan checkpoint terakhir 
    def checkpoint_update(self, db):
        checkpoint = db[self.checkpoint_collection].find_one({"_id": "last_processed"})
        
        if checkpoint:
            self.logger.info(f"Checkpoint ditemukan: {checkpoint.get('last_id')}")
            return checkpoint.get("last_id"), checkpoint.get("count", 0)
        else:
            self.logger.info("Tidak ada checkpoint ditemukan. Memulai dari awal.")
            return None, 0

        
    # fungsi untuk mengatur koneksi ke RabitMQ, 
    def setup_rabitmq(self, host="localhost", queue="data_stream"):

        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=host,
                    heartbeat=600,  # Heartbeat every 10 minutes
                    blocked_connection_timeout=300,  # 5 minutes timeout
                    connection_attempts=3,  # Retry 3x jika gagal
                    retry_delay=2  # Wait 2s between retries
                )
            )
            channel = connection.channel()
            channel.queue_declare(queue=queue, durable=True)

            channel.basic_qos(prefetch_count=1)
            
            self.logger.info(" Koneksi ke RabbitMQ berhasil.")
            return connection, channel
            
        except Exception as e:
            self.logger.error(f" Gagal menghubungkan ke RabbitMQ: {e}")
            return None, None\
            
    # fungsi proses batch data
    def prosess_batch(self, batch, channel, db):
        """Process batch dengan connection error handling"""
        if not batch:
            return 0

        try:
            published_count = 0
            
            self.logger.info(f" Processing batch: {len(batch)} documents")
            
            for idx, document in enumerate(batch):
                try:
                    # Extract raw data
                    raw_data = {
                        'User ID': document.get('User ID'),
                        'Item ID': document.get('Item ID'),
                        'Category ID': document.get('Category ID'),
                        'Behavior type': document.get('Behavior type'),
                        'Timestamp': document.get('Timestamp')
                    }
                    
                    # Streaming preprocessing
                    features = self.preprocessor.fit_transform_one(raw_data)
                    
                    # Check if features are valid (not all zeros)
                    if all(f == 0.0 for f in features):
                        self.logger.warning(f"All-zero features at idx {idx}, skipping...")
                        continue
                    
                    # Build message
                    message = {
                        'features': features,
                        'timestamp': int(document['Timestamp']),
                        'metadata': {
                            'original_id': str(document.get('_id', '')),
                            'behavior_type': document.get('Behavior type', ''),
                            'Item ID': int(document['Item ID']),
                            'Category ID': int(document['Category ID']),
                            'Timestamp': int(document['Timestamp'])
                        },
                        'raw_data': raw_data
                    }
                    
                    # Log first 3 messages
                    if published_count < 3:
                        self.logger.info(f"   Message #{published_count+1}:")
                        self.logger.info(f"   Raw: {raw_data}")
                        self.logger.info(f"   Features: {features}")
                    
                    # Publish dengan error handling
                    try:
                        channel.basic_publish(
                            exchange='',
                            routing_key=self.topic,
                            body=json.dumps(message),
                            properties=pika.BasicProperties(delivery_mode=2)
                        )
                        published_count += 1
                        
                    except pika.exceptions.ConnectionClosed:
                        self.logger.error("RabbitMQ connection closed!")
                        # Return count untuk trigger reconnect
                        return published_count
                    
                    time.sleep(0.01)
                
                except Exception as e:
                    self.logger.error(f"Error processing document {idx}: {e}")
                    continue
            
            self.logger.info(f"Batch complete: {published_count}/{len(batch)} messages published")
            
            # Save preprocessor state
            if published_count > 0:
                self.preprocessor.save("models/preprocessor.pkl")
            
            return published_count
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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


    def cleanup(self):
        """
        Cleanup saat producer selesai
        """
        try:
            # Save final preprocessor state
            self.preprocessor.save("models/preprocessor.pkl")
            self.logger.info("Final preprocessor state saved")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


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