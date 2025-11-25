import pickle
import logging
from river import preprocessing
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)

class StreamPreprocessor:
    
    def __init__(self):
        # Scalers untuk fitur numerik
        self.item_scaler = preprocessing.StandardScaler()
        self.category_scaler = preprocessing.StandardScaler()
        self.timestamp_scaler = preprocessing.MinMaxScaler()
        
        # Encoder untuk behavior (ordinal: pv < fav < cart < buy)

        self.behavior_mapping = {
            'pv': 0,
            'fav': 1,
            'cart': 2,
            'buy': 3
        }
        
        # State tracking
        self.n_samples_seen = 0
        
        logging.info("StreamPreprocessor initialized")
        
    def fit_transform_one(self, raw_data):
        """
        Learn dan transform satu data point
        """
        try:
            # Extract raw values
            item_id = raw_data['Item ID']
            category_id = raw_data['Category ID']
            behavior = raw_data['Behavior type']
            timestamp = raw_data['Timestamp']
            
            # 1. Encode behavior (ordinal)
            behavior_encoded = self.behavior_mapping.get(behavior, 0)

            # 2. Scale Item ID
            self.item_scaler.learn_one({'item': float(item_id)})  # Learn dulu
            item_scaled = self.item_scaler.transform_one({'item': float(item_id)})['item']  # Baru transform
            
            # 3. Scale Category ID
            self.category_scaler.learn_one({'category': float(category_id)})
            category_scaled = self.category_scaler.transform_one({'category': float(category_id)})['category']
            
            # 4. Scale Timestamp
            self.timestamp_scaler.learn_one({'timestamp': float(timestamp)})
            timestamp_scaled = self.timestamp_scaler.transform_one({'timestamp': float(timestamp)})['timestamp']
            
            # 5. Compile features
            features = [
                float(item_scaled),
                float(category_scaled),
                float(behavior_encoded),
                float(timestamp_scaled)
            ]
            
            self.n_samples_seen += 1
            
            return features
        
        except Exception as e:
            logging.error(f"Error in fit_transform_one: {e}")
            import traceback
            logging.error(traceback.format_exc()) 
            return [0.0, 0.0, 0.0, 0.0]    
        
    def transform_one(self, raw_data):
        try:
            item_id = raw_data['Item ID']
            category_id = raw_data['Category ID']
            behavior = raw_data['Behavior type']
            timestamp = raw_data['Timestamp']
            
            behavior_encoded = self.behavior_mapping.get(behavior, 0)
            
            # Transform only (no learning)
            item_scaled = self.item_scaler.transform_one({'item': item_id})['item']
            category_scaled = self.category_scaler.transform_one({'category': category_id})['category']
            timestamp_scaled = self.timestamp_scaler.transform_one({'timestamp': timestamp})['timestamp']
            
            features = [
                item_scaled,
                category_scaled,
                behavior_encoded,
                timestamp_scaled
            ]
            
            return features
        
        except Exception as e:
            logging.error(f"Error in transform_one: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def get_state(self):
        return {
            'n_samples_seen': self.n_samples_seen,
            'item_scaler_mean': getattr(self.item_scaler, 'mean', {}).get('item', 0),
            'category_scaler_mean': getattr(self.category_scaler, 'mean', {}).get('category', 0),
            'timestamp_scaler_min': getattr(self.timestamp_scaler, 'min', {}).get('timestamp', 0),
            'timestamp_scaler_max': getattr(self.timestamp_scaler, 'max', {}).get('timestamp', 1),
        }
    
    def save(self, path="models/preprocessor.pkl"):

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "wb") as f:
                pickle.dump(self, f)
            
            # Save state info sebagai JSON untuk debugging
            state_path = path.replace('.pkl', '_state.json')
            with open(state_path, 'w') as f:
                json.dump(self.get_state(), f, indent=2)
            
            logging.info(f"Preprocessor saved to {path}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to save preprocessor: {e}")
            return False
    
    @staticmethod
    def load(path="models/preprocessor.pkl"):
        try:
            with open(path, "rb") as f:
                preprocessor = pickle.load(f)
            
            logging.info(f"Preprocessor loaded from {path}")
            logging.info(f"   Samples seen: {preprocessor.n_samples_seen}")
            
            return preprocessor
        
        except FileNotFoundError:
            logging.warning(f" Preprocessor file not found: {path}")
            logging.info("   Creating new preprocessor...")
            return StreamPreprocessor()
        
        except Exception as e:
            logging.error(f"Failed to load preprocessor: {e}")
            return StreamPreprocessor()


# ===== UTILITY FUNCTIONS =====

def create_or_load_preprocessor(path="models/preprocessor.pkl"):
    return StreamPreprocessor.load(path)


# # ===== TESTING =====
# if __name__ == "__main__":
#     # Test preprocessor
#     print("="*60)
#     print("TESTING STREAM PREPROCESSOR")
#     print("="*60)
    
#     preprocessor = StreamPreprocessor()
    
#     # Test data
#     test_data = [
#         {'User ID': 100, 'Item ID': 5001, 'Category ID': 10, 'Behavior type': 'pv', 'Timestamp': 1000},
#         {'User ID': 101, 'Item ID': 5002, 'Category ID': 10, 'Behavior type': 'fav', 'Timestamp': 1100},
#         {'User ID': 102, 'Item ID': 5003, 'Category ID': 11, 'Behavior type': 'cart', 'Timestamp': 1200},
#         {'User ID': 103, 'Item ID': 5004, 'Category ID': 12, 'Behavior type': 'buy', 'Timestamp': 1300},
#     ]
    
#     print("\Processing test data:")
#     for i, data in enumerate(test_data):
#         features = preprocessor.fit_transform_one(data)
#         print(f"\nData {i+1}:")
#         print(f"  Raw: {data}")
#         print(f"  Features: {features}")
    
#     print("\nPreprocessor state:")
#     print(json.dumps(preprocessor.get_state(), indent=2))
    
#     # Test save/load
#     print("\Testing save/load...")
#     preprocessor.save("test_preprocessor.pkl")
    
#     loaded = StreamPreprocessor.load("test_preprocessor.pkl")
#     print(f"Loaded preprocessor with {loaded.n_samples_seen} samples seen")
    
#     # Test transform only (no learning)
#     print("\Testing transform_one (no learning):")
#     new_data = {'User ID': 104, 'Item ID': 5005, 'Category ID': 13, 'Behavior type': 'buy', 'Timestamp': 1400}
#     features = loaded.transform_one(new_data)
#     print(f"  Features: {features}")
    
#     print("\n" + "="*60)
#     print("TEST COMPLETE")
#     print("="*60)