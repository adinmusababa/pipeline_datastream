# Changelog: User ID Handling dalam Data Streaming

## Ringkasan Perubahan
Skenario streaming data telah diubah untuk memisahkan `User ID` dari proses clustering OML (Online Machine Learning). User ID sekarang hanya digunakan untuk:
- Penyimpanan di database (untuk analisis dan audit)
- Penampilan di frontend/dashboard (untuk identifikasi user)

User ID **tidak lagi** digunakan dalam proses clustering, hanya variabel yang relevan untuk segmentasi perilaku.

---

## Detail Perubahan

### 1. **Producer.py** (`backend/producer.py`)
**Fungsi: `prosess_batch()`**

#### Sebelum:
```python
raw_data = {
    'User ID': document.get('User ID'),          # ❌ Disertakan dalam preprocessing
    'Item ID': document.get('Item ID'),
    'Category ID': document.get('Category ID'),
    'Behavior type': document.get('Behavior type'),
    'Timestamp': document.get('Timestamp')
}

features = self.preprocessor.fit_transform_one(raw_data)  # User ID di-transform

message = {
    'metadata': {
        'User ID': metadata.get('User ID'),       # ❌ Tidak konsisten
        ...
    }
}
```

#### Sesudah:
```python
# Extract User ID terpisah SEBELUM preprocessing
user_id = document.get('User ID')

# Raw data untuk preprocessing (TANPA User ID)
raw_data = {
    'Item ID': document.get('Item ID'),
    'Category ID': document.get('Category ID'),
    'Behavior type': document.get('Behavior type'),
    'Timestamp': document.get('Timestamp')
}

# Features hanya dari variabel yang relevan untuk clustering
features = self.preprocessor.fit_transform_one(raw_data)

message = {
    'features': features,                    # ✅ Hanya non-User ID features
    'metadata': {
        'user_id': user_id,                 # ✅ User ID untuk database/frontend
        'behavior_type': document.get('Behavior type'),
        'Item ID': int(document['Item ID']),
        'Category ID': int(document['Category ID']),
        'Timestamp': int(document['Timestamp'])
    },
    'raw_data': raw_data                    # ✅ Raw data tanpa User ID
}
```

**Keuntungan:**
- User ID tidak mempengaruhi proses clustering
- Features lebih fokus pada perilaku user (behavior, item, category)
- Konsistensi metadata

---

### 2. **Consumer.py** (`backend/consumer.py`)
**Fungsi: `process_datapoint()`**

#### Sebelum:
```python
cluster_data = {
    "features": features,
    "raw_data": raw_data,
    "cluster_id": int(cluster_id),
    "User ID": metadata.get('User ID'),          # ❌ Inconsistent naming
    "Item ID": metadata.get('Item ID'),
    ...
}
```

#### Sesudah:
```python
cluster_data = {
    "features": features,                       # ✅ Hanya non-User ID features
    "raw_data": raw_data,
    "cluster_id": int(cluster_id),
    "user_id": metadata.get('user_id'),        # ✅ Konsisten dengan naming dari producer
    "Item ID": metadata.get('Item ID'),
    "Category ID": metadata.get('Category ID'),
    "Behavior type": metadata.get('behavior_type'),
    "original_id": original_id,
    "processing_counter": self.counter
}
```

**Perubahan:**
- User ID diambil dari `metadata['user_id']` yang dikirim producer
- User ID disimpan ke database dengan nama field `user_id` (konsisten)
- Features dalam clustering tetap tidak mengandung User ID

---

### 3. **Frontend.py** (`frontend/frontend.py`)
**Fungsi: `load_cluster_data()`**

#### Sebelum:
```python
def load_cluster_data(limit=10000):
    """Load data hasil clustering dari database"""
    try:
        db = get_database()
        docs = list(db.clusters.find().sort("timestamp", -1).limit(limit))
        
        if not docs:
            return None
        
        return pd.DataFrame(docs)  # ❌ Field 'user_id' tidak di-rename
```

#### Sesudah:
```python
def load_cluster_data(limit=10000):
    """Load data hasil clustering dari database"""
    try:
        db = get_database()
        docs = list(db.clusters.find().sort("timestamp", -1).limit(limit))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        
        # ✅ Rename field untuk konsistensi dengan analisis
        if 'user_id' in df.columns and 'User ID' not in df.columns:
            df['User ID'] = df['user_id']
        
        return df
```

**Keuntungan:**
- Frontend secara otomatis memiliki kolom `User ID` yang dapat ditampilkan
- Tidak ada kolom kosong untuk User ID di visualisasi
- Kompatibel dengan existing visualization code

---

## Data Flow Diagram

```
┌─────────────────────────┐
│  Raw Data (MongoDB)     │
│ - User ID               │
│ - Item ID               │
│ - Category ID           │
│ - Behavior Type         │
│ - Timestamp             │
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────┐
│    PRODUCER.py                     │
│  - Extract User ID (SEPARATE)      │
│  - Preprocess non-User ID data     │
│  - Create features (without User ID)│
│  - Send message with:              │
│    * features (for OML)            │
│    * metadata.user_id (for DB/FE)  │
└────────────┬────────────────────────┘
             │
             ▼
        [RabbitMQ]
             │
             ▼
┌────────────────────────────────────┐
│    CONSUMER.py                     │
│  - Receive message                 │
│  - Extract features (OML process)  │
│  - Extract user_id from metadata   │
│  - Cluster using features ONLY     │
│  - Save to DB:                     │
│    * features                      │
│    * cluster_id                    │
│    * user_id                       │
└────────────┬────────────────────────┘
             │
             ▼
    [MongoDB: clusters]
             │
             ▼
┌────────────────────────────────────┐
│    FRONTEND.py                     │
│  - Load from DB                    │
│  - Rename: user_id → User ID       │
│  - Display visualization with      │
│    User ID column (NOT EMPTY!)     │
└────────────────────────────────────┘
```

---

## Testing & Validation

### Checklist:
- [ ] Producer berjalan tanpa error
- [ ] Consumer menerima data dengan `metadata.user_id`
- [ ] Database menyimpan kolom `user_id` (bukan `User ID`)
- [ ] Frontend menampilkan kolom `User ID` di hover_data dan tabel detail
- [ ] Clustering menggunakan hanya non-User ID features
- [ ] User ID tidak kosong di dashboard

### Monitoring:
```python
# Di Consumer
self.logger.info(f"User ID in message: {metadata.get('user_id')}")
self.logger.info(f"Saved cluster data user_id: {cluster_data.get('user_id')}")

# Di Frontend
print(df[['User ID', 'cluster_id']].head())  # Pastikan tidak ada NaN
```

---

## API Changes

### Message Format (Producer → Consumer)

**Lama:**
```json
{
  "features": [...],
  "metadata": {
    "User ID": "U123",
    "Item ID": 456,
    "Category ID": 789
  }
}
```

**Baru:**
```json
{
  "features": [...],
  "metadata": {
    "user_id": "U123",
    "Item ID": 456,
    "Category ID": 789,
    "behavior_type": "view"
  }
}
```

### Database Schema (clusters collection)

**Lama:**
```json
{
  "_id": ObjectId,
  "features": [...],
  "cluster_id": 0,
  "User ID": "U123",           // Konsistensi nama field
  "Item ID": 456,
  "timestamp": ISODate
}
```

**Baru:**
```json
{
  "_id": ObjectId,
  "features": [...],
  "cluster_id": 0,
  "user_id": "U123",           // Naming konsisten: snake_case
  "Item ID": 456,
  "Category ID": 789,
  "Behavior type": "view",
  "timestamp": ISODate
}
```

---

## Catatan Penting

1. **Backward Compatibility:** Data lama dengan field `User ID` harus dimigrasikan atau disesuaikan
2. **Preprocessing:** Pastikan `modelPreprocessing.py` tidak mencakup User ID
3. **Database Migration:** Pertimbangkan untuk menjalankan script migrasi jika ada data existing
4. **Logging:** Tambahkan log untuk memastikan User ID ditangani dengan benar di setiap tahap

---

## Manfaat Perubahan

✅ **Separation of Concerns:** User ID hanya untuk identifikasi, bukan untuk clustering  
✅ **Better Data Quality:** Hanya relevan features yang diproses  
✅ **Cleaner Results:** Cluster berdasarkan perilaku, bukan ID  
✅ **Frontend Improvement:** User ID selalu tersedia di dashboard  
✅ **Consistency:** Naming convention yang konsisten (snake_case untuk DB)  

---

Dibuat: 28 Desember 2025
