

1. ambil data yang masuk di dalam rabitMQ
2. lalukan clustering secara incremental dengan algoritma DBSTREAM
3. coba featursnya seleksi dulu, tidak semua masuk ke dalam proses clustering, tetap kirimkan semua datanya, yang diproses hanya featurs yang memiliki makna yang baik. baru nanti hasil clusternya gabungkan dengan data asli, atau data asli kasihkan label cluster yang telah terbentuk
4. 


A. yang peru dioptimasikan
1. Efisiensi Database, apakah memang semua element itu harus disimpan?
2. sebelum dilakukan proses penyimpanan untuk metric archive dan hasil clusteringnya, terlebih dahulu lakukan penggabungan HAC, baru simpan matric archive. jika memang matrics archieve harus disimpan dlu untuk pelacakan awal di database, maka simpan dulu sementara di dalam memori atau apa. namun, ingat, jika program dimatikan jangan sampai mengubah struktur dari awal.


B. Tampilan di dahboard akan memuat apa saja:
1. performa model, meliputi jumlah data yang diproses, matrix evaluasi, 
2. footprint, latency, memori dll
3. visualisasi hasil cluster, cluster distribution, cluster proportion
4. interpretasi dari hasil cluster, misal cluster 0 karakter bagaimana atau segmentasinya bagaimana
5. cluster profil yang terbuat. 

c. Pisahkan per fungsi atau metod dengan pemisahan
untuk database ambil dari  fungsi di file modelDB.py
1. setup_indexes
2. restore_full_state
3. save_full_state
4. save_model_snapshot

untuk model consumer modelConsumer.py
1. load_or_initialize_model
2. process_datapoint
3. is_already_processed
4. restore_model_state

untuk model evaluasi modelEvaluasi.py
1. evaluate_internal_metrics
3. compute_dunn_index
4. compute_cluster_distances

untuk model storefootprint: di file storagefootprint.py
1. track_storage_footprint

untuk management logging di file logging.py

induk dari proses adalah di dalam file modelConsumer.py di mana ini yang anantinya akan memproses OML data stream dan dengan memanfaatkan semua fungsi atau metod dari file-file yang dibuat, agar memudahhkan memanajemnet atau maintenec.


Cluster count & size distribution: tiba-tiba naik/ turun menandakan drift or new cluster.

Online silhouette approximated: compute silhouette with sample set (silhouette expensive; use sampling).

Inter / intra cluster distances: monitor mean intra-cluster distance and nearest neighbor inter-cluster distance ratio.

Cluster stability: measure how much centroids move per time unit (centroid drift).

Outlier rate: fraction of points not assigned (or in low-weight microclusters).

Distribution divergence: KL/JS divergence between feature distributions in successive windows.

Deteksi concept drift

Monitor metrik di atas; gunakan statistical tests:

CUSUM / Page-Hinkley on mean distance or outlier rate.

Sudden increase in number of microclusters or drop in mean weight → potential drift.

Saat drift terdeteksi: trigger retrain/rebuild window / increase decay rate so older clusters fade faster.

E. Latency & throughput monitoring

Metrik teknis penting: processing latency per event (p50/p95/p99), throughput (msgs/sec), memory usage (#microclusters, mem footprint), GC pauses.

Tools: Prometheus + Grafana. Ekspos metrik dari worker (histograms for latency, counters for events, gauges for #clusters).