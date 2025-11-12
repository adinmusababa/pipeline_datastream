1. ambil data yang masuk di dalam rabitMQ
2. lalukan clustering secara incremental dengan algoritma DBSTREAM
3. 

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