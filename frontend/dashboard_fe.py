import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import plotly.express as px
# from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Konfigurasi halaman
st.set_page_config(page_title=" Online DBSCAN Monitoring", layout="wide")

# CSS untuk styling
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .business-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .segment-card {
        border-left: 4px solid #3b82f6;
        padding: 15px;
        background: #f8fafc;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar: Kontrol Visualisasi
with st.sidebar:
    st.header("Kontrol Visualisasi")
    max_data_points = st.number_input("Max Data untuk Visualisasi", 1000, 50000, 10000, 1000)
    point_size = st.slider("Ukuran Titik", 2, 10, 3)
    show_labels = st.checkbox("Tampilkan Label Cluster", value=True)
    # show_centroids = st.checkbox("Tampilkan Centroid", value=False)

# Fungsi koneksi MongoDB
@st.cache_resource
def get_db():
    client = MongoClient("mongodb://localhost:27017/")
    return client["clustering_datastreaming"]

# Load metrik evaluasi dari MongoDB
# @st.cache_data(ttl=15)
def load_metrics():
    db = get_db()
    doc = db.metrics.find_one({"_id": "latest"})
    if not doc:
        return None
    return {
        'total_data': doc.get('total_data', 0),
        'active_clusters': doc.get('active_clusters', 0),
        'silhouette': doc.get('silhouette', 0),
        'davies_bouldin': doc.get('davies_bouldin', 0),
        'timestamp': doc.get('timestamp')
    }

# Load semua data cluster dari database
# @st.cache_data(ttl=15)
def load_all_cluster_data(limit=10000):
    """
    Load semua data cluster yang sudah final (termasuk hasil merge HAC)
    """
    try:
        db = get_db()
        # Ambil semua data dari database, bukan buffer
        docs = list(db.clusters.find().sort("timestamp", -1).limit(limit))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        # st.info(f"Loaded {len(df)} data points from database (real-time cluster results)")
        return df
    
    except Exception as e:
        st.error(f"Error loading cluster data: {e}")
        return None

def calculate_processing_latency(sample_size=1000):
    """
    Hitung rata-rata waktu processing per data point
    Berdasarkan timestamp antar data yang masuk
    """
    try:
        db = get_db()
        # Ambil sample data terbaru dengan timestamp
        docs = list(db.clusters.find(
            {},
            {"timestamp": 1}
        ).sort("timestamp", -1).limit(sample_size))
        
        if len(docs) < 2:
            return None
        
        df = pd.DataFrame(docs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Hitung selisih waktu antar data point
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # Filter outliers (> 10 detik dianggap anomali)
        df_clean = df[df['time_diff'] <= 10]
        
        if len(df_clean) < 2:
            return None
        
        # Statistik
        avg_latency = df_clean['time_diff'].mean()
        median_latency = df_clean['time_diff'].median()
        p95_latency = df_clean['time_diff'].quantile(0.95)
        min_latency = df_clean['time_diff'].min()
        max_latency = df_clean['time_diff'].max()
        
        # Throughput (data per detik)
        throughput = 1 / avg_latency if avg_latency > 0 else 0
        
        return {
            'avg_latency_ms': avg_latency * 1000,  # konversi ke ms
            'median_latency_ms': median_latency * 1000,
            'p95_latency_ms': p95_latency * 1000,
            'min_latency_ms': min_latency * 1000,
            'max_latency_ms': max_latency * 1000,
            'throughput': throughput,
            'sample_size': len(df_clean)
        }
    
    except Exception as e:
        st.error(f"Error calculating latency: {e}")
        return None


# Reduksi Dimensi dengan PCA
# @st.cache_data(ttl=15)
def apply_pca_to_clusters(df):
    """
    Reduksi dimensi menggunakan PCA untuk visualisasi 2D
    Menggunakan SEMUA variabel yang digunakan saat clustering
    """
    try:
        if 'features' not in df.columns:
            st.error("Kolom 'features' tidak ditemukan")
            return None
        
        # Extract features
        features_list = df['features'].tolist()
        features_array = np.array(features_list)
        
        # Validasi
        if features_array.shape[1] < 2:
            st.error("Fitur tidak cukup untuk PCA")
            return None
        
        # Standardisasi
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # PCA ke 2 dimensi
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        # Tambahkan ke dataframe
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        st.info(f"PCA Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, Total={sum(explained_var):.2%}")
        
        return df
    
    except Exception as e:
        st.error(f"Error in PCA: {e}")
        return None

# Interpretasi Bisnis Otomatis
def interpret_cluster_business(cluster_data):
    """
    Interpretasi otomatis untuk setiap cluster berdasarkan behavior dominan
    """
    interpretations = []
    
    for cluster_id in sorted(cluster_data['cluster_id'].unique()):
        cluster_subset = cluster_data[cluster_data['cluster_id'] == cluster_id]
        size = len(cluster_subset)
        
        # Behavior distribution
        behavior_counts = Counter(cluster_subset['Behavior type'].dropna())
        total_behavior = sum(behavior_counts.values())
        
        if total_behavior == 0:
            continue
        
        behavior_pct = {
            'pv': (behavior_counts.get('pv', 0) / total_behavior) * 100,
            'fav': (behavior_counts.get('fav', 0) / total_behavior) * 100,
            'cart': (behavior_counts.get('cart', 0) / total_behavior) * 100,
            'buy': (behavior_counts.get('buy', 0) / total_behavior) * 100
        }
        
        # Klasifikasi segmen
        segment = classify_segment(behavior_pct, cluster_subset, size, len(cluster_data))
        segment['cluster_id'] = int(cluster_id)
        segment['size'] = size
        segment['behavior_distribution'] = behavior_pct
        
        interpretations.append(segment)
    
    return interpretations

def classify_segment(behavior_pct, cluster_subset, size, total_size):
    """
    Klasifikasi segmen bisnis berdasarkan behavior pattern
    """
    buy_rate = behavior_pct['buy']
    cart_rate = behavior_pct['cart']
    fav_rate = behavior_pct['fav']
    pv_rate = behavior_pct['pv']
    
    # Hitung metrik tambahan
    unique_users = cluster_subset['User ID'].nunique() if 'User ID' in cluster_subset.columns else 0
    unique_items = cluster_subset['Item ID'].nunique() if 'Item ID' in cluster_subset.columns else 0
    unique_categories = cluster_subset['Category ID'].nunique() if 'Category ID' in cluster_subset.columns else 0
    
    # High-Value Buyers
    if buy_rate > 15:
        return {
            'segment_name': 'High-Value Buyers',
            'description': f'Segmen dengan konversi pembelian tinggi ({buy_rate:.1f}%). Pengguna aktif yang sering menyelesaikan transaksi.',
            'characteristics': [
                f'• {unique_users} unique users dengan repeat purchase tinggi',
                f'• Membeli dari {unique_categories} kategori berbeda',
                f'• Buy rate {buy_rate:.1f}% (target: >15%)',
                f'• Mewakili {(size/total_size)*100:.1f}% dari total user'
            ],
            'recommendation': ' **Strategi**: Loyalty program, cross-sell premium products, VIP benefits, early access, personalized offers',
            'priority': 'HIGH',
            'color': '#ef4444'
        }
    
    # Cart Abandoners
    elif cart_rate > 20:
        return {
            'segment_name': 'Cart Abandoners',
            'description': f'Banyak menambahkan ke keranjang ({cart_rate:.1f}%) tapi jarang checkout. Potensi konversi tinggi.',
            'characteristics': [
                f'• {unique_users} users dengan abandoned carts',
                f'• Tertarik pada {unique_items} produk berbeda',
                f'• Cart rate {cart_rate:.1f}%, Buy rate hanya {buy_rate:.1f}%',
                f'• Conversion gap: {cart_rate - buy_rate:.1f}%'
            ],
            'recommendation': '💡 **Strategi**: Email reminder otomatis, discount incentive, free shipping, urgency tactics, retargeting ads',
            'priority': 'HIGH',
            'color': '#f59e0b'
        }
    
    # Wishlist Collectors
    elif fav_rate > 20:
        return {
            'segment_name': 'Wishlist Collectors',
            'description': f'Suka menyimpan favorit ({fav_rate:.1f}%) tapi belum membeli. Fase pertimbangan.',
            'characteristics': [
                f'• {unique_users} users sedang compare products',
                f'• {unique_items} items di wishlist',
                f'• Favorite rate {fav_rate:.1f}%',
                f'• Perlu trigger untuk convert'
            ],
            'recommendation': '**Strategi**: Price drop alerts, limited-time offers, social proof, review highlights, installment options',
            'priority': 'MEDIUM',
            'color': '#8b5cf6'
        }
    
    # Window Shoppers
    elif pv_rate > 60:
        return {
            'segment_name': 'Window Shoppers',
            'description': f'Dominan browsing ({pv_rate:.1f}%). Engagement rendah, perlu aktivasi.',
            'characteristics': [
                f'• {unique_users} passive browsers',
                f'• Melihat {unique_items} produk tanpa aksi',
                f'• View rate {pv_rate:.1f}%',
                f'• Conversion rate sangat rendah: {buy_rate:.1f}%'
            ],
            'recommendation': '**Strategi**: Personalized recommendations, content marketing, interactive quizzes, seasonal campaigns, retargeting',
            'priority': 'LOW',
            'color': '#10b981'
        }
    
    # Balanced Engagers
    else:
        return {
            'segment_name': 'Balanced Engagers',
            'description': f'Pola seimbang dengan buy rate {buy_rate:.1f}%. Regular users dengan growth potential.',
            'characteristics': [
                f'• {unique_users} regular customers',
                f'• Balanced behavior pattern',
                f'• Buy: {buy_rate:.1f}%, Cart: {cart_rate:.1f}%, Fav: {fav_rate:.1f}%',
                f'• Stable customer base'
            ],
            'recommendation': ' **Strategi**: A/B testing, referral programs, email nurturing, community building, gamification',
            'priority': 'MEDIUM',
            'color': '#3b82f6'
        }

# Visualisasi Cluster 2D dengan PCA
def create_pca_cluster_plot(df):
    """
    Visualisasi scatter plot 2D menggunakan hasil PCA
    Menampilkan SEMUA data cluster 
    """
    try:
        if 'pca1' not in df.columns or 'pca2' not in df.columns:
            st.error("PCA belum diterapkan pada data")
            return None
        
        fig = px.scatter(
            df, 
            x='pca1', 
            y='pca2', 
            color='cluster_id',
            hover_data=['User ID', 'Item ID', 'Behavior type', 'Category ID'] if 'User ID' in df.columns else None,
            title=f"Cluster Visualization (PCA 2D) - {len(df):,} Data Points",
            labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'},
            color_continuous_scale='Viridis',
            opacity=0.6,
            size=[point_size] * len(df),
            size_max=point_size
        )
        
        fig.update_layout(
            height=600,
            showlegend=show_labels,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        
        # # Tambahkan centroids jika diminta
        # if show_centroids:
        #     centroids = df.groupby('cluster_id')[['pca1', 'pca2']].mean().reset_index()
        #     fig.add_trace(go.Scatter(
        #         x=centroids['pca1'],
        #         y=centroids['pca2'],
        #         mode='markers',
        #         marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='black')),
        #         name='Centroids',
        #         showlegend=True
        #     ))
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        return None
        
# Status Pipeline 
def render_pipeline_status():
    st.markdown("### Status Pipeline Monitoring")
    
    try:
        metrics = load_metrics()
        if not metrics:
            st.warning("Status pipeline belum tersedia.")
            return
        
        latency_data = calculate_processing_latency(sample_size=1000)

        # Status dan Timestamp
        col1, col2, col3 = st.columns(3)
        
        with col1:
            local_time = metrics['timestamp']
            if isinstance(local_time, str):
                local_time = datetime.fromisoformat(local_time.replace('Z', '+00:00'))
            st.metric("Total Data Processed", f"{metrics['total_data']:,}")
        
        with col2:
            st.metric("Active Clusters", metrics['active_clusters'])
        
        with col3:
            st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
        
        # Tambahan metrik
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin']:.4f}")
            if metrics['davies_bouldin'] < 1:
                st.success("Excellent cluster quality")
            elif metrics['davies_bouldin'] < 2:
                st.warning("Acceptable quality")
            else:
                st.error("Poor cluster separation")
        
        with col5:
            last_eval = metrics['timestamp']
            if isinstance(last_eval, str):
                last_eval = datetime.fromisoformat(last_eval.replace('Z', '+00:00'))
            time_diff = datetime.utcnow() - last_eval
            st.metric("Last Update", f"{int(time_diff.total_seconds())}s ago")
        
        with col6:
            if latency_data:
                st.metric("waktu pemrosesan rata-rata data", f"{latency_data['avg_latency_ms']:.1f} ms")
                st.metric("Throughput", f"{latency_data['throughput']:.1f} data/s")
            else:
                st.info("Latency data not available")
    
    except Exception as e:
        st.error(f"Error: {e}")


# @st.cache_data(ttl=30)
def load_cluster_growth_data():
    """
    Load data pertumbuhan cluster dari metrics_archive
    """
    try:
        db = get_db()
        # Ambil data metrics archive (sorted by timestamp)
        docs = list(db.metrics_archive.find().sort("timestamp", 1))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading cluster growth data: {e}")
        return None


# @st.cache_data(ttl=30)
def load_merge_history():
    """
    Load riwayat merge cluster
    """
    try:
        db = get_db()
        docs = list(db.merge_history.find().sort("merge_timestamp", 1))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        df['merge_timestamp'] = pd.to_datetime(df['merge_timestamp'])
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading merge history: {e}")
        return None


def render_cluster_growth_monitoring(metrics_df, merge_df):
    """
    Render monitoring pertumbuhan cluster & evaluasi
    """
    st.subheader("📈 Cluster Growth Monitoring")
    
    if metrics_df is None or metrics_df.empty:
        st.warning("⚠️ Belum ada data monitoring tersedia")
        return
    
    # === 1. CLUSTER GROWTH OVER TIME ===
    st.markdown("#### 🔢 Pertumbuhan Jumlah Cluster")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Line chart: Active clusters over time
        fig_cluster_growth = go.Figure()
        
        fig_cluster_growth.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['active_clusters'],
            mode='lines+markers',
            name='Active Clusters',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=6)
        ))
        
        # Tambahkan garis trend
        if len(metrics_df) > 5:
            z = np.polyfit(range(len(metrics_df)), metrics_df['active_clusters'], 1)
            p = np.poly1d(z)
            fig_cluster_growth.add_trace(go.Scatter(
                x=metrics_df['timestamp'],
                y=p(range(len(metrics_df))),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash', width=1)
            ))
        
        fig_cluster_growth.update_layout(
            title="Jumlah Cluster Aktif Over Time",
            xaxis_title="Waktu",
            yaxis_title="Jumlah Cluster",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cluster_growth, use_container_width=True)
        
        # Statistik cluster growth
        st.markdown("**📊 Statistik Pertumbuhan:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Min Clusters", int(metrics_df['active_clusters'].min()))
        with col_b:
            st.metric("Max Clusters", int(metrics_df['active_clusters'].max()))
        with col_c:
            st.metric("Avg Clusters", f"{metrics_df['active_clusters'].mean():.1f}")
    
    with col2:
        # Bar chart: Cluster growth per data processed
        fig_cluster_per_data = go.Figure()
        
        # Hitung cluster per 1000 data
        metrics_df['cluster_per_1k'] = (metrics_df['active_clusters'] / metrics_df['total_data']) * 1000
        
        fig_cluster_per_data.add_trace(go.Bar(
            x=metrics_df['total_data'],
            y=metrics_df['cluster_per_1k'],
            marker=dict(color='#10b981'),
            name='Clusters per 1K data'
        ))
        
        fig_cluster_per_data.update_layout(
            title="Cluster Density (per 1000 data)",
            xaxis_title="Total Data Processed",
            yaxis_title="Clusters / 1000 data",
            height=400
        )
        
        st.plotly_chart(fig_cluster_per_data, use_container_width=True)
        
        # Warning jika over-segmentation
        latest_density = metrics_df['cluster_per_1k'].iloc[-1]
        if latest_density > 100:
            st.error(f"⚠️ **Over-segmentation detected!** {latest_density:.1f} clusters per 1K data")
        elif latest_density > 50:
            st.warning(f"⚠️ **Watch out!** {latest_density:.1f} clusters per 1K data")
        else:
            st.success(f"✅ **Good density**: {latest_density:.1f} clusters per 1K data")
    
    st.markdown("---")
    
    # === 2. EVALUATION METRICS OVER TIME ===
    st.markdown("#### 📉 Evaluasi Metrik Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Silhouette Score over time
        fig_silhouette = go.Figure()
        
        fig_silhouette.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['silhouette'],
            mode='lines+markers',
            name='Silhouette',
            line=dict(color='#8b5cf6', width=2),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ))
        
        # Threshold lines
        fig_silhouette.add_hline(y=0.5, line_dash="dash", line_color="green", 
                                annotation_text="Good (>0.5)", annotation_position="right")
        fig_silhouette.add_hline(y=0.25, line_dash="dash", line_color="orange", 
                                annotation_text="Fair (>0.25)", annotation_position="right")
        
        fig_silhouette.update_layout(
            title="Silhouette Score Over Time",
            xaxis_title="Waktu",
            yaxis_title="Silhouette Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_silhouette, use_container_width=True)
        
        # Statistik Silhouette
        st.markdown("**📊 Silhouette Statistics:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Min", f"{metrics_df['silhouette'].min():.3f}")
        with col_b:
            st.metric("Max", f"{metrics_df['silhouette'].max():.3f}")
        with col_c:
            st.metric("Avg", f"{metrics_df['silhouette'].mean():.3f}")
    
    with col2:
        # Davies-Bouldin Index over time
        fig_dbi = go.Figure()
        
        fig_dbi.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['davies_bouldin'],
            mode='lines+markers',
            name='DBI',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        # Threshold lines
        fig_dbi.add_hline(y=1.0, line_dash="dash", line_color="green", 
                         annotation_text="Excellent (<1.0)", annotation_position="right")
        fig_dbi.add_hline(y=2.0, line_dash="dash", line_color="orange", 
                         annotation_text="Acceptable (<2.0)", annotation_position="right")
        
        fig_dbi.update_layout(
            title="Davies-Bouldin Index Over Time",
            xaxis_title="Waktu",
            yaxis_title="DBI (lower is better)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_dbi, use_container_width=True)
        
        # Statistik DBI
        st.markdown("**📊 DBI Statistics:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Min", f"{metrics_df['davies_bouldin'].min():.3f}")
        with col_b:
            st.metric("Max", f"{metrics_df['davies_bouldin'].max():.3f}")
        with col_c:
            st.metric("Avg", f"{metrics_df['davies_bouldin'].mean():.3f}")
    
    st.markdown("---")
    
    # === 3. COMBINED METRICS VIEW ===
    st.markdown("#### 📊 Combined Metrics View")
    
    fig_combined = go.Figure()
    
    # Normalize metrics untuk comparison
    metrics_df['silhouette_norm'] = (metrics_df['silhouette'] - metrics_df['silhouette'].min()) / (metrics_df['silhouette'].max() - metrics_df['silhouette'].min())
    metrics_df['dbi_norm'] = 1 - ((metrics_df['davies_bouldin'] - metrics_df['davies_bouldin'].min()) / (metrics_df['davies_bouldin'].max() - metrics_df['davies_bouldin'].min()))
    metrics_df['cluster_norm'] = (metrics_df['active_clusters'] - metrics_df['active_clusters'].min()) / (metrics_df['active_clusters'].max() - metrics_df['active_clusters'].min())
    
    fig_combined.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['silhouette_norm'],
        mode='lines',
        name='Silhouette (norm)',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['dbi_norm'],
        mode='lines',
        name='DBI Inverse (norm)',
        line=dict(color='#ef4444', width=2)
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['cluster_norm'],
        mode='lines',
        name='Cluster Count (norm)',
        line=dict(color='#3b82f6', width=2, dash='dash')
    ))
    
    fig_combined.update_layout(
        title="Normalized Metrics Comparison (0-1 scale)",
        xaxis_title="Waktu",
        yaxis_title="Normalized Value",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.markdown("---")
    
    # === 4. MERGE HISTORY ===
    if merge_df is not None and not merge_df.empty:
        st.markdown("#### 🔀 Merge History")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Merge events over time
            fig_merge = go.Figure()
            
            # Count merges per hour
            merge_df['hour'] = merge_df['merge_timestamp'].dt.floor('H')
            merge_counts = merge_df.groupby('hour').size().reset_index(name='merge_count')
            
            fig_merge.add_trace(go.Bar(
                x=merge_counts['hour'],
                y=merge_counts['merge_count'],
                marker=dict(color='#f59e0b'),
                name='Merge Events'
            ))
            
            fig_merge.update_layout(
                title="Merge Events Over Time",
                xaxis_title="Waktu",
                yaxis_title="Jumlah Merge",
                height=350
            )
            
            st.plotly_chart(fig_merge, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Merge Statistics:**")
            st.metric("Total Merges", len(merge_df))
            st.metric("Avg Threshold", f"{merge_df['threshold_used'].mean():.3f}")
            
            # Most merged cluster
            most_merged = merge_df['old_cluster_id'].value_counts().head(3)
            st.markdown("**Top Merged Clusters:**")
            for cluster, count in most_merged.items():
                st.text(f"Cluster {cluster}: {count}x merged")
    
    st.markdown("---")
    
    # === 5. TREND ANALYSIS ===
    st.markdown("#### 📈 Trend Analysis & Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Cluster growth trend
        recent_clusters = metrics_df['active_clusters'].tail(10).mean()
        old_clusters = metrics_df['active_clusters'].head(10).mean()
        cluster_trend = ((recent_clusters - old_clusters) / old_clusters) * 100
        
        st.metric(
            "Cluster Growth Trend",
            f"{recent_clusters:.1f}",
            f"{cluster_trend:+.1f}%",
            delta_color="inverse" if cluster_trend > 20 else "normal"
        )
        
        if cluster_trend > 20:
            st.warning("⚠️ Rapid cluster growth detected!")
        elif cluster_trend < -20:
            st.info("ℹ️ Clusters consolidating")
        else:
            st.success("✅ Stable cluster count")
    
    with col2:
        # Quality trend
        recent_silhouette = metrics_df['silhouette'].tail(10).mean()
        old_silhouette = metrics_df['silhouette'].head(10).mean()
        quality_trend = ((recent_silhouette - old_silhouette) / abs(old_silhouette)) * 100
        
        st.metric(
            "Quality Trend",
            f"{recent_silhouette:.3f}",
            f"{quality_trend:+.1f}%"
        )
        
        if quality_trend < -10:
            st.error("❌ Quality declining!")
        elif quality_trend > 10:
            st.success("✅ Quality improving!")
        else:
            st.info("➡️ Quality stable")
    
    with col3:
        # Data efficiency
        total_data = metrics_df['total_data'].max()
        final_clusters = metrics_df['active_clusters'].iloc[-1]
        data_per_cluster = total_data / final_clusters if final_clusters > 0 else 0
        
        st.metric(
            "Data per Cluster",
            f"{data_per_cluster:.0f}",
            "points/cluster"
        )
        
        if data_per_cluster < 50:
            st.warning("⚠️ Clusters too small")
        elif data_per_cluster > 500:
            st.warning("⚠️ Clusters too large")
        else:
            st.success("✅ Good distribution")


# Dashboard UI utama
def update_dashboard():
    st.title(" Dashboard Monitoring Model Data Stram")

    
    # Status Pipeline
    render_pipeline_status()
    
    st.markdown("---")
    
    # Load data dari database (real-time)
    with st.spinner(" Loading cluster data from database..."):
        df = load_all_cluster_data(limit=max_data_points)
    
    if df is None or df.empty:
        st.warning(" Menunggu data cluster dari sistem...")
        return
    
    # Apply PCA
    with st.spinner(" Applying PCA dimensionality reduction..."):
        df_pca = apply_pca_to_clusters(df)
    
    if df_pca is None:
        st.error("Gagal melakukan PCA")
        return
    
    # Pastikan cluster_id bukan list
    if isinstance(df_pca.loc[0, 'cluster_id'], list):
        df_pca['cluster_id'] = df_pca['cluster_id'].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x
        )
    
    # Interpretasi Bisnis
    with st.spinner("Generating business insights..."):
        interpretations = interpret_cluster_business(df_pca)
    

    # Tab organization
    tab1, tab2, tab3, tab4 = st.tabs([
        "Cluster Visualization", 
        "Business Insights", 
        "Cluster Profiles",
        "Growth Monitoring"])
    
    
    with tab1:
        st.subheader("2D Cluster Visualization (PCA)")
        # st.info(f"Menampilkan **{len(df_pca):,} data points** hasil clustering DBSTREAM + HAC merge")
        
        fig = create_pca_cluster_plot(df_pca)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribusi ukuran cluster
        col1, col2 = st.columns(2)
        with col1:
            cluster_sizes = df_pca['cluster_id'].value_counts().sort_index()
            fig_bar = px.bar(
                x=cluster_sizes.index, 
                y=cluster_sizes.values,
                labels={'x': 'Cluster ID', 'y': 'Number of Data Points'},
                title="Cluster Size Distribution"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=cluster_sizes.values,
                names=cluster_sizes.index,
                title="Cluster Proportion",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("Business Segment Analysis & Recommendations")
        st.markdown("Interpretasi otomatis berdasarkan **dominasi perilaku pengguna** di setiap cluster")
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        sorted_interp = sorted(interpretations, key=lambda x: priority_order.get(x['priority'], 3))
        
        for interp in sorted_interp:
            with st.container():
                st.markdown(f"""
                <div style="border-left: 5px solid {interp['color']}; padding: 15px; background-color: #f8fafc; border-radius: 5px; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: {interp['color']};">{interp['segment_name']}</h3>
                    <p style="margin: 5px 0;"><b>Cluster ID:</b> {interp['cluster_id']} | <b>Size:</b> {interp['size']:,} users | <b>Priority:</b> <span style="background: {interp['color']}; color: white; padding: 2px 8px; border-radius: 3px;">{interp['priority']}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Deskripsi:**\n{interp['description']}")
                    st.markdown("**Karakteristik:**")
                    for char in interp['characteristics']:
                        st.markdown(char)
                    st.markdown(interp['recommendation'])
                
                with col2:
                    # Behavior distribution chart
                    behavior_df = pd.DataFrame(list(interp['behavior_distribution'].items()), columns=['Behavior', 'Percentage'])
                    fig_behavior = px.bar(
                        behavior_df,
                        x='Behavior',
                        y='Percentage',
                        title=f"Cluster {interp['cluster_id']} Behavior Mix",
                        color='Percentage',
                        color_continuous_scale='Blues'
                    )
                    fig_behavior.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_behavior, use_container_width=True)
                
                st.markdown("---")
    
    with tab3:
        st.subheader("Detailed Cluster Profiles")
        
        # Summary table
        summary_data = []
        for interp in interpretations:
            summary_data.append({
                'Cluster': interp['cluster_id'],
                'Segment': interp['segment_name'],
                'Size': interp['size'],
                'Buy %': f"{interp['behavior_distribution']['buy']:.1f}%",
                'Cart %': f"{interp['behavior_distribution']['cart']:.1f}%",
                'Fav %': f"{interp['behavior_distribution']['fav']:.1f}%",
                'View %': f"{interp['behavior_distribution']['pv']:.1f}%",
                'Priority': interp['priority']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Sample data dari setiap cluster
        st.markdown("### Sample Data per Cluster")
        selected_cluster = st.selectbox("Pilih Cluster untuk Detail", sorted(df_pca['cluster_id'].unique()))
        
        cluster_sample = df_pca[df_pca['cluster_id'] == selected_cluster].head(50)
        display_cols = ['User ID', 'Item ID', 'Category ID', 'Behavior type', 'timestamp', 'cluster_id']
        display_cols = [col for col in display_cols if col in cluster_sample.columns]
        
        st.dataframe(cluster_sample[display_cols], use_container_width=True)


    with tab4:
        st.subheader("📈 Cluster Growth & Evaluation Monitoring")
        st.markdown("Track pertumbuhan cluster dan evaluasi metrik dari waktu ke waktu")
        
        # Load monitoring data
        with st.spinner("🔄 Loading monitoring data..."):
            metrics_history = load_cluster_growth_data()
            merge_history = load_merge_history()
        
        if metrics_history is not None and not metrics_history.empty:
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                # Time range filter
                time_range = st.selectbox(
                    "Time Range",
                    ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
                    index=0
                )
            
            with col2:
                # Data points to show
                show_points = st.number_input(
                    "Show Last N Points",
                    min_value=10,
                    max_value=len(metrics_history),
                    value=min(100, len(metrics_history)),
                    step=10
                )
            
            # Apply filters
            if time_range != "All Time":
                now = datetime.utcnow()
                if time_range == "Last 24 Hours":
                    cutoff = now - timedelta(hours=24)
                elif time_range == "Last 7 Days":
                    cutoff = now - timedelta(days=7)
                else:  # Last 30 Days
                    cutoff = now - timedelta(days=30)
                
                metrics_filtered = metrics_history[metrics_history['timestamp'] >= cutoff]
            else:
                metrics_filtered = metrics_history.tail(show_points)
            
            # Render monitoring
            render_cluster_growth_monitoring(metrics_filtered, merge_history)
        else:
            st.warning("Data Monitoring OML Tidak ditenumkan didatabase. Jalankan sistem terlebih dahulu.")

# Main Loop
if __name__ == "__main__":
    # st_autorefresh(interval=150000, key="dashboard_refresh")
    update_dashboard()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")