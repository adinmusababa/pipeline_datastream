import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import uuid

# KONFIGURASI
st.set_page_config(
    page_title="Dashboard Clustering",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# STYLING
st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .segment-header {
        padding: 15px;
        border-left: 4px solid #2563eb;
        background-color: #f8fafc;
        border-radius: 4px;
        margin: 10px 0;
    }
    .info-text {
        font-size: 14px;
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# # SIDEBAR CONTROLS
# st.sidebar.title("Pengaturan Visualisasi")

# max_data_points = st.sidebar.number_input(
#     "Jumlah Data Maksimal", 
#     min_value=1000, 
#     max_value=50000, 
#     value=10000, 
#     step=1000,
#     help="Jumlah data yang akan ditampilkan dalam visualisasi"
# )

# point_size = st.sidebar.slider(
#     "Ukuran Titik", 
#     min_value=2, 
#     max_value=10, 
#     value=3,
#     help="Ukuran marker pada scatter plot"
# )

# show_labels = st.sidebar.checkbox(
#     "Tampilkan Label Cluster", 
#     value=True,
#     help="Menampilkan legend cluster pada visualisasi"
# )

# refresh_interval = st.sidebar.selectbox(
#     "Auto-Refresh Interval",
#     options=[15, 30, 60, 120],
#     index=1,
#     format_func=lambda x: f"{x} detik",
#     help="Interval refresh otomatis dashboard"
# )

# st.sidebar.markdown("---")
# st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# DATABASE CONNECTION
@st.cache_resource
def get_database():
    """Koneksi ke MongoDB"""
    client = MongoClient("mongodb://localhost:27017/")
    return client["clustering_datastreaming"]


# DATA LOADING FUNCTIONS
def load_metrics():
    """Load metrik evaluasi terbaru"""
    db = get_database()
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


def load_cluster_data(limit=10000):
    """Load data hasil clustering dari database"""
    try:
        db = get_database()
        docs = list(db.clusters.find().sort("timestamp", -1).limit(limit))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        
        # Rename field untuk konsistensi dengan analisis
        if 'user_id' in df.columns and 'User ID' not in df.columns:
            df['User ID'] = df['user_id']
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def load_metrics_history():
    """Load riwayat metrik untuk monitoring"""
    try:
        db = get_database()
        docs = list(db.metrics_archive.find().sort("timestamp", 1))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return None


def load_merge_history():
    """Load riwayat penggabungan cluster"""
    try:
        db = get_database()
        docs = list(db.merge_history.find().sort("merge_timestamp", 1))
        
        if not docs:
            return None
        
        df = pd.DataFrame(docs)
        df['merge_timestamp'] = pd.to_datetime(df['merge_timestamp'])
        return df
    
    except Exception as e:
        return None


def calculate_latency(sample_size=1000):
    """Hitung waktu pemrosesan rata-rata per data"""
    try:
        db = get_database()
        docs = list(db.clusters.find(
            {}, {"timestamp": 1}
        ).sort("timestamp", -1).limit(sample_size))
        
        if len(docs) < 2:
            return None
        
        df = pd.DataFrame(docs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # Filter outliers
        df_clean = df[(df['time_diff'] > 0) & (df['time_diff'] <= 10)]
        
        if len(df_clean) < 2:
            return None
        
        avg_latency = df_clean['time_diff'].mean()
        throughput = 1 / avg_latency if avg_latency > 0 else 0
        
        return {
            'avg_latency_ms': avg_latency * 1000,
            'throughput': throughput
        }
    
    except Exception as e:
        return None


# DATA PROCESSING
def apply_pca(df):
    """Reduksi dimensi dengan PCA untuk visualisasi 2D"""
    try:
        features_array = np.array(df['features'].tolist())
        
        if features_array.shape[1] < 2:
            st.error("Jumlah fitur tidak mencukupi untuk PCA")
            return None
        
        # Standardisasi
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]
        
        # # Info explained variance
        # explained_var = pca.explained_variance_ratio_
        # variance_info = f"Explained Variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%}"
        # st.caption(variance_info)
        
        return df
    
    except Exception as e:
        st.error(f"Error dalam PCA: {e}")
        return None

def analyze_cluster_behavior(cluster_data):
    """
    Analisis perilaku untuk setiap cluster berdasarkan user-level aggregation
    """
    results = []
    
    for cluster_id in sorted(cluster_data['cluster_id'].unique()):
        subset = cluster_data[cluster_data['cluster_id'] == cluster_id]
        
        # ===== USER-LEVEL AGGREGATION =====
        # Ambil behavior terakhir (final stage) per user
        user_behaviors = subset.groupby('User ID').agg({
            'Behavior type': 'last',  # Behavior terakhir dalam journey
            'Item ID': 'nunique',      # Jumlah item dilihat
            'Category ID': 'nunique',  # Jumlah kategori dieksplorasi
            'timestamp': 'max'         # Waktu terakhir aktivitas
        }).reset_index()
        
        size = len(user_behaviors)  # Jumlah UNIQUE USERS
        
        # Hitung distribusi behavior (user-level)
        behavior_counts = Counter(user_behaviors['Behavior type'].dropna())
        total = sum(behavior_counts.values())
        
        if total == 0:
            continue
        
        # Persentase berdasarkan JUMLAH USER
        pct = {
            'pv': (behavior_counts.get('pv', 0) / total) * 100,
            'fav': (behavior_counts.get('fav', 0) / total) * 100,
            'cart': (behavior_counts.get('cart', 0) / total) * 100,
            'buy': (behavior_counts.get('buy', 0) / total) * 100
        }
        
        # Hitung metrik tambahan per user
        avg_items_per_user = user_behaviors['Item ID'].mean()
        avg_categories_per_user = user_behaviors['Category ID'].mean()
        
        # Klasifikasi segmen
        segment = classify_segment(
            pct, 
            user_behaviors,  # Pass user_behaviors, bukan subset
            size, 
            len(cluster_data['User ID'].unique())  # Total unique users
        )
        
        segment['cluster_id'] = int(cluster_id)
        segment['size'] = size
        segment['percentages'] = pct
        segment['avg_items_per_user'] = avg_items_per_user
        segment['avg_categories_per_user'] = avg_categories_per_user
        
        results.append(segment)
    
    return results

# def analyze_cluster_behavior(cluster_data):
#     """Analisis perilaku untuk setiap cluster"""
#     results = []
    
#     for cluster_id in sorted(cluster_data['cluster_id'].unique()):
#         subset = cluster_data[cluster_data['cluster_id'] == cluster_id]
#         size = len(subset)
        
#         # Hitung distribusi behavior
#         behavior_counts = Counter(subset['Behavior type'].dropna())
#         total = sum(behavior_counts.values())
        
#         if total == 0:
#             continue
        
#         # Persentase tiap behavior
#         pct = {
#             'pv': (behavior_counts.get('pv', 0) / total) * 100,
#             'fav': (behavior_counts.get('fav', 0) / total) * 100,
#             'cart': (behavior_counts.get('cart', 0) / total) * 100,
#             'buy': (behavior_counts.get('buy', 0) / total) * 100
#         }
        
#         # Klasifikasi segmen
#         segment = classify_segment(pct, subset, size, len(cluster_data))
#         segment['cluster_id'] = int(cluster_id)
#         segment['size'] = size
#         segment['percentages'] = pct
        
#         results.append(segment)
    
#     return results

# fungsi interpretasi cluster
def classify_segment(pct, user_behaviors, size, total_users):
    """
    Klasifikasi tipe segmen berdasarkan perilaku dominan
    
    Args:
        pct: persentase behavior (user-level)
        user_behaviors: DataFrame aggregated per user
        size: jumlah user di cluster ini
        total_users: total unique users di semua cluster
    """
    buy = pct['buy']
    cart = pct['cart']
    fav = pct['fav']
    pv = pct['pv']
    
    # ✅ METRIK SUDAH BENAR (nunique di user_behaviors)
    n_users = len(user_behaviors)
    avg_items = user_behaviors['Item ID'].mean()
    avg_categories = user_behaviors['Category ID'].mean()
    
    # Logika klasifikasi (tetap sama, tapi interpretasinya beda)
    if buy > 15:
        return {
            'name': 'High-Value Buyers',
            'description': f'{buy:.1f}% dari user di cluster ini melakukan pembelian. User dengan konversi tinggi.',
            'metrics': [
                f'{n_users} unique users',
                f'Rata-rata {avg_items:.1f} item per user',
                f'Eksplorasi {avg_categories:.1f} kategori per user',
                f'Buy rate: {buy:.1f}% dari user',  # ✅ JELAS: persentase USER, bukan snapshot
                f'Proporsi: {(size/total_users)*100:.1f}% dari total users'
            ],
            'strategy': 'Loyalty program untuk repeat purchase, cross-selling berdasarkan history',
            'priority': 'HIGH',
            'color': '#dc2626'
        }
    
    elif cart > 20:
        return {
            'name': 'Cart Abandoners',
            'description': f'{cart:.1f}% dari user menambahkan item ke cart tapi tidak checkout. Potensi konversi tinggi.',
            'metrics': [
                f'{n_users} users dengan abandoned cart',
                f'Rata-rata {avg_items:.1f} item diminati per user',
                f'Cart rate: {cart:.1f}% dari user',
                f'Buy rate hanya: {buy:.1f}% dari user',
                f'Gap konversi: {cart - buy:.1f}% perlu diaktivasi'
            ],
            'strategy': 'Email reminder dengan item di cart, diskon khusus, urgency tactics',
            'priority': 'HIGH',
            'color': '#ea580c'
        }
    
    elif fav > 20:
        return {
            'name': 'Wishlist Collectors',
            'description': f'{fav:.1f}% dari user menyimpan favorit. Dalam fase pertimbangan pembelian.',
            'metrics': [
                f'{n_users} users dalam fase research',
                f'Rata-rata {avg_items:.1f} item di wishlist per user',
                f'Favorite rate: {fav:.1f}% dari user',
                f'Belum convert ke purchase'
            ],
            'strategy': 'Price drop alerts, limited-time offers, social proof reviews',
            'priority': 'MEDIUM',
            'color': '#7c3aed'
        }
    
    elif pv > 60:
        return {
            'name': 'Window Shoppers',
            'description': f'{pv:.1f}% dari user hanya browsing. Engagement rendah, perlu aktivasi.',
            'metrics': [
                f'{n_users} passive users',
                f'Rata-rata {avg_items:.1f} item dilihat per user',
                f'View-only rate: {pv:.1f}% dari user',
                f'Conversion sangat rendah: {buy:.1f}%'
            ],
            'strategy': 'Personalisasi rekomendasi, content marketing, quiz interaktif',
            'priority': 'LOW',
            'color': '#059669'
        }
    
    else:
        return {
            'name': 'Balanced Engagers',
            'description': f'Perilaku seimbang. {buy:.1f}% dari user melakukan pembelian.',
            'metrics': [
                f'{n_users} regular users',
                f'Rata-rata {avg_items:.1f} item per user',
                f'Balanced journey: Buy {buy:.1f}%, Cart {cart:.1f}%, Fav {fav:.1f}%',
                'Stable customer base dengan growth potential'
            ],
            'strategy': 'A/B testing untuk optimasi, referral program, gamification',
            'priority': 'MEDIUM',
            'color': '#2563eb'
        }
    
# def classify_segment(pct, subset, size, total_size):
#     """Klasifikasi tipe segmen berdasarkan perilaku dominan"""
#     buy = pct['buy']
#     cart = pct['cart']
#     fav = pct['fav']
#     pv = pct['pv']
    
#     # Hitung statistik
#     n_users = subset['User ID'].nunique() if 'User ID' in subset.columns else 0
#     n_items = subset['Item ID'].nunique() if 'Item ID' in subset.columns else 0
#     n_categories = subset['Category ID'].nunique() if 'Category ID' in subset.columns else 0
    
#     # Logika klasifikasi
#     if buy > 20:
#         return {
#             'name': 'High-Value Buyers',
#             'description': f'Konversi pembelian tinggi ({buy:.1f}%). Pengguna dengan transaksi aktif.',
#             'metrics': [
#                 f'{n_users} unique users',
#                 f'{n_categories} kategori produk',
#                 f'Buy rate: {buy:.1f}%',
#                 f'Proporsi: {(size/total_size)*100:.1f}% dari total'
#             ],
#             'strategy': 'Loyalty program, cross-selling, VIP benefits',
#             'priority': 'HIGH',
#             'color': '#dc2626'
#         }
    
#     elif cart > 20:
#         return {
#             'name': 'Cart Abandoners',
#             'description': f'Banyak item di keranjang ({cart:.1f}%) namun jarang checkout.',
#             'metrics': [
#                 f'{n_users} users dengan abandoned cart',
#                 f'{n_items} produk diminati',
#                 f'Cart: {cart:.1f}%, Buy: {buy:.1f}%',
#                 f'Gap konversi: {cart - buy:.1f}%'
#             ],
#             'strategy': 'Email reminder, diskon khusus',
#             'priority': 'HIGH',
#             'color': '#ea580c'
#         }
    
#     elif fav > 20:
#         return {
#             'name': 'Wishlist Collectors',
#             'description': f'Sering menyimpan favorit ({fav:.1f}%) namun belum membeli.',
#             'metrics': [
#                 f'{n_users} users dalam fase pertimbangan',
#                 f'{n_items} items di wishlist',
#                 f'Favorite rate: {fav:.1f}%',
#                 'Potensi konversi tinggi'
#             ],
#             'strategy': 'Price alerts, limited offers, social proof',
#             'priority': 'MEDIUM',
#             'color': '#7c3aed'
#         }
    
#     elif pv > 60:
#         return {
#             'name': 'Window Shoppers',
#             'description': f'Dominan browsing ({pv:.1f}%). Engagement rendah.',
#             'metrics': [
#                 f'{n_users} passive users',
#                 f'{n_items} produk dilihat',
#                 f'View rate: {pv:.1f}%',
#                 f'Conversion: {buy:.1f}%'
#             ],
#             'strategy': 'Personalisasi, content marketing, retargeting',
#             'priority': 'LOW',
#             'color': '#059669'
#         }
    
#     else:
#         return {
#             'name': 'Balanced Engagers',
#             'description': f'Perilaku seimbang (Buy: {buy:.1f}%).',
#             'metrics': [
#                 f'{n_users} regular customers',
#                 'Pola engagement seimbang',
#                 f'Buy: {buy:.1f}%, Cart: {cart:.1f}%',
#                 'Stable customer base'
#             ],
#             'strategy': 'A/B testing, referral program, email nurturing',
#             'priority': 'MEDIUM',
#             'color': '#2563eb'
#         }

# Fisuaslisasi control
def render_visualization_controls(df):
    """
    Render kontrol visualisasi di atas chart
    Returns: dict dengan settings yang dipilih user
    """
    st.markdown("### Pengaturan Visualisasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metrics = load_metrics()
        if metrics:
            total_data = metrics.get('total_data', 0)
        else:
            total_data = 0 

        default_value = min(10000, total_data) if total_data > 0 else 1000

        max_points = st.number_input(
            "Jumlah Data Maksimal",
            min_value=1000,
            max_value=total_data if total_data > 0 else 1000,
            value=default_value,
            step=1,
            help="Jumlah data yang ditampilkan"

        )

    
    with col2:
        point_size = st.slider(
            "Ukuran Titik",
            min_value=2,
            max_value=10,
            value=3,
            help="Ukuran marker pada plot"
        )
    
    with col3:
        show_legend = st.checkbox(
            "Tampilkan Legend",
            value=True,
            help="Menampilkan legend cluster"
        )
    
    # Filter Cluster ID
    st.markdown("#### Filter Cluster")
    
    all_clusters = sorted(df['cluster_id'].unique())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_clusters = st.multiselect(
            "Pilih Cluster ID yang akan ditampilkan",
            options=all_clusters,
            default=all_clusters,
            help="Kosongkan untuk menampilkan semua cluster"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        if st.button("Reset Filter", use_container_width=True):
            st.rerun()
    
    # Info
    if selected_clusters:
        n_selected = len(selected_clusters)
        n_total = len(all_clusters)
        st.caption(f"Menampilkan {n_selected} dari {n_total} cluster")
    else:
        st.warning("Pilih minimal 1 cluster untuk ditampilkan")
    
    st.markdown("---")
    
    return {
        'max_points': max_points,
        'point_size': point_size,
        'show_legend': show_legend,
        'selected_clusters': selected_clusters if selected_clusters else all_clusters
    }


# VISUALIZATION FUNCTIONS
def create_cluster_plot(df, settings):  
    """Buat scatter plot 2D hasil PCA"""
    
    # Filter data berdasarkan selected clusters
    df_filtered = df[df['cluster_id'].isin(settings['selected_clusters'])]
    
    if df_filtered.empty:
        st.warning("Tidak ada data untuk cluster yang dipilih")
        return None
    
    fig = px.scatter(
        df_filtered,
        x='pca1',
        y='pca2',
        color='cluster_id',
        hover_data=['User ID', 'Item ID', 'Behavior type', 'Category ID'] if 'User ID' in df_filtered.columns else None,
        # title=f"Visualisasi Cluster (PCA 2D) - {len(df_filtered):,} Data Points",
        labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'},
        color_continuous_scale='Viridis',
        opacity=0.6
    )
    
    fig.update_traces(marker=dict(size=settings['point_size']))
    
    fig.update_layout(
        height=600,
        showlegend=settings['show_legend'],  # 
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(showgrid=True, gridcolor='#e5e7eb')
    )
    
    return fig

def create_growth_chart(metrics_df):
    """Chart pertumbuhan jumlah cluster"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['active_clusters'],
        mode='lines+markers',
        name='Active Clusters',
        line=dict(color='#2563eb', width=2)
    ))
    
    # Trend line
    if len(metrics_df) > 5:
        z = np.polyfit(range(len(metrics_df)), metrics_df['active_clusters'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=p(range(len(metrics_df))),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=1)
        ))
    
    fig.update_layout(
        title="Pertumbuhan Jumlah Cluster",
        xaxis_title="Waktu",
        yaxis_title="Jumlah Cluster",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_metrics_chart(metrics_df, metric_name, title, color):
    """Chart untuk metrik evaluasi (Silhouette/DBI)"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df[metric_name],
        mode='lines+markers',
        line=dict(color=color, width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Waktu",
        yaxis_title=metric_name.replace('_', ' ').title(),
        height=400
    )
    
    return fig


# MAIN COMPONENTS
def render_status_section(metrics, latency):
    """Render status monitoring section"""
    st.subheader("Status Sistem")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Data Processed",
            f"{metrics['total_data']:,}"
        )
    
    with col2:
        st.metric(
            "Active Clusters",
            metrics['active_clusters']
        )
    
    with col3:
        st.metric(
            "Silhouette Score",
            f"{metrics['silhouette']:.3f}"
        )
    
    with col4:
        st.metric(
            "Davies-Bouldin Index",
            f"{metrics['davies_bouldin']:.3f}"
        )
    
    # Performance metrics
    if latency:
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                "Avg Processing Time",
                f"{latency['avg_latency_ms']:.1f} ms"
            )
        
        with col6:
            st.metric(
                "Throughput",
                f"{latency['throughput']:.1f} data/s"
            )
        
        with col7:
            def humanize_timedelta(td):
                seconds = int(td.total_seconds())
                if seconds < 60:
                    return f"{seconds}s ago"
                elif seconds < 3600:  # kurang dari 1 jam
                    minutes = seconds // 60
                    return f"{minutes}m ago"
                elif seconds < 86400:  # kurang dari 24 jam
                    hours = seconds // 3600
                    return f"{hours}h ago"
                else:
                    days = seconds // 86400
                    return f"{days}d ago"

            # penggunaan
            last_update = metrics['timestamp']
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))

            time_diff = datetime.utcnow() - last_update

            st.metric(
                "Last Update",
                humanize_timedelta(time_diff)
            )
            # last_update = metrics['timestamp']
            # if isinstance(last_update, str):
            #     last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            # time_diff = datetime.utcnow() - last_update
            # st.metric(
            #     "Last Update",
            #     f"{int(time_diff.total_seconds())}s ago"
            # )

def render_cluster_analysis(df_pca, interpretations, settings):  
    """Render analisis cluster dan visualisasi"""

    visualization_settings = render_visualization_controls(df_pca)
    
    # Visualisasi
    st.subheader("Visualisasi Cluster")
    fig = create_cluster_plot(df_pca, visualization_settings)  
    
    if fig:  # 
        st.plotly_chart(fig, use_container_width=True)
    
    # Filter data untuk distribusi berdasarkan selected clusters
    df_filtered = df_pca[df_pca['cluster_id'].isin(visualization_settings['selected_clusters'])]
    
    # Distribusi cluster (gunakan df_filtered)
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_sizes = df_filtered['cluster_id'].value_counts().sort_index()  
        fig_bar = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            labels={'x': 'Cluster ID', 'y': 'Jumlah Data'},
            title="Distribusi Ukuran Cluster"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            values=cluster_sizes.values,
            names=cluster_sizes.index,
            title="Proporsi Cluster",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    return visualization_settings  



def render_business_insights(interpretations):
    """Render interpretasi bisnis"""
    st.subheader("Analisis Segmen Bisnis")
    st.text('catatan: Interpretasi ini adalah Karakteristik Dominan dari cluster, bukan segmentasi eksklusif.')
    
    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    sorted_data = sorted(interpretations, key=lambda x: priority_order.get(x['priority'], 3))
    
    for item in sorted_data:
        with st.expander(f"Cluster {item['cluster_id']}: {item['name']} ({item['size']:,} users)"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Deskripsi:** {item['description']}")
                st.markdown("**Karakteristik:**")
                for metric in item['metrics']:
                    st.text(f"  - {metric}")
                st.markdown(f"**Strategi:** {item['strategy']}")
                
                # Priority badge
                priority_colors = {'HIGH': '#dc2626', 'MEDIUM': '#ea580c', 'LOW': '#059669'}
                st.markdown(
                    f"<span style='background:{priority_colors[item['priority']]}; color:white; padding:4px 12px; border-radius:4px; font-size:12px;'>{item['priority']} PRIORITY</span>",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Behavior distribution
                behavior_df = pd.DataFrame(
                    list(item['percentages'].items()),
                    columns=['Behavior', 'Percentage']
                )
                fig = px.bar(
                    behavior_df,
                    x='Behavior',
                    y='Percentage',
                    title="Behavior Distribution",
                    color='Percentage',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=250, showlegend=False)
                # st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(fig, use_container_width=True, key=f"business_insight_{uuid.uuid4()}")


def render_monitoring(metrics_df, merge_df):
    """Render monitoring pertumbuhan cluster"""
    st.subheader("Monitoring Pertumbuhan")
    
    # Cluster growth
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_growth_chart(metrics_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.caption("Statistik Cluster")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Min", int(metrics_df['active_clusters'].min()))
        with stat_col2:
            st.metric("Max", int(metrics_df['active_clusters'].max()))
        with stat_col3:
            st.metric("Avg", f"{metrics_df['active_clusters'].mean():.1f}")
    
    with col2:
        # Cluster density
        metrics_df['density'] = (metrics_df['active_clusters'] / metrics_df['total_data']) * 1000
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df['total_data'],
            y=metrics_df['density'],
            marker_color='#059669'
        ))
        fig.update_layout(
            title="Cluster Density (per 1000 data)",
            xaxis_title="Total Data",
            yaxis_title="Clusters / 1000 data",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Evaluation metrics
    st.markdown("---")
    st.subheader("Evaluasi Metrik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_metrics_chart(metrics_df, 'silhouette', 'Silhouette Score Over Time', '#7c3aed')
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=0.25, line_dash="dash", line_color="orange", annotation_text="Fair")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_metrics_chart(metrics_df, 'davies_bouldin', 'Davies-Bouldin Index Over Time', '#dc2626')
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=2.0, line_dash="dash", line_color="orange", annotation_text="Acceptable")
        st.plotly_chart(fig, use_container_width=True)
    
    # # Merge history
    # if merge_df is not None and not merge_df.empty:
    #     st.markdown("---")
    #     st.subheader("Riwayat Merge")
        
    #     col1, col2 = st.columns([3, 1])
        
    #     with col1:
    #         merge_df['hour'] = merge_df['merge_timestamp'].dt.floor('H')
    #         merge_counts = merge_df.groupby('hour').size().reset_index(name='count')
            
    #         fig = go.Figure()
    #         fig.add_trace(go.Bar(
    #             x=merge_counts['hour'],
    #             y=merge_counts['count'],
    #             marker_color='#ea580c'
    #         ))
    #         fig.update_layout(
    #             title="Merge Events Over Time",
    #             xaxis_title="Waktu",
    #             yaxis_title="Jumlah Merge",
    #             height=300
    #         )
    #         st.plotly_chart(fig, use_container_width=True)
        
    #     with col2:
    #         st.metric("Total Merges", len(merge_df))
    #         st.metric("Avg Threshold", f"{merge_df['threshold_used'].mean():.3f}")


# MAIN APP
def main():
    st.title("Dashboard Monitoring Sistem Clustering")
    st.caption("Real-time monitoring untuk sistem clustering berbasis DBSTREAM")
    
    # Load data
    metrics = load_metrics()
    
    if not metrics:
        st.warning("Sistem belum aktif. Menunggu data...")
        return
    
    # Status section
    latency = calculate_latency()
    render_status_section(metrics, latency)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Visualisasi Cluster",
        "Analisis Bisnis",
        "Detail Cluster",
        "Monitoring"
    ])
    
    with tab1:
            with st.spinner("Memuat data cluster..."):
                df = load_cluster_data(50000) 
            
            if df is not None and not df.empty:
                # Handle list cluster_id
                if isinstance(df.loc[0, 'cluster_id'], list):
                    df['cluster_id'] = df['cluster_id'].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
                    )
                
                df_pca = apply_pca(df)
                
                if df_pca is not None:
                    interpretations = analyze_cluster_behavior(df_pca)
                    viz_settings = render_cluster_analysis(df_pca, interpretations, None)
    
    with tab2:
        if df is not None and df_pca is not None:
            render_business_insights(interpretations)
        else:
            st.info("Data belum tersedia")
    
    with tab3:
        st.subheader("Ringkasan Cluster")
        
        if df_pca is not None and interpretations:
            # Summary table
            summary = []
            for item in interpretations:
                summary.append({
                    'Cluster ID': item['cluster_id'],
                    'Segmen': item['name'],
                    'Jumlah': item['size'],
                    'Buy %': f"{item['percentages']['buy']:.1f}%",
                    'Cart %': f"{item['percentages']['cart']:.1f}%",
                    'Priority': item['priority']
                })
            
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            
            # Detail per cluster
            st.markdown("---")
            st.subheader("Detail Data")
            selected = st.selectbox("Pilih Cluster", sorted(df_pca['cluster_id'].unique()))
            
            cluster_data = df_pca[df_pca['cluster_id'] == selected].head(100)
            cols = ['User ID', 'Item ID', 'Category ID', 'Behavior type', 'timestamp', 'cluster_id']
            cols = [c for c in cols if c in cluster_data.columns]
            
            st.dataframe(cluster_data[cols], use_container_width=True)
        else:
            st.info("Data belum tersedia")
    
    with tab4:
        metrics_history = load_metrics_history()
        merge_history = load_merge_history()
        
        if metrics_history is not None and not metrics_history.empty:
            # Filter options
            col1, col2 = st.columns([3, 1])
            with col1:
                time_filter = st.selectbox(
                    "Rentang Waktu",
                    ["Semua", "24 Jam Terakhir", "7 Hari Terakhir", "30 Hari Terakhir"]
                )
            with col2:

                show_n = st.number_input(
                    "Tampilkan N Data Terakhir",
                    min_value=10,
                    # max_value=total_data if total_data > 0 else 1000,
                    max_value=len(metrics_history),
                    value=min(100, len(metrics_history))               
                
                )
            
            # Apply filter
            if time_filter != "Semua":
                now = datetime.utcnow()
                if time_filter == "24 Jam Terakhir":
                    cutoff = now - timedelta(hours=24)
                elif time_filter == "7 Hari Terakhir":
                    cutoff = now - timedelta(days=7)
                else:
                    cutoff = now - timedelta(days=30)
                
                filtered = metrics_history[metrics_history['timestamp'] >= cutoff]
            else:
                filtered = metrics_history.tail(show_n)
            
            render_monitoring(filtered, merge_history)
        else:
            st.info("Belum ada data monitoring")


if __name__ == "__main__":
    # Auto-refresh (30 detik)
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=90000, key="refresh") 
    
    main()