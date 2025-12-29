Potongan Kode Program untuk halaman yang dihabas
def render_monitoring(metrics_df):
    st.subheader("Monitoring Pertumbuhan")

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['active_clusters'],
        mode='lines+markers',
        name='Active Clusters'
    ))
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("Evaluasi Metrik")

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['silhouette'],
        mode='lines+markers',
        name='Silhouette'
    ))
    st.plotly_chart(fig_sil, use_container_width=True)

    fig_dbi = go.Figure()
    fig_dbi.add_trace(go.Scatter(
        x=metrics_df['timestamp'],
        y=metrics_df['davies_bouldin'],
        mode='lines+markers',
        name='DBI'
    ))
    st.plotly_chart(fig_dbi, use_container_width=True)
