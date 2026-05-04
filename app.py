import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #212529;
    }
    .metric-sub {
        font-size: 11px;
        color: #adb5bd;
        margin-top: 2px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #212529;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 13px;
        color: #6c757d;
        margin-bottom: 16px;
    }
    .badge-best {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
    }
    .badge-warn {
        background: #fff3cd;
        color: #856404;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "close", "market_cap", "volume",
    "sentiment_mean", "sentiment_pos", "sentiment_neg",
    "close_lag1", "close_lag3", "close_lag7",
    "ma7", "ma14", "ma30",
    "std7", "std14",
    "ret_1d", "ret_7d"
]

@st.cache_data
def load_data():
    merged = pd.read_csv(os.path.join(DATA_DIR, "btc_merged.csv"), parse_dates=["date"])
    features = pd.read_csv(os.path.join(DATA_DIR, "btc_features.csv"), parse_dates=["date"])
    metrics = pd.read_csv(os.path.join(DATA_DIR, "results_metrics.csv"))
    return merged, features, metrics

@st.cache_resource
def load_models():
    models = {}
    for horizon in ["t1", "t30"]:
        with open(os.path.join(MODEL_DIR, f"lr_{horizon}.pkl"), "rb") as f:
            models[f"lr_{horizon}"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f"rf_{horizon}.pkl"), "rb") as f:
            models[f"rf_{horizon}"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f"xscaler_{horizon}.pkl"), "rb") as f:
            models[f"xscaler_{horizon}"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f"yscaler_{horizon}.pkl"), "rb") as f:
            models[f"yscaler_{horizon}"] = pickle.load(f)
    try:
        import tensorflow as tf
        for horizon in ["t1", "t30"]:
            models[f"lstm_{horizon}"] = tf.keras.models.load_model(
                os.path.join(MODEL_DIR, f"lstm_{horizon}.keras")
            )
    except Exception:
        pass
    return models

merged, features, metrics_df = load_data()
models = load_models()

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
train_df = features[features["date"] <= TRAIN_END]
val_df = features[(features["date"] > TRAIN_END) & (features["date"] <= VAL_END)]
test_df = features[features["date"] > VAL_END]

with st.sidebar:
    st.markdown("### ₿ BTC Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["Overview", "Prediksi Harga", "Analisis Sentimen", "Perbandingan Model", "Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Filter Global**")
    horizon_opt = st.selectbox("Horizon Prediksi", ["t+1 (1 hari)", "t+30 (30 hari)"])
    horizon_key = "t1" if "t+1" in horizon_opt else "t30"
    horizon_col = "target_t1" if horizon_key == "t1" else "target_t30"
    model_opt = st.selectbox("Model", ["Linear Regression", "Random Forest", "LSTM"])
    model_key = {"Linear Regression": "lr", "Random Forest": "rf", "LSTM": "lstm"}[model_opt]
    st.markdown("---")
    st.markdown(
        "<span style='font-size:11px;color:#adb5bd;'>Skripsi S1 Informatika<br>Data: CoinGecko + HuggingFace<br>Model: LR · RF · LSTM</span>",
        unsafe_allow_html=True
    )

def get_predictions(split_df, model_key, horizon_key):
    X = split_df[FEATURE_COLS].values.astype(np.float32)
    xscaler = models[f"xscaler_{horizon_key}"]
    yscaler = models[f"yscaler_{horizon_key}"]
    X_s = xscaler.transform(X)
    if model_key == "lstm":
        WINDOW = 30
        if f"lstm_{horizon_key}" not in models:
            return None
        lstm = models[f"lstm_{horizon_key}"]
        if len(X_s) < WINDOW:
            return None
        Xs = np.array([X_s[i:i+WINDOW] for i in range(len(X_s)-WINDOW+1)])
        preds_s = lstm.predict(Xs, verbose=0).ravel()
        preds = yscaler.inverse_transform(preds_s.reshape(-1,1)).ravel()
        return preds
    else:
        mdl = models[f"{model_key}_{horizon_key}"]
        preds_s = mdl.predict(X_s)
        preds = yscaler.inverse_transform(preds_s.reshape(-1,1)).ravel()
        return preds

def compute_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)

if page == "Overview":
    st.markdown('<p class="section-header">Dashboard Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Prediksi harga Bitcoin berbasis data historis CoinGecko + sentimen Twitter</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Historis", f"{len(merged):,} hari", "2021–2025")
    with col2:
        n_tweets = int(merged["sentiment_mean"].ne(0).sum()) if "sentiment_mean" in merged.columns else 0
        st.metric("Hari Ada Sentimen", f"{n_tweets:,} hari", "dari 1.826 hari")
    with col3:
        best_mape = metrics_df[metrics_df["split"]=="test"]["MAPE"].min()
        st.metric("Best MAPE (Test)", f"{best_mape:.2f}%", "Linear Regression t+1")
    with col4:
        last_price = merged["close"].iloc[-1]
        st.metric("Harga BTC Terakhir", f"${last_price:,.0f}", str(merged["date"].iloc[-1].date()))

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**Actual vs Predicted — Validation 2024**")
        val_preds = get_predictions(val_df, model_key, horizon_key)
        y_val = val_df[horizon_col].values
        dates_val = val_df["date"].values

        if val_preds is not None:
            n = min(len(val_preds), len(y_val), len(dates_val))
            if model_key == "lstm":
                dates_plot = dates_val[29:29+n]
                y_plot = y_val[29:29+n]
            else:
                dates_plot = dates_val[:n]
                y_plot = y_val[:n]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_plot, y=y_plot, name="Actual", line=dict(color="#212529", width=2)))
            fig.add_trace(go.Scatter(x=dates_plot, y=val_preds[:n], name=model_opt, line=dict(color="#f7931a", width=1.5, dash="dot")))
            fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=1.1),
                              xaxis_title="", yaxis_title="USD", plot_bgcolor="white",
                              yaxis=dict(gridcolor="#f0f0f0"), xaxis=dict(gridcolor="#f0f0f0"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model LSTM tidak tersedia atau data tidak cukup.")

    with col_right:
        st.markdown("**Ringkasan Metrik Terbaik**")
        best_rows = (metrics_df[metrics_df["split"]=="test"]
                     .sort_values(["horizon","RMSE"])
                     .groupby("horizon").head(1)
                     .reset_index(drop=True))
        st.dataframe(
            best_rows[["horizon","model","RMSE","MAE","MAPE"]].style.format({"RMSE":"{:,.0f}","MAE":"{:,.0f}","MAPE":"{:.2f}%"}),
            use_container_width=True, hide_index=True
        )
        st.markdown("**Distribusi Sentimen**")
        sent_nonzero = merged[merged["sentiment_mean"] != 0]["sentiment_mean"]
        pos = (sent_nonzero > 0.05).sum()
        neg = (sent_nonzero < -0.05).sum()
        neu = len(sent_nonzero) - pos - neg
        fig_pie = go.Figure(go.Pie(
            labels=["Positif", "Negatif", "Netral"],
            values=[pos, neg, neu],
            hole=0.5,
            marker_colors=["#28a745","#dc3545","#adb5bd"]
        ))
        fig_pie.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), showlegend=True,
                              legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Prediksi Harga":
    st.markdown('<p class="section-header">Prediksi Harga Bitcoin</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-sub">Model: {model_opt} | Horizon: {horizon_opt}</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Validation 2024", "Test 2025"])

    def plot_pred_tab(split_df, split_label):
        preds = get_predictions(split_df, model_key, horizon_key)
        y_true = split_df[horizon_col].values
        dates = split_df["date"].values

        if preds is None:
            st.info("Model tidak tersedia untuk konfigurasi ini.")
            return

        n = min(len(preds), len(y_true), len(dates))
        if model_key == "lstm":
            offset = 29
            dates_p = dates[offset:offset+n]
            y_p = y_true[offset:offset+n]
        else:
            dates_p = dates[:n]
            y_p = y_true[:n]

        rmse = float(np.sqrt(np.mean((y_p - preds[:n])**2)))
        mae = float(np.mean(np.abs(y_p - preds[:n])))
        mape = compute_mape(y_p, preds[:n])

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"${rmse:,.2f}")
        c2.metric("MAE", f"${mae:,.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_p, y=y_p, name="Actual", line=dict(color="#212529", width=2)))
        fig.add_trace(go.Scatter(x=dates_p, y=preds[:n], name=f"Prediksi ({model_opt})", line=dict(color="#f7931a", width=1.8, dash="dot")))
        fig.update_layout(
            height=380, title=f"{split_label} — BTC {horizon_opt}",
            xaxis_title="Tanggal", yaxis_title="Harga BTC (USD)",
            plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
            yaxis=dict(gridcolor="#f0f0f0"), xaxis=dict(gridcolor="#f0f0f0"),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        residuals = y_p - preds[:n]
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=dates_p, y=residuals, marker_color=["#dc3545" if r < 0 else "#28a745" for r in residuals], name="Residual"))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#212529")
        fig_res.update_layout(height=220, title="Residual (Actual - Predicted)", plot_bgcolor="white",
                              yaxis=dict(gridcolor="#f0f0f0"), xaxis=dict(gridcolor="#f0f0f0"),
                              margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_res, use_container_width=True)

    with tab1:
        plot_pred_tab(val_df, "Validation 2024")
    with tab2:
        plot_pred_tab(test_df, "Test 2025")

elif page == "Analisis Sentimen":
    st.markdown('<p class="section-header">Analisis Sentimen Pasar</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Skor sentimen harian dari tweet Bitcoin menggunakan VADER Sentiment Analyzer</p>', unsafe_allow_html=True)

    sent_df = merged[["date","sentiment_mean","sentiment_pos","sentiment_neg","close"]].copy()
    sent_df["month"] = sent_df["date"].dt.to_period("M").astype(str)
    sent_df["label"] = sent_df["sentiment_mean"].apply(
        lambda x: "Positif" if x > 0.05 else ("Negatif" if x < -0.05 else "Netral")
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Sentimen", f"{sent_df['sentiment_mean'].mean():.4f}", "Skala -1 sampai +1")
    col2.metric("Hari Sentimen Positif", f"{(sent_df['sentiment_mean']>0.05).sum()}", "dari hari yang ada tweet")
    col3.metric("Hari Sentimen Negatif", f"{(sent_df['sentiment_mean']<-0.05).sum()}", "dari hari yang ada tweet")

    st.markdown("---")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sent_df["date"], y=sent_df["sentiment_mean"],
        mode="lines", name="Sentimen Harian",
        line=dict(color="#6c5ce7", width=1),
        fill="tozeroy", fillcolor="rgba(108,92,231,0.1)"
    ))
    fig1.add_hline(y=0.05, line_dash="dash", line_color="#28a745", annotation_text="Batas Positif")
    fig1.add_hline(y=-0.05, line_dash="dash", line_color="#dc3545", annotation_text="Batas Negatif")
    fig1.update_layout(height=300, title="Tren Sentimen Harian Bitcoin (2021–2025)",
                       plot_bgcolor="white", xaxis_title="Tanggal", yaxis_title="Skor Sentimen",
                       yaxis=dict(gridcolor="#f0f0f0"), xaxis=dict(gridcolor="#f0f0f0"),
                       margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig1, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        monthly_sent = sent_df.groupby("month")["sentiment_mean"].mean().reset_index()
        fig2 = go.Figure(go.Bar(
            x=monthly_sent["month"], y=monthly_sent["sentiment_mean"],
            marker_color=["#28a745" if v > 0 else "#dc3545" for v in monthly_sent["sentiment_mean"]]
        ))
        fig2.update_layout(height=280, title="Rata-rata Sentimen per Bulan",
                           plot_bgcolor="white", xaxis_title="", yaxis_title="Skor",
                           yaxis=dict(gridcolor="#f0f0f0"), margin=dict(l=0,r=0,t=40,b=0))
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        sent_nonzero = sent_df[sent_df["sentiment_mean"] != 0]
        corr = sent_nonzero["sentiment_mean"].corr(sent_nonzero["close"])
        fig3 = go.Figure(go.Scatter(
            x=sent_nonzero["sentiment_mean"],
            y=sent_nonzero["close"],
            mode="markers",
            marker=dict(color="#f7931a", size=5, opacity=0.5)
        ))
        fig3.update_layout(
            height=280,
            title=f"Korelasi Sentimen vs Harga BTC (r = {corr:.3f})",
            plot_bgcolor="white", xaxis_title="Skor Sentimen", yaxis_title="Harga BTC (USD)",
            yaxis=dict(gridcolor="#f0f0f0"), xaxis=dict(gridcolor="#f0f0f0"),
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Overlay: Sentimen vs Harga BTC**")
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=sent_df["date"], y=sent_df["close"], name="Harga BTC",
                              line=dict(color="#f7931a", width=2)), secondary_y=False)
    fig4.add_trace(go.Scatter(x=sent_df["date"], y=sent_df["sentiment_mean"], name="Sentimen",
                              line=dict(color="#6c5ce7", width=1), opacity=0.7), secondary_y=True)
    fig4.update_layout(height=320, plot_bgcolor="white", margin=dict(l=0,r=0,t=20,b=0),
                       legend=dict(orientation="h", y=1.1),
                       yaxis=dict(title="Harga (USD)", gridcolor="#f0f0f0"),
                       yaxis2=dict(title="Skor Sentimen"))
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Perbandingan Model":
    st.markdown('<p class="section-header">Perbandingan Model Machine Learning</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Evaluasi Linear Regression, Random Forest, dan LSTM berdasarkan RMSE, MAE, dan MAPE</p>', unsafe_allow_html=True)

    horizon_filter = st.radio("Pilih Horizon", ["t+1", "t+30"], horizontal=True)
    horizon_map = {"t+1": "t+1", "t+30": "t+30"}
    sub_df = metrics_df[metrics_df["horizon"] == horizon_filter]

    st.markdown("---")
    st.markdown("**Tabel Metrik Lengkap**")
    styled = sub_df[["model","split","RMSE","MAE","MAPE"]].copy()
    st.dataframe(
        styled.style.format({"RMSE":"{:,.2f}","MAE":"{:,.2f}","MAPE":"{:.2f}%"})
              .highlight_min(subset=["RMSE","MAE","MAPE"], color="#d4edda"),
        use_container_width=True, hide_index=True
    )

    st.markdown("---")
    for metric in ["RMSE", "MAE", "MAPE"]:
        pivot = sub_df.pivot(index="model", columns="split", values=metric).reset_index()
        fig = go.Figure()
        colors = {"validation": "#6c5ce7", "test": "#f7931a"}
        for split_col in ["validation", "test"]:
            if split_col in pivot.columns:
                fig.add_trace(go.Bar(
                    name=split_col.capitalize(),
                    x=pivot["model"],
                    y=pivot[split_col],
                    marker_color=colors.get(split_col, "#adb5bd")
                ))
        fig.update_layout(
            height=280, barmode="group", title=f"{metric} per Model — Horizon {horizon_filter}",
            plot_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=1.1), margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Kesimpulan Otomatis**")
    best_test = sub_df[sub_df["split"]=="test"].sort_values("RMSE").iloc[0]
    worst_test = sub_df[sub_df["split"]=="test"].sort_values("RMSE").iloc[-1]
    st.success(f"Model terbaik untuk horizon **{horizon_filter}** adalah **{best_test['model']}** dengan RMSE = {best_test['RMSE']:,.2f} dan MAPE = {best_test['MAPE']:.2f}%.")
    st.warning(f"Model dengan performa terendah adalah **{worst_test['model']}** dengan RMSE = {worst_test['RMSE']:,.2f}.")

elif page == "Data Explorer":
    st.markdown('<p class="section-header">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Eksplorasi data historis BTC dan fitur yang digunakan dalam pemodelan</p>', unsafe_allow_html=True)

    tab_hist, tab_feat = st.tabs(["Data Historis BTC", "Feature Engineering"])

    with tab_hist:
        col1, col2 = st.columns(2)
        with col1:
            date_start = st.date_input("Dari tanggal", value=pd.to_datetime("2024-01-01"))
        with col2:
            date_end = st.date_input("Sampai tanggal", value=pd.to_datetime("2025-01-01"))

        filtered = merged[(merged["date"] >= pd.to_datetime(date_start)) & (merged["date"] <= pd.to_datetime(date_end))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered["date"], y=filtered["close"],
                                 name="Harga BTC", line=dict(color="#f7931a", width=2)))
        fig.update_layout(height=300, plot_bgcolor="white", xaxis_title="Tanggal",
                          yaxis_title="Harga (USD)", yaxis=dict(gridcolor="#f0f0f0"),
                          xaxis=dict(gridcolor="#f0f0f0"), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(filtered[["date","close","market_cap","volume","sentiment_mean"]].reset_index(drop=True)
                     .style.format({"close":"{:,.2f}","market_cap":"{:,.0f}","volume":"{:,.0f}","sentiment_mean":"{:.4f}"}),
                     use_container_width=True, hide_index=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "btc_filtered.csv", "text/csv")

    with tab_feat:
        st.markdown("**Daftar fitur yang digunakan dalam model:**")
        feat_info = {
            "close": "Harga penutupan BTC (USD)",
            "market_cap": "Market capitalization BTC",
            "volume": "Volume perdagangan harian",
            "sentiment_mean": "Rata-rata skor sentimen VADER harian",
            "sentiment_pos": "Rata-rata skor positif VADER",
            "sentiment_neg": "Rata-rata skor negatif VADER",
            "close_lag1": "Harga BTC 1 hari sebelumnya",
            "close_lag3": "Harga BTC 3 hari sebelumnya",
            "close_lag7": "Harga BTC 7 hari sebelumnya",
            "ma7": "Moving average 7 hari",
            "ma14": "Moving average 14 hari",
            "ma30": "Moving average 30 hari",
            "std7": "Rolling std deviation 7 hari",
            "std14": "Rolling std deviation 14 hari",
            "ret_1d": "Return harian (perubahan % 1 hari)",
            "ret_7d": "Return mingguan (perubahan % 7 hari)",
        }
        feat_df = pd.DataFrame({"Fitur": list(feat_info.keys()), "Keterangan": list(feat_info.values())})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("**Korelasi fitur terhadap harga besok (target t+1)**")
        corr_data = features[FEATURE_COLS + ["target_t1"]].corr()["target_t1"].drop("target_t1").sort_values()
        fig_corr = go.Figure(go.Bar(
            x=corr_data.values, y=corr_data.index,
            orientation="h",
            marker_color=["#dc3545" if v < 0 else "#28a745" for v in corr_data.values]
        ))
        fig_corr.update_layout(height=420, plot_bgcolor="white", xaxis_title="Korelasi",
                               yaxis_title="", xaxis=dict(gridcolor="#f0f0f0"),
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
