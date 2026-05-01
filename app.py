import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    load_data,
    clean_data,
    feature_engineering,
    get_missing_summary,
    plot_missingness_heatmap,
    plot_histogram,
    plot_boxplot_by_year,
    plot_correlation_heatmap,
    plot_time_trend,
    plot_top_states_bar,
    run_regression,
    run_classification,
    run_clustering,
    run_dense_nn,
    run_lstm,
    get_numeric_columns,
    get_core_columns,
)

st.set_page_config(page_title="US Immigration Analysis App", layout="wide")

CSV_PATH = "state_data_2013-2023_20250514_3.csv"

@st.cache_data
def load_all():
    df = load_data(CSV_PATH)
    df_clean = clean_data(df)
    df_fe = feature_engineering(df_clean)
    return df, df_clean, df_fe

st.title("US Immigration Analysis Web App")
st.caption("Generated from the provided Jupyter notebook. This app covers EDA, ML, clustering, and deep learning.")
st.info(
    "Author: Tianle Chen | Advisor: Qingyang Xiao |" \
    "Copyright © 2026 Tianle Chen. All rights reserved." \
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Credits")
st.sidebar.markdown("**Author**: Tianle Chen")
st.sidebar.markdown("**Advisor**: Qingyang Xiao")                    
st.sidebar.markdown("**Copyright**: © 2026 Tianle Chen. All rights reserved.")

try:
    df, df_clean, df_fe = load_all()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Preview",
        "EDA Visualization",
        "Regression",
        "Classification",
        "Clustering",
        "Dense NN",
        "LSTM",
    ],
)

if menu == "Home":
    st.header("Project Overview")
    st.markdown(
        """
        This app analyzes US immigration data by state and year.

        Included modules:
        - Data preview and missing-value diagnostics
        - Exploratory visualization
        - Regression for `Naturalizations Total`
        - Classification for top-10 `Nonimmigrants Total`
        - PCA + KMeans clustering
        - Dense neural network regression
        - LSTM sequence modeling
        """
    )
    st.subheader("Dataset shape")
    st.write({"raw_rows": df.shape[0], "raw_cols": df.shape[1], "clean_rows": df_clean.shape[0], "feature_engineered_cols": df_fe.shape[1]})

elif menu == "Data Preview":
    st.header("Data Preview")
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Cleaned Data", "Missing Values"])
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
    with tab2:
        st.dataframe(df_clean.head(50), use_container_width=True)
    with tab3:
        st.dataframe(get_missing_summary(df), use_container_width=True)
        fig = plot_missingness_heatmap(df)
        st.pyplot(fig)

elif menu == "EDA Visualization":
    st.header("Exploratory Data Analysis")
    numeric_cols = get_numeric_columns(df_clean)
    core_cols = get_core_columns(df_clean)

    eda_type = st.selectbox(
        "Choose visualization",
        ["Histogram", "Boxplot by Year", "Correlation Heatmap", "Time Trend", "Top States Bar Chart"],
    )

    if eda_type == "Histogram":
        col = st.selectbox("Numeric feature", numeric_cols, index=0)
        fig = plot_histogram(df_clean, col)
        st.pyplot(fig)

    elif eda_type == "Boxplot by Year":
        candidates = [c for c in core_cols if c != "Year"] or numeric_cols
        col = st.selectbox("Metric", candidates, index=0)
        fig = plot_boxplot_by_year(df_clean, col)
        st.pyplot(fig)

    elif eda_type == "Correlation Heatmap":
        fig = plot_correlation_heatmap(df_clean)
        st.pyplot(fig)

    elif eda_type == "Time Trend":
        trend_candidates = [c for c in [
            "Population",
            "Nonimmigrants Total",
            "Lawful Permanent Residents Total",
            "Naturalizations Total",
            "Refugees Total",
            "Asylees Total",
        ] if c in df_clean.columns]
        col = st.selectbox("Trend metric", trend_candidates, index=0)
        fig = plot_time_trend(df_clean, col)
        st.pyplot(fig)

    elif eda_type == "Top States Bar Chart":
        candidates = [c for c in core_cols if c != "Year"] or numeric_cols
        col = st.selectbox("Ranking metric", candidates, index=0)
        fig = plot_top_states_bar(df_clean, col)
        st.pyplot(fig)

elif menu == "Regression":
    st.header("Regression: Predict Naturalizations Total")
    model_name = st.selectbox(
        "Model",
        [
            "LinearRegression",
            "Ridge(alpha=1.0)",
            "Lasso(alpha=0.001)",
            "RandomForestRegressor",
            "HistGradientBoostingRegressor",
        ],
    )
    if st.button("Run Regression"):
        with st.spinner("Training regression model..."):
            results = run_regression(df_fe, model_name=model_name)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{results['mae']:.2f}")
        c2.metric("RMSE", f"{results['rmse']:.2f}")
        c3.metric("R²", f"{results['r2']:.4f}")
        st.pyplot(results["fig_pred_actual"])
        st.pyplot(results["fig_residuals"])

elif menu == "Classification":
    st.header("Classification: Top-10 Nonimmigrants Total")
    model_name = st.selectbox("Classifier", ["LogisticRegression", "RandomForestClassifier"])
    if st.button("Run Classification"):
        with st.spinner("Training classifier..."):
            results = run_classification(df_fe, model_name=model_name)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{results['accuracy']:.4f}")
        if results["roc_auc"] is not None:
            c2.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        st.text("Classification report")
        st.code(results["classification_report"])
        st.pyplot(results["fig_confusion"])
        if results["fig_roc"] is not None:
            st.pyplot(results["fig_roc"])

elif menu == "Clustering":
    st.header("PCA + KMeans Clustering")
    k = st.slider("Number of clusters", min_value=2, max_value=10, value=6)
    if st.button("Run Clustering"):
        with st.spinner("Running PCA and KMeans..."):
            results = run_clustering(df_clean, k=k)
        st.pyplot(results["fig_scatter"])
        st.subheader("Cluster assignments")
        st.dataframe(results["cluster_df"], use_container_width=True)
        st.subheader("Cluster centroids")
        st.dataframe(results["centroids"], use_container_width=True)

elif menu == "Dense NN":
    st.header("Dense Neural Network Regression")
    if st.button("Run Dense NN"):
        with st.spinner("Training Dense NN... This may take a while."):
            results = run_dense_nn(df_fe)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{results['mae']:.2f}")
        c2.metric("RMSE", f"{results['rmse']:.2f}")
        c3.metric("R²", f"{results['r2']:.4f}")
        st.pyplot(results["fig_loss"])
        st.pyplot(results["fig_mae"])
        st.pyplot(results["fig_pred_actual"])

elif menu == "LSTM":
    st.header("LSTM Sequence Model")
    if st.button("Run LSTM"):
        with st.spinner("Training LSTM... This may take a while."):
            results = run_lstm(df_fe)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{results['mae']:.2f}")
        c2.metric("RMSE", f"{results['rmse']:.2f}")
        c3.metric("R²", f"{results['r2']:.4f}")
        st.pyplot(results["fig_loss"])
        st.pyplot(results["fig_pred_actual"])
