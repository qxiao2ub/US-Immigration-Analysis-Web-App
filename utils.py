import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    accuracy_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["Year"] = pd.to_numeric(df_clean["Year"], errors="coerce").astype("Int64")
    for c in df_clean.columns:
        if c == "State":
            continue
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    df_clean = df_clean.dropna(subset=["State", "Year"]).copy()
    df_clean["Year"] = df_clean["Year"].astype(int)
    return df_clean


def get_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if c != "State" and pd.api.types.is_numeric_dtype(df[c])]


def get_core_columns(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    core_candidates = [c for c in numeric_cols if ("Total" in c) or ("Per Million" in c) or (c == "Population")]
    return core_candidates[:12]


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame({
        "missing_rate": df.isna().mean(),
        "missing_count": df.isna().sum(),
        "dtype": df.dtypes.astype(str),
    }).sort_values(["missing_rate", "missing_count"], ascending=False)
    return missing


def plot_missingness_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(df.isna().T, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(df.shape[1]))
    ax.set_yticklabels(df.columns)
    ax.set_title("Missingness heatmap (rows vs columns)")
    ax.set_xlabel("Row index")
    fig.tight_layout()
    return fig


def plot_histogram(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[column].dropna().values
    ax.hist(data, bins=30)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_boxplot_by_year(df: pd.DataFrame, metric: str):
    years = sorted(df["Year"].dropna().unique())
    groups = [df.loc[df["Year"] == y, metric].dropna().values for y in years]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(groups, tick_labels=years, showfliers=False)
    ax.set_title(f"Boxplot by Year: {metric}")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = get_numeric_columns(df)
    num_df = df[numeric_cols].copy()
    num_df_filled = num_df.apply(lambda s: s.fillna(s.median()), axis=0)
    corr = num_df_filled.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_time_trend(df: pd.DataFrame, metric: str):
    numeric_cols = get_numeric_columns(df)
    time_df = df.groupby("Year")[numeric_cols].sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_df.index, time_df[metric].values, marker="o")
    ax.set_title(f"National Time Trend: {metric}")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric)
    fig.tight_layout()
    return fig


def plot_top_states_bar(df: pd.DataFrame, metric: str):
    latest_year = int(df["Year"].max())
    latest = df[df["Year"] == latest_year][["State", metric]].dropna().sort_values(metric, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(latest["State"], latest[metric].values)
    ax.set_title(f"Top 15 States in {latest_year}: {metric}")
    ax.set_xlabel("State")
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def feature_engineering(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_fe = df_clean.sort_values(["State", "Year"]).copy()
    total_cols = [c for c in df_fe.columns if c.endswith("Total")]

    for c in total_cols:
        df_fe[f"log1p_{c}"] = np.log1p(df_fe[c])

    for c in total_cols:
        df_fe[f"yoy_{c}"] = df_fe.groupby("State")[c].pct_change()

    for c in total_cols:
        df_fe[f"lag1_{c}"] = df_fe.groupby("State")[c].shift(1)

    return df_fe


def _prepare_regression_data(df_fe: pd.DataFrame):
    TARGET = "Naturalizations Total"
    X = df_fe.drop(columns=[TARGET]).copy()

    leak_cols = [c for c in X.columns if "Naturalizations Per Million" in c]
    for c in [f"log1p_{TARGET}", f"yoy_{TARGET}", f"lag1_{TARGET}"]:
        if c in X.columns:
            leak_cols.append(c)

    X = X.drop(columns=leak_cols, errors="ignore")
    y = df_fe[TARGET].copy()

    data = X.copy()
    data[TARGET] = y
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET]).copy()

    train = data[data["Year"] <= 2021].copy()
    test = data[data["Year"] >= 2022].copy()

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    cat_features = ["State"]
    num_features = [c for c in X_train.columns if c not in cat_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_features),
        ],
        remainder="drop",
    )

    return TARGET, train, test, X_train, y_train, X_test, y_test, cat_features, num_features, preprocess


def run_regression(df_fe: pd.DataFrame, model_name: str = "RandomForestRegressor"):
    _, _, _, X_train, y_train, X_test, y_test, _, _, preprocess = _prepare_regression_data(df_fe)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "Lasso(alpha=0.001)": Lasso(alpha=0.001, max_iter=20000),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    }
    model = models[model_name]

    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(y_test, pred, alpha=0.7)
    mn = min(y_test.min(), pred.min())
    mx = max(y_test.max(), pred.max())
    ax1.plot([mn, mx], [mn, mx])
    ax1.set_title(f"{model_name}: Predicted vs Actual")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    fig1.tight_layout()

    resid = y_test - pred
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(pred, resid, alpha=0.7)
    ax2.axhline(0)
    ax2.set_title(f"{model_name}: Residuals vs Predicted")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual (Actual - Pred)")
    fig2.tight_layout()

    return {
        "model": pipe,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_test": y_test,
        "pred": pred,
        "fig_pred_actual": fig1,
        "fig_residuals": fig2,
    }


def run_classification(df_fe: pd.DataFrame, model_name: str = "RandomForestClassifier"):
    TARGET, _, _, _, _, _, _, _, _, preprocess = _prepare_regression_data(df_fe)

    CLS_METRIC = "Nonimmigrants Total"
    tmp = df_fe[["State", "Year", CLS_METRIC]].dropna().copy()
    tmp["is_top10"] = 0

    for y in tmp["Year"].unique():
        sub = tmp[tmp["Year"] == y].copy()
        top_states = sub.sort_values(CLS_METRIC, ascending=False).head(10)["State"].tolist()
        tmp.loc[(tmp["Year"] == y) & (tmp["State"].isin(top_states)), "is_top10"] = 1

    X_full = df_fe.drop(columns=[TARGET]).copy()
    leak_cols = [c for c in X_full.columns if "Naturalizations Per Million" in c]
    for c in [f"log1p_{TARGET}", f"yoy_{TARGET}", f"lag1_{TARGET}"]:
        if c in X_full.columns:
            leak_cols.append(c)
    X_full = X_full.drop(columns=leak_cols, errors="ignore")

    data_cls = X_full.copy()
    data_cls[TARGET] = df_fe[TARGET]
    data_cls = data_cls.replace([np.inf, -np.inf], np.nan)
    data_cls = data_cls.dropna(subset=[TARGET]).copy()
    data_cls = data_cls.merge(tmp[["State", "Year", "is_top10"]], on=["State", "Year"], how="inner")

    train_c = data_cls[data_cls["Year"] <= 2021].copy()
    test_c = data_cls[data_cls["Year"] >= 2022].copy()

    Xc_train = train_c.drop(columns=["is_top10", TARGET])
    yc_train = train_c["is_top10"]
    Xc_test = test_c.drop(columns=["is_top10", TARGET])
    yc_test = test_c["is_top10"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=5000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    }
    model = models[model_name]
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(Xc_train, yc_train)
    pred = pipe.predict(Xc_test)

    acc = accuracy_score(yc_test, pred)
    report = classification_report(yc_test, pred, digits=3)

    cm = confusion_matrix(yc_test, pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, interpolation="nearest")
    fig_cm.colorbar(im, ax=ax_cm)
    ax_cm.set_title(f"{model_name}: Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig_cm.tight_layout()

    roc_auc = None
    fig_roc = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        proba = pipe.predict_proba(Xc_test)[:, 1]
        roc_auc = roc_auc_score(yc_test, proba)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(yc_test, proba, ax=ax_roc)
        ax_roc.set_title(f"{model_name}: ROC Curve")
        fig_roc.tight_layout()

    return {
        "model": pipe,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "classification_report": report,
        "fig_confusion": fig_cm,
        "fig_roc": fig_roc,
    }


def run_clustering(df_clean: pd.DataFrame, k: int = 6):
    total_cols = [c for c in df_clean.columns if c.endswith("Total")]
    pm_features = [c for c in df_clean.columns if "Per Million" in c and pd.api.types.is_numeric_dtype(df_clean[c])]

    work = df_clean.copy()
    if len(pm_features) >= 3:
        feats = pm_features
    else:
        feats = []
        for c in total_cols:
            new_col = f"percap_{c}"
            work[new_col] = work[c] / work["Population"]
            feats.append(new_col)

    profile = work.groupby("State")[feats].mean(numeric_only=True).replace([np.inf, -np.inf], np.nan)
    profile = profile.fillna(profile.median(numeric_only=True))

    scaler = StandardScaler()
    Z = scaler.fit_transform(profile.values)

    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(Z)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(Z)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(Z2[:, 0], Z2[:, 1], alpha=0.8)
    for i, st in enumerate(profile.index):
        if i % 4 == 0:
            ax.text(Z2[i, 0], Z2[i, 1], st, fontsize=8)
    ax.set_title(f"States in PCA space (2D), k={k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    cluster_df = pd.DataFrame({"State": profile.index, "cluster": clusters}).sort_values(["cluster", "State"]).reset_index(drop=True)
    centroids = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=profile.columns)

    return {
        "fig_scatter": fig,
        "cluster_df": cluster_df,
        "centroids": centroids,
    }


def run_dense_nn(df_fe: pd.DataFrame):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as e:
        raise ImportError("TensorFlow is not installed. Please install requirements.txt first.") from e

    _, _, _, X_train, y_train, X_test, y_test, cat_features, num_features, _ = _prepare_regression_data(df_fe)

    try:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe_dense = OneHotEncoder(handle_unknown="ignore", sparse=False)

    prep_keras = ColumnTransformer(
        transformers=[
            ("cat", ohe_dense, cat_features),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_features),
        ],
        remainder="drop",
    )

    Xtr_arr = prep_keras.fit_transform(X_train.copy())
    Xte_arr = prep_keras.transform(X_test.copy())
    Xtr_a, Xval_a, ytr_a, yval_a = train_test_split(Xtr_arr, y_train.values, test_size=0.2, random_state=42)

    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(Xtr_arr.shape[1],)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.10),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    history = model.fit(
        Xtr_a, ytr_a,
        validation_data=(Xval_a, yval_a),
        epochs=120,
        batch_size=32,
        callbacks=[es],
        verbose=0,
    )

    pred_te = model.predict(Xte_arr, verbose=0).ravel()
    mae = mean_absolute_error(y_test, pred_te)
    rmse = np.sqrt(mean_squared_error(y_test, pred_te))
    r2 = r2_score(y_test, pred_te)

    fig_loss, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history.history["loss"], label="train_loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_title("Dense NN: MSE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    fig_loss.tight_layout()

    fig_mae, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(history.history["mae"], label="train_mae")
    ax2.plot(history.history["val_mae"], label="val_mae")
    ax2.set_title("Dense NN: MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    fig_mae.tight_layout()

    fig_pred, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(y_test, pred_te, alpha=0.7)
    mn = min(y_test.min(), pred_te.min())
    mx = max(y_test.max(), pred_te.max())
    ax3.plot([mn, mx], [mn, mx])
    ax3.set_title("Dense NN: Predicted vs Actual")
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    fig_pred.tight_layout()

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fig_loss": fig_loss,
        "fig_mae": fig_mae,
        "fig_pred_actual": fig_pred,
    }


def run_lstm(df_fe: pd.DataFrame):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as e:
        raise ImportError("TensorFlow is not installed. Please install requirements.txt first.") from e

    TARGET = "Naturalizations Total"
    LOOKBACK = 3

    numeric_cols = get_numeric_columns(df_fe)
    rank_cols = [c for c in df_fe.columns if c.endswith("Rank")]

    seq_features = []
    for cand in [
        "Population",
        "Lawful Permanent Residents Total",
        "Nonimmigrants Total",
        "Refugees Total",
        "Asylees Total",
    ]:
        if cand in df_fe.columns:
            seq_features.append(cand)

    if len(seq_features) < 2:
        seq_features = [c for c in numeric_cols if c not in rank_cols][:5]

    df_seq = df_fe[["State", "Year"] + seq_features + [TARGET]].sort_values(["State", "Year"]).copy()
    df_seq = df_seq.replace([np.inf, -np.inf], np.nan)

    imp = SimpleImputer(strategy="median")
    df_seq[seq_features] = imp.fit_transform(df_seq[seq_features])

    sc = StandardScaler()
    df_seq[seq_features] = sc.fit_transform(df_seq[seq_features])

    X_list, y_list, meta = [], [], []
    for st, g in df_seq.groupby("State"):
        g = g.sort_values("Year").reset_index(drop=True)
        for i in range(len(g) - LOOKBACK):
            X_win = g.loc[i:i + LOOKBACK - 1, seq_features].values
            y_next = g.loc[i + LOOKBACK, TARGET]
            year_next = g.loc[i + LOOKBACK, "Year"]
            if pd.isna(y_next):
                continue
            X_list.append(X_win)
            y_list.append(y_next)
            meta.append((st, year_next))

    X_seq = np.stack(X_list)
    y_seq = np.array(y_list)
    meta_df = pd.DataFrame(meta, columns=["State", "Year_next"])

    train_idx = meta_df["Year_next"] <= 2021
    test_idx = meta_df["Year_next"] >= 2022

    X_seq_train, y_seq_train = X_seq[train_idx.values], y_seq[train_idx.values]
    X_seq_test, y_seq_test = X_seq[test_idx.values], y_seq[test_idx.values]

    Xtr_s, Xval_s, ytr_s, yval_s = train_test_split(X_seq_train, y_seq_train, test_size=0.2, random_state=42)

    tf.random.set_seed(42)
    lstm = keras.Sequential([
        keras.layers.Input(shape=(LOOKBACK, len(seq_features))),
        keras.layers.LSTM(64),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    lstm.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    es2 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    hist2 = lstm.fit(
        Xtr_s, ytr_s,
        validation_data=(Xval_s, yval_s),
        epochs=100,
        batch_size=32,
        callbacks=[es2],
        verbose=0,
    )

    pred_s = lstm.predict(X_seq_test, verbose=0).ravel()
    mae_s = mean_absolute_error(y_seq_test, pred_s)
    rmse_s = np.sqrt(mean_squared_error(y_seq_test, pred_s))
    r2_s = r2_score(y_seq_test, pred_s)

    fig_loss, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(hist2.history["loss"], label="train_loss")
    ax1.plot(hist2.history["val_loss"], label="val_loss")
    ax1.set_title("LSTM: MSE Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    fig_loss.tight_layout()

    fig_pred, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_seq_test, pred_s, alpha=0.7)
    mn = min(y_seq_test.min(), pred_s.min())
    mx = max(y_seq_test.max(), pred_s.max())
    ax2.plot([mn, mx], [mn, mx])
    ax2.set_title("LSTM: Predicted vs Actual")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    fig_pred.tight_layout()

    return {
        "model": lstm,
        "mae": mae_s,
        "rmse": rmse_s,
        "r2": r2_s,
        "fig_loss": fig_loss,
        "fig_pred_actual": fig_pred,
    }
