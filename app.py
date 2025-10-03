# app.py
import os
import pickle
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# --- sklearn imports at top to avoid NameError ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# optional mlflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

st.set_page_config(page_title="Delivery Time — Predictor & Trainer", layout="wide")
st.title("Delivery Time Prediction & Model Comparison")

# constants
DATA_FILE = "amazon_delivery_cleaned_dataset.csv"
SMALL_MODEL_FILE = "small_model.pkl"
FEATS = ["Order_Time", "Pickup_Time", "Weather", "Traffic", "distance_km"]
TARGET = "Delivery_Time"

# --- Load data (upload or local) ---
uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")

df = None
if uploaded is not None:
    # uploaded is a Streamlit UploadedFile (file-like). Ensure pointer at start.
    try:
        # rewind just in case
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded uploaded file `{getattr(uploaded, 'name', 'uploaded_csv')}`")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        df = None

# If no upload or upload failed, try local file
if df is None:
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            st.sidebar.info(f"Loaded local dataset `{DATA_FILE}`")
        except Exception as e:
            st.sidebar.error(f"Failed to read local `{DATA_FILE}`: {e}")
            df = None
    else:
        st.sidebar.warning(f"No local `{DATA_FILE}` found and no upload provided. Please upload a CSV.")
        df = None

if df is None:
    st.stop()

st.sidebar.markdown(f"Dataset rows: **{len(df)}**")
st.write("## Dataset preview")
st.dataframe(df.head())

# --- Simple preprocessing helpers ---
def safe_hour(x):
    try:
        return pd.to_datetime(x).hour
    except Exception:
        try:
            return int(x)
        except Exception:
            return 0

for t in ["Order_Time", "Pickup_Time"]:
    if t in df.columns:
        df[t] = df[t].apply(safe_hour)

if "Weather" not in df.columns:
    df["Weather"] = "Unknown"
else:
    df["Weather"] = df["Weather"].fillna("Unknown")

if "Traffic" not in df.columns:
    df["Traffic"] = "Unknown"
else:
    df["Traffic"] = df["Traffic"].fillna("Unknown")

if "distance_km" not in df.columns:
    df["distance_km"] = 5.0
df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(df["distance_km"].median())

# -------------------------
# SMALL persisted model: used for single-row predictions
# -------------------------
def save_small_model(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        # don't crash the app if saving fails
        pass

def load_small_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Train and persist a tiny model if missing and enough data
m_small = None
model_cols = []
if os.path.exists(SMALL_MODEL_FILE):
    try:
        m_small, model_cols = load_small_model(SMALL_MODEL_FILE)
    except Exception:
        m_small = None
        model_cols = []

if m_small is None and TARGET in df.columns:
    try:
        sub_cols = [c for c in FEATS if c in df.columns]
        sub = df[[TARGET] + sub_cols].dropna()
        if len(sub) >= 10:
            sub = sub.sample(min(2000, len(sub)), random_state=1)
            X_small = pd.get_dummies(sub[sub_cols], drop_first=True)
            y_small = sub[TARGET].astype(float)
            # ensure there is more than one row after dummying
            if X_small.shape[0] > 1 and X_small.shape[1] > 0:
                Xtr, Xte, ytr, yte = train_test_split(X_small, y_small, test_size=0.2, random_state=1)
                m_small = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=1)
                m_small.fit(Xtr, ytr)
                model_cols = X_small.columns.tolist()
                save_small_model((m_small, model_cols), SMALL_MODEL_FILE)
    except Exception:
        m_small = None
        model_cols = []

# -------------------------
# UI: Predictor
# -------------------------
st.header("Quick Predictor (single input)")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        dist = st.number_input("Distance (km)", value=float(df["distance_km"].median()))
        order_time = st.time_input("Order time", value=datetime.now().time())
        pickup_time = st.time_input("Pickup time", value=datetime.now().time())
    with col2:
        traffic_choices = sorted(df["Traffic"].dropna().unique().astype(str))
        weather_choices = sorted(df["Weather"].dropna().unique().astype(str))
        traffic = st.selectbox("Traffic", traffic_choices if traffic_choices else ["Unknown"])
        weather = st.selectbox("Weather", weather_choices if weather_choices else ["Unknown"])

    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    row = {
        "distance_km": float(dist),
        "Order_Time": int(order_time.hour),
        "Pickup_Time": int(pickup_time.hour),
        "Traffic": traffic,
        "Weather": weather
    }
    st.write("Input:", row)

    if m_small is None or not model_cols:
        st.info("Tiny model missing — attempting to train a small model now.")
        # attempt to retrain tiny model on the fly (safe)
        try:
            sub_cols = [c for c in FEATS if c in df.columns]
            sub = df[[TARGET] + sub_cols].dropna().sample(min(2000, len(df)), random_state=1)
            X_small = pd.get_dummies(sub[sub_cols], drop_first=True)
            y_small = sub[TARGET].astype(float)
            if X_small.shape[0] > 1 and X_small.shape[1] > 0:
                m_small = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=1)
                m_small.fit(X_small, y_small)
                model_cols = X_small.columns.tolist()
                save_small_model((m_small, model_cols), SMALL_MODEL_FILE)
                st.success("Tiny model trained and saved.")
            else:
                st.error("Not enough features/rows to train tiny model.")
        except Exception as e:
            st.error("Failed to train tiny model: " + str(e))

    if m_small is not None and model_cols:
        xr = pd.get_dummies(pd.DataFrame([row]))
        # ensure all expected columns exist
        for c in model_cols:
            if c not in xr.columns:
                xr[c] = 0
        try:
            xr = xr[model_cols]
            pred = m_small.predict(xr)[0]
            st.success(f"Predicted {TARGET}: {pred:.1f}")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# -------------------------
# Model training / comparison
# -------------------------
st.header("Train, Compare & (Optional) Log Models")

if TARGET not in df.columns:
    st.warning(f"Dataset missing target column `{TARGET}` — training panel disabled.")
else:
    drop_for_training = [c for c in ["Order_ID", "Order_Date", "Order_Time", "Pickup_Time", "Delivery_Time"] if c in df.columns]
    X = df.drop(columns=drop_for_training, errors="ignore")
    y = df[TARGET].astype(float)

    # encode object columns
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    st.subheader("Training settings")
    test_size = st.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.write("Select models to train:")
    cols = st.columns(3)
    use_lr = cols[0].checkbox("LinearRegression", value=True)
    use_rf = cols[1].checkbox("RandomForest (100 trees)", value=True)
    use_gb = cols[2].checkbox("GradientBoosting (100 trees)", value=True)

    run_mlflow = st.checkbox("Log runs to MLflow (if installed)", value=False)
    if run_mlflow and not MLFLOW_AVAILABLE:
        st.warning("mlflow not available in this environment. Install mlflow to enable logging.")
        run_mlflow = False

    if run_mlflow:
        exp_name = st.text_input("MLflow experiment name", value="amazon_delivery_models")
        try:
            mlflow.set_experiment(exp_name)
        except Exception:
            pass

    if st.button("Train & Compare Models"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state))
        except Exception as e:
            st.error(f"Failed to split data: {e}")
            st.stop()

        models = {}
        if use_lr:
            models["LinearRegression"] = LinearRegression()
        if use_rf:
            models["RandomForest"] = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        if use_gb:
            models["GradientBoosting"] = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

        if not models:
            st.warning("Select at least one model.")
        else:
            results = []
            progress = st.progress(0)
            total = len(models)
            i = 0
            with st.spinner("Training models..."):
                for name, model in models.items():
                    i += 1
                    run = None
                    if run_mlflow:
                        try:
                            run = mlflow.start_run(run_name=name)
                        except Exception:
                            run = None
                    try:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        mae = mean_absolute_error(y_test, preds)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        r2 = r2_score(y_test, preds)

                        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
                        st.write(f"**{name}** → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

                        if run_mlflow:
                            try:
                                if hasattr(model, "get_params"):
                                    mlflow.log_params(model.get_params())
                                mlflow.log_metric("MAE", float(mae))
                                mlflow.log_metric("RMSE", float(rmse))
                                mlflow.log_metric("R2", float(r2))
                                mlflow.sklearn.log_model(model, artifact_path=name)
                            except Exception as e:
                                st.warning(f"MLflow logging failed for {name}: {e}")
                    except Exception as e:
                        st.error(f"Training failed for {name}: {e}")
                    finally:
                        if run_mlflow:
                            try:
                                mlflow.end_run()
                            except Exception:
                                pass
                        # update progress safely
                        try:
                            progress.progress(int(i / total * 100))
                        except Exception:
                            progress.progress(min(100, int(i * 100)))  # fallback

            if results:
                res_df = pd.DataFrame(results).set_index("Model")
                st.subheader("Results Summary")
                st.dataframe(res_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}))

                st.subheader("Metric comparison")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("MAE & RMSE (lower better)")
                    st.bar_chart(res_df[["MAE", "RMSE"]])
                with c2:
                    st.write("R² (higher better)")
                    st.bar_chart(res_df[["R2"]])

                st.success("Training & comparison finished.")
                if run_mlflow:
                    st.info(f"Runs logged to MLflow experiment `{exp_name}` (if mlflow tracking accessible).")
