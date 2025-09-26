
import streamlit as st
import pandas as pd
import numpy as np
import joblib   # ✅ correct import
import os
import json
import nltk

# Ensure NLTK packages (safe small download at runtime)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

st.title("Airbnb Price Prediction — Streamlit App")
st.write(
    """
This app predicts Airbnb listing prices given features. 
It is intentionally flexible: you can either upload your trained `model.pkl` (and optional `preprocessor.pkl`) **or** upload a CSV for batch predictions.
If you don't have model files, you can still enter a single example via JSON input and the app will attempt to predict if a compatible model is loaded.
"""
)

# Sidebar: model / preprocessor upload
st.sidebar.header("Model & Files")
uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_prep = st.sidebar.file_uploader("Upload preprocessor pipeline (optional) (.pkl/.joblib)", type=["pkl","joblib"])
use_sample_files = st.sidebar.checkbox("Try to auto-load model/preprocessor from current directory", value=True)

MODEL = None
PREPROCESSOR = None
model_path_hint = "/workspace or current working directory"

def load_from_bytesio(fobj):
    # joblib.load accepts file-like objects for newer versions, but safer to write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(fobj.read())
        tmp.flush()
        tmp_path = tmp.name
    obj = joblib.load(tmp_path)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return obj

# Try uploads first
if uploaded_model is not None:
    try:
        MODEL = load_from_bytesio(uploaded_model)
        st.sidebar.success("Model loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

if uploaded_prep is not None:
    try:
        PREPROCESSOR = load_from_bytesio(uploaded_prep)
        st.sidebar.success("Preprocessor loaded from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded preprocessor: {e}")

# Try to auto-load from disk (useful when deploying container with files)
if MODEL is None and use_sample_files:
    candidates = [f for f in os.listdir('.') if f.lower().endswith(('.pkl','.joblib'))]
    pref_order = ["model.pkl","final_model.pkl","random_forest.pkl","xgboost.pkl","model.joblib","model.sav"] 
    chosen = None
    for p in pref_order:
        if p in candidates:
            chosen = p
            break
    if not chosen and candidates:
        chosen = candidates[0]
    if chosen:
        try:
            MODEL = joblib.load(chosen)
            st.sidebar.success(f"Auto-loaded model: {chosen}")
        except Exception as e:
            st.sidebar.warning(f"Found candidate {chosen} but failed to load: {e}")

if PREPROCESSOR is None and use_sample_files:
    candidates = [f for f in os.listdir('.') if f.lower().startswith(('preproc','preprocessor','pipeline','prep')) and f.lower().endswith(('.pkl','.joblib'))]
    if candidates:
        try:
            PREPROCESSOR = joblib.load(candidates[0])
            st.sidebar.success(f"Auto-loaded preprocessor: {candidates[0]}")
        except Exception as e:
            st.sidebar.warning(f"Found candidate {candidates[0]} but failed to load: {e}")

# Show model info if loaded
if MODEL is not None:
    st.subheader("Loaded model info")
    try:
        st.write(type(MODEL))
        if hasattr(MODEL, "feature_names_in_"):
            st.write("feature_names_in_:", getattr(MODEL, "feature_names_in_"))
        if hasattr(MODEL, "get_params"):
            st.write("Model params sample:", list(MODEL.get_params().keys())[:10])
    except Exception:
        pass
else:
    st.info("No model loaded yet. Upload a model on the left sidebar or place model files in the app directory and enable auto-load.")

# Prediction area: upload CSV or manual entry
st.header("Make predictions")
col1, col2 = st.columns([1,2])

with col1:
    uploaded_csv = st.file_uploader("Upload CSV for batch prediction (first row should be header with feature columns)", type=["csv"])
    sample_mode = st.checkbox("Show sample input JSON (useful for manual predictions)", value=False)
    if sample_mode:
        st.write("Example JSON for a single row (modify keys to match your model's expected features):")
        st.code(json.dumps({
            "accommodates": 2,
            "bedrooms": 1,
            "bathrooms": 1,
            "latitude": 40.7128,
            "longitude": -74.0060,
            "room_type": "Entire home/apt",
            "neighbourhood": "Downtown"
        }, indent=2))

with col2:
    manual_json = st.text_area("Or paste a single-row JSON here (overrides manual form). Example: {"accommodates":2, "bedrooms":1}", height=120)
    manual_predict_btn = st.button("Predict single sample from JSON")

def predict_df(df: pd.DataFrame):
    if MODEL is None:
        st.error("No model loaded — cannot predict. Please upload a model.")
        return None
    X = df.copy()
    try:
        if PREPROCESSOR is not None:
            Xp = PREPROCESSOR.transform(X)
        else:
            if hasattr(MODEL, "named_steps") and "preprocessor" in MODEL.named_steps:
                Xp = MODEL.named_steps["preprocessor"].transform(X)
            else:
                if hasattr(MODEL, "feature_names_in_"):
                    needed = list(MODEL.feature_names_in_)
                    missing = [c for c in needed if c not in X.columns]
                    if missing:
                        st.warning(f"Missing columns expected by model: {missing}. Predictions may fail.")
                    Xp = X[ [c for c in needed if c in X.columns] ]
                else:
                    Xp = X
        preds = MODEL.predict(Xp)
        return preds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
        st.write("Uploaded data (first 10 rows):")
        st.dataframe(df.head(10))
        if st.button("Run batch prediction on uploaded CSV"):
            preds = predict_df(df)
            if preds is not None:
                df_out = df.copy()
                df_out["predicted_price"] = preds
                st.success("Prediction completed. Preview:")
                st.dataframe(df_out.head(20))
                csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions as CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if manual_predict_btn:
    try:
        input_obj = json.loads(manual_json)
        df_single = pd.DataFrame([input_obj])
        st.write("Single input parsed:")
        st.dataframe(df_single)
        preds = predict_df(df_single)
        if preds is not None:
            st.success(f"Predicted price: {float(preds[0]):.2f}")
    except Exception as e:
        st.error(f"Failed to parse JSON or predict: {e}")

if not manual_json:
    st.markdown("### Quick manual input (simple form) — fill in the fields below for a basic single prediction")
    with st.form("quick_form"):
        accommodates = st.number_input("accommodates", min_value=1, max_value=16, value=2)
        bedrooms = st.number_input("bedrooms", min_value=0, max_value=10, value=1)
        bathrooms = st.number_input("bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        room_type = st.selectbox("room_type", options=["Entire home/apt","Private room","Shared room","Hotel room"])
        neighbourhood = st.text_input("neighbourhood", value="")
        submit_quick = st.form_submit_button("Predict (quick form)")
        if submit_quick:
            entry = {
                "accommodates": accommodates,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "room_type": room_type,
                "neighbourhood": neighbourhood
            }
            df_single = pd.DataFrame([entry])
            st.write("Input:")
            st.dataframe(df_single)
            preds = predict_df(df_single)
            if preds is not None:
                st.success(f"Predicted price: {float(preds[0]):.2f}")

st.markdown("---")
st.caption("Notes: This app expects your model to be a scikit-learn compatible estimator saved with joblib or pickle. "
           "If you used custom preprocessing, upload that pipeline as `preprocessor.pkl` or include it inside a sklearn Pipeline with the model.")
st.caption("If you want, place your model files in the same directory as this app and enable auto-load (sidebar).")
manual_json = st.text_area(
    'Or paste a single-row JSON here (overrides manual form). Example: {"accommodates":2, "bedrooms":1}',
    height=120
)
