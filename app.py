import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection Intelligence System",
                   layout="wide")

st.title("💳 Fraud Detection Intelligence System")

# ==================================================
# Positional Encoding Layer
# ==================================================

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)

        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(
            10000,
            (2 * (i // 2)) / d_model
        )

        angle_rads = pos * angle_rates

        pe = np.zeros((seq_len, d_model))

        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(pe, tf.float32)

    def call(self, x):
        return x + self.pos_encoding


# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fraud_model.keras",
        custom_objects={
            "PositionalEncodingLayer":
            PositionalEncodingLayer
        }
    )

model = load_model()

# ==================================================
# Upload File
# ==================================================

uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==========================================
    # EDA
    # ==========================================

    st.subheader("📊 Dataset Statistics")

    fraud_count = df["Class"].sum()
    legit_count = len(df) - fraud_count

    fraud_pct = fraud_count / len(df) * 100
    legit_pct = legit_count / len(df) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud %", f"{fraud_pct:.4f}%")
    col2.metric("Legitimate %", f"{legit_pct:.4f}%")
    col3.metric(
        "Imbalance Ratio",
        f"1 : {int(legit_count/max(fraud_count,1))}"
    )

    # ==========================================
    # Distribution
    # ==========================================

    st.subheader("Transaction Amount Distribution")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df["Amount"], bins=50)
    ax.set_title("Amount Distribution")

    st.pyplot(fig)

    # ==========================================
    # Fraud vs Non Fraud
    # ==========================================

    st.subheader("Fraud vs Non-Fraud")

    fig, ax = plt.subplots()

    sns.countplot(
        x="Class",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    # ==========================================
    # Correlation Heatmap
    # ==========================================

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(10,8)
    )

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ==========================================
    # Sequence Creation
    # ==========================================

    st.subheader("🔄 Sequence Generation")

    seq_len = 5

    df = df.sort_values("Time")

    X_seq = []

    for i in range(len(df) - seq_len):

        seq = (
            df.iloc[i:i+seq_len]
            .drop("Class", axis=1)
            .values
        )

        X_seq.append(seq)

    X_seq = np.array(X_seq)

    st.write("Total Sequences:", X_seq.shape[0])

    # ==========================================
    # Prediction
    # ==========================================

    st.subheader("🔮 Fraud Prediction")

    with st.spinner("Running model..."):

        predictions = model.predict(
            X_seq,
            verbose=0
        )

    predictions = predictions.flatten()

    st.write("Fraud Probability Sample")

    st.dataframe(
        pd.DataFrame(
            predictions[:20],
            columns=["Fraud Probability"]
        )
    )

    # ==========================================
    # High Risk Transactions
    # ==========================================

    threshold = st.slider(
        "Fraud Threshold",
        0.0,
        1.0,
        0.5
    )

    high_risk = np.where(
        predictions > threshold
    )[0]

    st.subheader("🔥 High Risk Transactions")

    st.write(
        f"Transactions above {threshold}"
    )

    st.write(high_risk[:50])

    # ==========================================
    # Attention Visualization
    # ==========================================

    st.subheader("🧠 Attention Visualization")

    sample = X_seq[0]

    importance = np.mean(
        np.abs(sample),
        axis=1
    )

    fig, ax = plt.subplots()

    ax.bar(
        range(seq_len),
        importance
    )

    ax.set_title(
        "Transaction Importance"
    )

    ax.set_xlabel(
        "Transaction Position"
    )

    ax.set_ylabel(
        "Importance Score"
    )

    st.pyplot(fig)

    # ==========================================
    # Fraud Probability Trend
    # ==========================================

    st.subheader(
        "📈 Fraud Probability Trend"
    )

    fig, ax = plt.subplots()

    ax.plot(
        predictions[:200]
    )

    ax.set_title(
        "Fraud Probability Across Transactions"
    )

    ax.set_xlabel(
        "Transaction"
    )

    ax.set_ylabel(
        "Probability"
    )

    st.pyplot(fig)

    # ==========================================
    # Top Risk Transactions
    # ==========================================

    st.subheader(
        "🚨 Top Risk Transactions"
    )

    top_idx = np.argsort(
        predictions
    )[-10:]

    risk_df = pd.DataFrame({
        "Transaction":
        top_idx,
        "Fraud Probability":
        predictions[top_idx]
    })

    st.dataframe(
        risk_df.sort_values(
            "Fraud Probability",
            ascending=False
        )
    )

else:
    st.info(
        "Upload creditcard.csv to begin analysis."
    )
