import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection System")

uploaded_file = st.file_uploader("📂 Upload a CSV file with transaction data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Input Data Preview")
    st.dataframe(df.head())

    # Preprocess
    try:
        scaled_data = df.copy()
        scaled_data[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    # Prepare features for prediction
    features_for_prediction = scaled_data.drop(columns=["Class"]) if "Class" in scaled_data.columns else scaled_data

    # Predict
    predictions = model.predict(features_for_prediction)
    df["IsFraud"] = predictions

    # Summary
    total = len(df)
    frauds = df["IsFraud"].sum()
    fraud_percent = frauds / total * 100

    st.subheader("📊 Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraudulent", frauds)
    col3.metric("Fraud %", f"{fraud_percent:.2f}%")

    # Bar Chart: Fraud Count
    st.subheader("📈 Fraud vs. Legit")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="IsFraud", data=df, palette="Set2", ax=ax1)
    ax1.set_xticklabels(["Legit", "Fraud"])
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    # Histogram of fraud amounts
    st.subheader("💰 Fraud Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df[df["IsFraud"] == 1]["Amount"], bins=20, kde=True, ax=ax2, color="red")
    ax2.set_xlabel("Amount")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # Evaluation (only if true labels exist)
    if "Class" in df.columns:
        st.subheader("🧪 Evaluation (vs True Labels)")
        y_true = df["Class"]
        y_pred = df["IsFraud"]

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig3, ax3 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
        disp.plot(ax=ax3)
        st.pyplot(fig3)

        # ROC curve
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(features_for_prediction)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            st.subheader("📉 ROC Curve")
            fig4, ax4 = plt.subplots()
            ax4.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax4.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax4.set_xlabel("False Positive Rate")
            ax4.set_ylabel("True Positive Rate")
            ax4.set_title("ROC Curve")
            ax4.legend()
            st.pyplot(fig4)

    # Toggle to show all or frauds only
    st.subheader("📄 Prediction Results")
    option = st.radio("Show results for:", ["Only Fraudulent", "All Transactions"])
    if option == "Only Fraudulent":
        st.dataframe(df[df["IsFraud"] == 1])
    else:
        st.dataframe(df)

    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "fraud_results.csv", "text/csv")
