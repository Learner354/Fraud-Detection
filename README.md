
```markdown
# 💳 Fraud Detection System (Random Forest + SMOTE)

A machine learning system to detect fraudulent financial transactions using classification techniques, imbalance handling, and SHAP explainability — deployed via Streamlit for interactive fraud flagging.

---

## 📌 Project Overview

This project demonstrates:

- Binary classification using **Random Forest** and **Logistic Regression**
- Handling **imbalanced data** using **SMOTE**
- Model explainability with **SHAP**
- A **Streamlit** web app for fraud detection on uploaded CSVs

---

## 📁 Project Structure

```

├── app.py                   # Streamlit web application
├── random\_forest\_model.pkl  # Trained Random Forest model
├── scaler.pkl               # Scaler used on input features
├── test\_input.csv           # Example input data (without labels)
├── requirements.txt         # List of Python dependencies
├── README.md                # Project documentation

````

---

## 📊 Dataset

- Based on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: 30 anonymized numerical columns (V1–V28, Time, Amount)
- Highly **imbalanced**: ~0.17% fraud cases

---

## ⚙️ Model Training Summary

- Models Trained:
  - `RandomForestClassifier`
  - `LogisticRegression`
- Imbalance handled with: **SMOTE**
- Evaluation: Compared performance (accuracy, precision, recall)
- Explainability:
  - Used **SHAP (TreeExplainer)** to interpret feature influence
  - Plotted feature importance using `summary_plot`

---

## 🚀 Running the App

### ✅ 1. Install requirements

```bash
pip install -r requirements.txt
````

### ✅ 2. Launch Streamlit

```bash
streamlit run app.py
```

### ✅ 3. Upload a CSV

Upload a CSV file with **30 input features** (same columns as training data, no `Class` column). The app:

* Applies preprocessing (scaling)
* Predicts fraud (`0` or `1`)
* Displays fraud predictions
* Lets you download the results

---

## 🧪 Sample Test Data

A sample input file is provided as `test_input.csv`.
You can also extract `X_test` from your notebook and save:

```python
X_test.to_csv("test_input.csv", index=False)
```

---

## 🧠 Model Explainability

* Uses `shap.TreeExplainer` for fast and accurate SHAP value generation
* Visualizations include:

  * SHAP summary plots
  * Force plots (optional)

These help understand **which features influence fraud predictions**.

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
shap
streamlit
joblib
```

---

## ✅ Future Improvements

* Compare predictions from Logistic Regression and Random Forest
* Add threshold slider for fraud probability
* Integrate SHAP force plots into the UI
* Upload Excel files or REST API endpoint

---

## 👨‍💻 Author

Made with ❤️ as a first machine learning project to demonstrate end-to-end data science workflow — from preprocessing to deployment.

---

## 📜 License

This project is open-source and free to use for educational purposes.

```
