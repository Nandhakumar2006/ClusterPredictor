import pickle
import numpy as np
import gradio as gr
from scipy.cluster.hierarchy import fcluster

# =========================
# Load all saved models
# =========================
with open("scaler_kmeans_hier.pkl", "rb") as f:
    scaler_kh = pickle.load(f)

with open("scaler_dbscan.pkl", "rb") as f:
    scaler_db = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

with open("hierarchical_model.pkl", "rb") as f:
    hier_data = pickle.load(f)
linked = hier_data["linkage_matrix"]

with open("dbscan_model.pkl", "rb") as f:
    dbscan_model = pickle.load(f)

# =========================
# Helper: Map cluster IDs to names
# =========================
def get_cluster_name(cluster_id):
    mapping = {
        0: "💎 Premium Users",
        1: "💰 Average Spenders",
        2: "🛍️ Occasional Users",
        -1: "⚫ Noise / Outlier"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

# =========================
# Define Prediction Function
# =========================
def predict_cluster(
    model_choice,
    BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
    INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
    ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
    CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
    CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE
):
    # Prepare the input data
    X = np.array([[BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
                   INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
                   ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
                   CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
                   CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT, TENURE]])

    # ======================
    # K-Means Prediction
    # ======================
    if model_choice == "K-Means":
        X_scaled = scaler_kh.transform(X)
        cluster = int(kmeans_model.predict(X_scaled)[0])
        return f"🔹 Predicted Cluster (K-Means): {get_cluster_name(cluster)}"

    # ======================
    # Hierarchical Prediction
    # ======================
    elif model_choice == "Hierarchical":
        # Hierarchical clustering can't predict new points directly
        return "🔸 Hierarchical clustering does not support new data predictions. Please choose K-Means or DBSCAN."

    # ======================
    # DBSCAN Prediction
    # ======================
    elif model_choice == "DBSCAN":
        X_scaled = scaler_db.transform(X)
        labels = dbscan_model.fit_predict(X_scaled)
        cluster = int(labels[0])
        return f"🟢 Predicted Cluster (DBSCAN): {get_cluster_name(cluster)}"

# =========================
# Custom CSS for WOW Effect
# =========================
custom_css = """
body {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.gradio-container {
    background: transparent !important;
}
h1, h2, h3 {
    text-align: center;
    color: white !important;
}
input, select, textarea {
    border-radius: 12px !important;
}
button {
    background: linear-gradient(90deg, #ff6a00, #ee0979) !important;
    border: none !important;
    color: white !important;
    font-weight: bold;
    border-radius: 12px !important;
}
button:hover {
    background: linear-gradient(90deg, #ee0979, #ff6a00) !important;
}
"""

# =========================
# Build Gradio Interface
# =========================
iface = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Dropdown(["K-Means", "Hierarchical", "DBSCAN"], label="🎯 Select Model"),
        gr.Number(label="💳 BALANCE"),
        gr.Number(label="📈 BALANCE_FREQUENCY"),
        gr.Number(label="🛒 PURCHASES"),
        gr.Number(label="💥 ONEOFF_PURCHASES"),
        gr.Number(label="📦 INSTALLMENTS_PURCHASES"),
        gr.Number(label="💵 CASH_ADVANCE"),
        gr.Number(label="🔁 PURCHASES_FREQUENCY"),
        gr.Number(label="🔥 ONEOFF_PURCHASES_FREQUENCY"),
        gr.Number(label="🧾 PURCHASES_INSTALLMENTS_FREQUENCY"),
        gr.Number(label="🏧 CASH_ADVANCE_FREQUENCY"),
        gr.Number(label="💳 CASH_ADVANCE_TRX"),
        gr.Number(label="🛍️ PURCHASES_TRX"),
        gr.Number(label="💸 CREDIT_LIMIT"),
        gr.Number(label="💰 PAYMENTS"),
        gr.Number(label="📉 MINIMUM_PAYMENTS"),
        gr.Number(label="✅ PRC_FULL_PAYMENT"),
        gr.Number(label="⏳ TENURE"),
    ],
    outputs=gr.Textbox(label="🔮 Cluster Result", lines=2),
    title="🌈 Customer Cluster Prediction App",
    description="Enter customer data and select a model to predict whether the user is a 💎 Premium, 💰 Average, or 🛍️ Occasional spender.",
    css=custom_css,
)

if __name__ == "__main__":
    iface.launch()
