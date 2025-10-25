#### 🧭 Customer Segmentation & Clustering Analyzer
#### 🌐 Deployed via Gradio — Interactive Machine Learning App

This project delivers a comprehensive clustering dashboard built with Python, Scikit-learn, and Gradio.
It allows users to explore and compare how different clustering algorithms (K-Means, Hierarchical, and DBSCAN) segment customer behavior.

#### 🚀 Project Overview

The goal is to discover natural groupings among customers using their behavioral and spending data.
This helps in personalized marketing, customer retention, and business decision-making.

The app includes:

🧮 Three clustering models: K-Means, Hierarchical, and DBSCAN

⚙️ Automatic feature scaling & preprocessing

📊 Visualization of cluster patterns

🧠 Intelligent cluster labeling (Premium / Average / Occasional users)

🖥️ Intuitive Gradio UI with modern gradient styling

#### 🧩 Clustering Workflow

1. Data Preprocessing

Cleaned missing and irrelevant data

Removed outliers using IQR filtering

Scaled features using StandardScaler (separate scalers per model)

Chose key attributes: e.g. Annual Income, Spending Score, Age, Savings

2. K-Means Clustering

Used Elbow method to determine optimal k

Captures compact, spherical clusters

Cluster naming:

🟢 Premium Users — high income, high spend

🟡 Balanced Users — moderate income/spend

🔵 Occasional Users — lower spending frequency

3. Hierarchical Clustering

Agglomerative clustering using Ward linkage

Dendrogram visualization for cluster selection

4. DBSCAN

Density-based clustering to detect irregular patterns and outliers

Automatically identifies core, border, and noise points

Ideal for finding non-linear cluster shapes

#### 💻 Tech Stack
Category	Tools
Core	Python 3.10+, NumPy, Pandas, Scikit-learn
Visualization	Matplotlib, Seaborn
App Framework	Gradio
Model Export	Pickle
Deployment	Hugging Face Spaces / Localhost
🎨 Gradio UI Highlights

🌈 Animated gradient header with title banner

📂 File upload for custom CSVs

⚙️ Dropdown to select model (K-Means / Hierarchical / DBSCAN)

📊 Dynamic output visualization with cluster labels

🧠 Auto-naming of clusters based on mean spending behavior

🏗️ Run Locally

# Clone this repo
git clone https://github.com/yourusername/cluster-analyzer-gradio.git
cd cluster-analyzer-gradio

# Install dependencies
pip install -r requirements.txt

#### 📂 Project Structure

📁 cluster-analyzer/
│
├── cluster.ipynb              # Model training notebook
├── kmeans_model.pkl           # Saved K-Means model
├── hier_model.pkl             # Saved Hierarchical model
├── dbscan_model.pkl           # Saved DBSCAN model
├── scale_data_kmeans.pkl      # Scaler for K-Means
├── scale_data_hier.pkl        # Scaler for Hierarchical
├── scale_data_dbscan.pkl      # Scaler for DBSCAN
├── app.py                     # Gradio deployment file
├── requirements.txt           # Dependency list
└── README.md                  # Documentation

#### 🧠 Insights

DBSCAN effectively detects noise/outliers in real-world data.

K-Means provides clear, interpretable clusters for business use.

Hierarchical helps visualize relationships between groups.


###### APP LINK

https://huggingface.co/spaces/nandha-01/CreditCartClusterPrediction


