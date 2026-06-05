import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Customer Segmentation using GMM",
    layout="wide"
)

st.title(" Customer Personality Segmentation using GMM")

# Load Data
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "marketing_campaign.csv"

df = pd.read_csv(
    DATA_FILE,
    sep="\t"
)

# Missing Values
df["Income"] = df["Income"].fillna(
    df["Income"].median()
)

# Metrics
col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Income",
        int(df["Income"].mean())
    )

with col3:
    st.metric(
        "Average Recency",
        int(df["Recency"].mean())
    )

# Dataset Preview
st.subheader("Dataset Preview")

st.dataframe(df.head())

# Income Distribution
st.subheader("Income Distribution")

fig, ax = plt.subplots()

sns.histplot(
    df["Income"],
    kde=True,
    ax=ax
)

st.pyplot(fig)

# Product Spending
st.subheader("Product Spending Analysis")

spending = {
    "Wines":df["MntWines"].sum(),
    "Fruits":df["MntFruits"].sum(),
    "Meat":df["MntMeatProducts"].sum(),
    "Fish":df["MntFishProducts"].sum(),
    "Sweets":df["MntSweetProducts"].sum(),
    "Gold":df["MntGoldProds"].sum()
}

fig, ax = plt.subplots()

ax.bar(
    spending.keys(),
    spending.values()
)

st.pyplot(fig)

# Features
features = [
    "Income",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth"
]

# Heatmap
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    df[features].corr(),
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# Sidebar
st.sidebar.header("GMM Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    5
)

if st.button("Run GMM Clustering"):

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42
    )

    clusters = gmm.fit_predict(X_scaled)

    df["Cluster"] = clusters

    st.success("Clustering Completed")

    # PCA
    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(
        X_scaled
    )

    st.subheader("PCA Visualization")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        pca_result[:,0],
        pca_result[:,1],
        c=clusters,
        cmap="rainbow"
    )

    plt.colorbar(scatter)

    st.pyplot(fig)

    # Cluster Distribution
    st.subheader(
        "Cluster Distribution"
    )

    st.bar_chart(
        df["Cluster"].value_counts()
    )

    # Summary
    st.subheader("Cluster Summary")

    summary = df.groupby(
        "Cluster"
    )[features].mean()

    st.dataframe(summary)

    # Download
    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Results",
        csv,
        "customer_segments.csv",
        "text/csv"
    )
