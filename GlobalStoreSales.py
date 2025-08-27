#### Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans

#### Paths
base_path = r"your base path"
data_path = os.path.join(base_path, "datas")
model_path = os.path.join(base_path, "Models")

#### Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(data_path, "feature_engineering_preprocessed_data.csv"))
    monthly = pd.read_csv(os.path.join(data_path, "monthly_sales.csv"))
    quarterly = pd.read_csv(os.path.join(data_path, "quarterly_sales.csv"))
    yearly = pd.read_csv(os.path.join(data_path, "yearly_sales.csv"))
    top_products = pd.read_csv(os.path.join(data_path, "top_products.csv"))
    return df, monthly, quarterly, yearly, top_products

df, monthly_sales, quarterly_sales, yearly_sales, top_products = load_data()

#### Title
st.title("🛒 Global Superstore Sales Analysis of E-Commerce")
st.markdown("""
This dashboard leverages **Data Analytics** and **Machine Learning** to analyze customer behavior, predict churn, and visualize sales trends.
""")

#### Sidebar Navigation
section = st.sidebar.radio("📂 Select Section", [
    "Sales Trend Analysis",
    "Top Products",
    "Churn Impact",
    "Customer Segmentation"
])

#### Common Feature Set
features = ['Sales', 'Quantity', 'Profit', 'Purchase_Frequency', 'Engagement_Score', 'Tenure_Days']
X = df[features].fillna(df[features].mean())

#### Load or Train Scaler
scaler_path = os.path.join(model_path, 'scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, scaler_path)

X_scaled = scaler.transform(X)

####   Sales Trend Analysis
if section == "Sales Trend Analysis":
    st.header("📈 Sales Trend Analysis")
    st.subheader("Monthly Sales")
    fig, ax = plt.subplots()
    monthly_sales.plot(x='Month', y='Sales', kind='line', marker='o', ax=ax, color='teal')
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.subheader("Quarterly Sales")
    fig, ax = plt.subplots()
    quarterly_sales.plot(x='Quarter', y='Sales', kind='bar', ax=ax, color='coral')
    ax.set_title("Quarterly Sales Trend")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

    st.subheader("Yearly Sales")
    fig, ax = plt.subplots()
    yearly_sales.plot(x='Year', y='Sales', kind='bar', ax=ax, color='slateblue')
    ax.set_title("Yearly Sales Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

####  Top Products
elif section == "Top Products":
    st.header("🏆 Top-Performing Products")
    fig, ax = plt.subplots()
    sns.barplot(x=top_products['Sales'], y=top_products['Product Name'], ax=ax, palette='viridis')
    ax.set_title("Top 10 Products by Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Product Name")
    st.pyplot(fig)

#### Churn Impact
elif section == "Churn Impact":
    st.header("🔄 Churn Trends vs Revenue Impact")
    dbscan_path = os.path.join(model_path, 'dbscan_model.pkl')

    if os.path.exists(dbscan_path):
        dbscan = joblib.load(dbscan_path)
    else:
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        dbscan.fit(X_scaled)
        joblib.dump(dbscan, dbscan_path)

    df['DBSCAN_Label'] = dbscan.fit_predict(X_scaled)
    df['Anomaly'] = df['DBSCAN_Label'].apply(lambda x: 1 if x == -1 else 0)

    churn_revenue = df.groupby('Anomaly')['Sales'].sum().reset_index()
    churn_revenue['Status'] = churn_revenue['Anomaly'].map({0: 'Normal', 1: 'Churn-Prone'})

    fig, ax = plt.subplots()
    sns.barplot(data=churn_revenue, x='Status', y='Sales', palette={'Normal': 'gray', 'Churn-Prone': 'red'}, ax=ax)
    ax.set_title("Revenue Impact of Churn")
    ax.set_ylabel("Sales")
    st.pyplot(fig)

#####  Customer Segmentation
elif section == "Customer Segmentation":
    st.header("🧠 Customer Segmentation Insights")
    kmeans_path = os.path.join(model_path, 'kmeans_model.pkl')

    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
    else:
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(X_scaled)
        joblib.dump(kmeans, kmeans_path)

    df['KMeans_Cluster'] = kmeans.predict(X_scaled)

    fig, ax = plt.subplots()
    sns.countplot(data=df, x='KMeans_Cluster', palette='Set2', ax=ax)
    ax.set_title("Customer Distribution by Segment")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

    st.markdown("### Segment-wise Sales")
    cluster_sales = df.groupby('KMeans_Cluster')['Sales'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=cluster_sales, x='KMeans_Cluster', y='Sales', palette='Set2', ax=ax)
    ax.set_title("Sales by Customer Segment")
    st.pyplot(fig)
