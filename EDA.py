
#  EDA Ecommerce Dataset
# Hamna Amin(FA23-BST-028)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io

st.set_page_config(page_title="Ecommerce EDA", layout="wide")
st.title("📊 Ecommerce Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Ecommerce Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df)   # show full dataset
    st.write("📏 Shape of dataset:", df.shape)

    # ---- Basic Info ----
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.write("🟡 Missing Values:", df.isnull().sum().to_dict())
    st.write("🟡 Duplicate Records:", df.duplicated().sum())

    # ---- Data Cleaning ----
    if "price" in df.columns:
        df["price"] = df["price"].astype(float)   # keep float

    # Derived Column: Sales
    if {"price", "quantity", "discount"}.issubset(df.columns):
        df["sales"] = df["price"] * df["quantity"] * (1 - df["discount"])
        df["sales"] = df["sales"].astype(float)

    # Convert date
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["month"] = df["order_date"].dt.month_name()
        df["day_name"] = df["order_date"].dt.day_name()

    st.subheader("✅ Cleaned Dataset")
    st.dataframe(df)

    # ---- Univariate Analysis ----
    st.header("📌 Univariate Analysis")

    # Category Distribution
    if "category" in df.columns:
        st.subheader("Category Distribution")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(cat_counts, x="category", y="count",
                     title="Category Counts", text="count", color="category")
        st.plotly_chart(fig, use_container_width=True)

    # Region-wise Sales
    if {"region", "sales"}.issubset(df.columns):
        st.subheader("Sales Share by Region")
        fig = px.pie(df, names="region", values="sales", title="Sales Contribution by Region")
        st.plotly_chart(fig, use_container_width=True)

    # Price Distribution
    if "price" in df.columns:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        df["price"].hist(ax=ax, bins=30, edgecolor="black")
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # ---- Time Series Analysis ----
    if "order_date" in df.columns:
        st.header("📌 Time Series Analysis")

        monthly_sales = df.groupby(df["order_date"].dt.to_period("M"))["sales"].sum().reset_index()
        monthly_sales["order_date"] = monthly_sales["order_date"].dt.to_timestamp()

        st.subheader("Monthly Sales Trend")
        fig = px.line(monthly_sales, x="order_date", y="sales", markers=True, title="Sales Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Bivariate Analysis ----
    st.header("📌 Bivariate Analysis")

    # Sales by Category
    if {"category", "sales"}.issubset(df.columns):
        st.subheader("Sales Distribution by Category")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="category", y="sales", palette="Set2")
        st.pyplot(fig)

    # Weekday Sales
    if "day_name" in df.columns:
        st.subheader("Sales by Day of Week")
        weekday_sales = df.groupby("day_name")["sales"].sum().reset_index()
        fig = px.bar(weekday_sales, x="day_name", y="sales", color="day_name", title="Total Sales by Weekday")
        st.plotly_chart(fig, use_container_width=True)

    # Sales by Category across Regions
    if {"category", "region", "sales"}.issubset(df.columns):
        st.subheader("Sales by Category Across Regions")
        region_sales = df.groupby(["category", "region"])["sales"].sum().reset_index()
        fig = px.bar(region_sales, y="category", x="sales", color="region", orientation="h", title="Sales by Category Across Regions")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Product Analysis ----
    st.header("📌 Product Analysis")

    if {"product_id", "sales"}.issubset(df.columns):
        st.subheader("Top 10 Best-Selling Products (Treemap)")
        prod_sales = df.groupby("product_id")["sales"].sum().reset_index()
        prod_sales = prod_sales.sort_values("sales", ascending=False).head(10)
        fig = px.treemap(prod_sales, path=["product_id"], values="sales",
                         title="Top Products by Sales", color="sales", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        low_products = df.groupby("product_id")["sales"].sum().reset_index().sort_values(by="sales", ascending=True).head(10)
        st.subheader("Lowest Performing Products")
        fig = px.bar(low_products, x="product_id", y="sales", title="Low Performing Products")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Customer Analysis ----
    st.header("📌 Customer Analysis")

    if {"customer_id", "sales"}.issubset(df.columns):
        st.subheader("Top 10 Customers by Sales (Donut Chart)")
        cust_sales = df.groupby("customer_id")["sales"].sum().reset_index()
        cust_sales = cust_sales.sort_values("sales", ascending=False).head(10)
        fig = px.pie(cust_sales, values="sales", names="customer_id",
                     hole=0.4, title="Top Customers by Sales")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Correlation ----
    st.header("📌 Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number])
    if not numeric_cols.empty:
        corr = numeric_cols.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", annot=True)
        st.pyplot(fig)

    # ---- Extra Visualization ----
    st.header("📌 Extra Visualizations")

    if {"category", "region", "sales"}.issubset(df.columns):
        st.subheader("Treemap of Sales by Category and Region")
        fig = px.treemap(df, path=["category", "region"], values="sales", title="Treemap: Category & Region Sales Share")
        st.plotly_chart(fig, use_container_width=True)

    if {"discount", "sales"}.issubset(df.columns):
        st.subheader("Discount vs Sales (Boxplot)")
        df["discount_clipped"] = df["discount"].fillna(0).clip(lower=0)
        df["discount_bin"] = pd.cut(
            df["discount_clipped"],
            bins=[-1e-6, 0, 0.01, 0.05, 0.10, 0.20, 1.0],
            labels=["0%", "0-1%", "1-5%", "5-10%", "10-20%", ">20%"]
        )
        fig = px.box(df, x="discount_bin", y="sales",
                     points="outliers", title="Sales Distribution by Discount Bin")
        fig.update_layout(xaxis_title="Discount Bin", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠ Please upload a dataset file to proceed.")
