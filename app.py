from linear_regression import train_model
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="iPhone Sales Analytics Dashboard",
    page_icon="📱",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #374151;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📱 iPhone Sales Analytics Dashboard")
st.caption("Interactive Machine Learning & Sales Insights Platform")

@st.cache_data

def load_data():

    df = pd.read_csv("iphone_sales_dataset.csv")

    return df

df = load_data()

st.success("Dataset Loaded Successfully ✅")

st.sidebar.title("📊 Dashboard Controls")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

selected_feature = st.sidebar.selectbox(
    "Select Feature for Analysis",
    numeric_columns
)

st.subheader("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head(), use_container_width=True)

st.subheader("🧹 Data Cleaning")

cleaning_strategy = st.selectbox(
    "Choose Missing Value Handling",
    ["Mean", "Median", "Drop Rows"]
)

clean_df = df.copy()

if cleaning_strategy == "Mean":
    for col in numeric_columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].mean())

elif cleaning_strategy == "Median":
    for col in numeric_columns:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

else:
    clean_df = clean_df.dropna()

st.success("Data Cleaning Completed ✅")

st.subheader("📈 Interactive Visualizations")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    fig_hist = px.histogram(
        clean_df,
        x=selected_feature,
        nbins=30,
        title=f"Distribution of {selected_feature}",
        template="plotly_dark"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with viz_col2:
    fig_box = px.box(
        clean_df,
        y=selected_feature,
        title=f"Box Plot of {selected_feature}",
        template="plotly_dark"
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("🔥 Correlation Heatmap")

corr = clean_df[numeric_columns].corr()

fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    template="plotly_dark",
    title="Feature Correlation Matrix"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.subheader("🤖 Linear Regression Model")

numeric_df = clean_df.select_dtypes(include=np.number)

if len(numeric_df.columns) >= 2:

    target_column = st.selectbox(
        "Select Target Column",
        numeric_df.columns,
        index=len(numeric_df.columns)-1
    )

    r2, mse, y_test, predictions = train_model(numeric_df, target_column)

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("R² Score", f"{r2:.3f}")

    with metric2:
        st.metric("MSE", f"{mse:.3f}")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(
        go.Scatter(
            x=y_test,
            y=predictions,
            mode='markers',
            name='Predictions'
        )
    )

    fig_scatter.update_layout(
        template="plotly_dark",
        title="Actual vs Predicted",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

csv = clean_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name='cleaned_iphone_sales.csv',
    mime='text/csv'
)
