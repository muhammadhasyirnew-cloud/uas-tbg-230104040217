import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.linear_model import LinearRegression

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Retail Visitor Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

.dashboard-title{
    font-size:40px;
    font-weight:700;
    color:white;
}

.dashboard-subtitle{
    font-size:16px;
    color:white;
}

.header-box{
    padding:30px;
    border-radius:20px;
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    margin-bottom:20px;
}

.kpi-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    text-align:center;
}

.kpi-title{
    color:gray;
    font-size:15px;
}

.kpi-value{
    font-size:30px;
    font-weight:bold;
    color:#4f46e5;
}

.ai-box{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="header-box">
    <div class="dashboard-title">
        🏬 Smart Retail Visitor Prediction System
    </div>
    <div class="dashboard-subtitle">
        Big Data Analytics • PySpark • Parquet • Machine Learning • Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

visitor_total = pd.read_parquet(
    "output/visitor_total"
)

visitor_time = pd.read_parquet(
    "output/visitor_time"
)

ml_data = pd.read_parquet(
    "output/ml_visitor"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3081/3081559.png",
    width=120
)

st.sidebar.title("Navigation")

zone = st.sidebar.selectbox(
    "📍 Pilih Zona",
    visitor_time["zone"].unique()
)

# =====================================================
# KPI SECTION
# =====================================================

total_zone = visitor_total[
    visitor_total["zone"] == zone
]["total_visitors"].sum()

avg_visitors = round(
    ml_data["visitor_count"].mean(),
    0
)

max_visitors = int(
    ml_data["visitor_count"].max()
)

min_visitors = int(
    ml_data["visitor_count"].min()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Total Pengunjung
        </div>
        <div class="kpi-value">
            {int(total_zone):,}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Rata-rata
        </div>
        <div class="kpi-value">
            {avg_visitors}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Maksimum
        </div>
        <div class="kpi-value">
            {max_visitors}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Minimum
        </div>
        <div class="kpi-value">
            {min_visitors}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# TREND CHART
# =====================================================

st.subheader(f"📈 Trend Pengunjung Zona {zone}")

filtered = visitor_time[
    visitor_time["zone"] == zone
]

fig = px.line(
    filtered,
    x="minute_group",
    y="avg_visitors",
    markers=True,
    title=f"Trend Visitor {zone}"
)

fig.update_layout(
    template="plotly_white",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# MACHINE LEARNING
# =====================================================

st.subheader("🤖 AI Visitor Prediction")

X = ml_data[["hour"]]
y = ml_data["visitor_count"]

model = LinearRegression()
model.fit(X, y)

col1, col2 = st.columns([2,1])

with col1:

    pred_hour = st.slider(
        "Pilih Jam Prediksi",
        0,
        23,
        12
    )

with col2:

    prediction = model.predict(
        np.array([[pred_hour]])
    )

    st.markdown(f"""
    <div class="ai-box">
        <h3>Prediksi Pengunjung</h3>
        <h1 style="color:#4f46e5;">
            {int(prediction[0])}
        </h1>
        <p>Jam : {pred_hour}:00</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.write("")
st.write("")

st.markdown("""
---
### 📚 UAS Big Data Analytics

**Implementasi Big Data Pipeline menggunakan:**

✅ Apache Spark (PySpark)  
✅ Parquet Storage  
✅ Streamlit Dashboard  
✅ Plotly Visualization  
✅ Scikit-Learn Linear Regression  

**Smart Retail Visitor Prediction System**
""")