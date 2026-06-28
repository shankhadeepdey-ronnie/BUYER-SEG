import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("buyer_profile_with_segments.csv")

df = load_data()

st.title("🏠 Real Estate Buyer Segmentation Dashboard")

st.write("""
This dashboard presents machine learning based buyer segmentation and investment profiling
for real estate market intelligence using K-Means clustering.
""")

# Sidebar filters
st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].unique()),
    default=sorted(df["country"].unique())
)

region = st.sidebar.multiselect(
    "Region",
    sorted(df["region"].unique()),
    default=sorted(df["region"].unique())
)

purpose = st.sidebar.multiselect(
    "Acquisition Purpose",
    sorted(df["acquisition_purpose"].unique()),
    default=sorted(df["acquisition_purpose"].unique())
)

client_type = st.sidebar.multiselect(
    "Client Type",
    sorted(df["client_type"].unique()),
    default=sorted(df["client_type"].unique())
)

filtered_df = df[
    (df["country"].isin(country)) &
    (df["region"].isin(region)) &
    (df["acquisition_purpose"].isin(purpose)) &
    (df["client_type"].isin(client_type))
]

# Key metrics
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Buyers", f"{filtered_df.shape[0]:,}")
col2.metric("Total Investment", f"${filtered_df['total_investment'].sum():,.0f}")
col3.metric("Avg Property Price", f"${filtered_df['avg_property_price'].mean():,.0f}")
col4.metric("Avg Properties Bought", f"{filtered_df['total_properties_bought'].mean():.2f}")

st.markdown("---")

# Segment distribution
st.subheader("Buyer Segment Distribution")

segment_counts = filtered_df["buyer_segment"].value_counts()

fig, ax = plt.subplots(figsize=(9, 5))
segment_counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Buyer Segment")
ax.set_ylabel("Number of Buyers")
ax.set_title("Buyer Segment Distribution")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

# Segment summary
st.subheader("Segment-wise Investment Summary")

segment_summary = filtered_df.groupby("buyer_segment").agg(
    buyer_count=("client_id", "count"),
    avg_age=("age", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    avg_properties_bought=("total_properties_bought", "mean"),
    avg_total_investment=("total_investment", "mean"),
    avg_property_price=("avg_property_price", "mean"),
    avg_floor_area_sqft=("avg_floor_area_sqft", "mean"),
    avg_investment_per_sqft=("investment_per_sqft", "mean")
).round(2).reset_index()

st.dataframe(segment_summary, use_container_width=True)

# Investment graph
st.subheader("Average Total Investment by Segment")

avg_investment = segment_summary.sort_values(
    "avg_total_investment",
    ascending=False
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(avg_investment["buyer_segment"], avg_investment["avg_total_investment"])
ax.set_xlabel("Buyer Segment")
ax.set_ylabel("Average Total Investment")
ax.set_title("Average Total Investment by Segment")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

# Property price graph
st.subheader("Average Property Price by Segment")

avg_price = segment_summary.sort_values(
    "avg_property_price",
    ascending=False
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(avg_price["buyer_segment"], avg_price["avg_property_price"])
ax.set_xlabel("Buyer Segment")
ax.set_ylabel("Average Property Price")
ax.set_title("Average Property Price by Segment")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

# Loan pattern
st.subheader("Loan Application Pattern by Segment")

loan_table = pd.crosstab(
    filtered_df["buyer_segment"],
    filtered_df["loan_applied"],
    normalize="index"
).mul(100).round(2)

st.dataframe(loan_table, use_container_width=True)

# Acquisition purpose
st.subheader("Acquisition Purpose by Segment")

purpose_table = pd.crosstab(
    filtered_df["buyer_segment"],
    filtered_df["acquisition_purpose"],
    normalize="index"
).mul(100).round(2)

st.dataframe(purpose_table, use_container_width=True)

# Geographic analysis
st.subheader("Top Buyer Regions")

top_regions = filtered_df["region"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(9, 5))
top_regions.plot(kind="bar", ax=ax)
ax.set_xlabel("Region")
ax.set_ylabel("Number of Buyers")
ax.set_title("Top 10 Buyer Regions")
plt.xticks(rotation=35, ha="right")
st.pyplot(fig)

# Segment insights
st.subheader("Segment Insights and Recommendations")

segment_choice = st.selectbox(
    "Select Buyer Segment",
    sorted(filtered_df["buyer_segment"].unique())
)

if segment_choice == "High-Value Bulk Investors":
    st.success("Smallest but most valuable group. They buy the highest number of properties and have the highest average total investment.")
    st.write("Recommended Strategy: Offer exclusive investor packages, early access to new listings, portfolio deals, and premium relationship management.")

elif segment_choice == "Luxury Space Buyers":
    st.success("Buyers who prefer large and premium properties with the highest average property price and floor area.")
    st.write("Recommended Strategy: Promote luxury apartments, larger floor-area units, premium locations, and lifestyle-based campaigns.")

elif segment_choice == "Budget-Conscious Home Buyers":
    st.success("Large buyer base with the lowest average total investment and lowest average property price.")
    st.write("Recommended Strategy: Promote budget-friendly properties, flexible payment options, and loan-support messaging.")

elif segment_choice == "Mid-Value Home Buyers":
    st.success("Middle-tier buyers with moderate investment levels and relatively high investment per square foot.")
    st.write("Recommended Strategy: Use personalized recommendations based on price-per-square-foot value and location preference.")

st.markdown("---")

st.subheader("Filtered Buyer Data")
st.dataframe(filtered_df, use_container_width=True)
