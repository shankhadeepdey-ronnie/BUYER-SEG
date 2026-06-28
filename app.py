import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Real Estate Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Real Estate Buyer Segmentation Dashboard")
st.write(
    "Machine Learning based buyer segmentation and investment profiling "
    "for real estate market intelligence."
)

# ==========================================================
# Helper Functions
# ==========================================================

def find_file(possible_paths):
    """
    Finds the first existing file from a list of possible paths.
    This makes the app work whether files are in root or inside csv_files/.
    """
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None


@st.cache_data
def load_csv(possible_paths):
    file_path = find_file(possible_paths)
    if file_path is None:
        return None, None

    df = pd.read_csv(file_path)
    return df, file_path


def safe_currency(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "N/A"


def safe_number(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "N/A"


def show_bar_chart(data, x_col, y_col, title):
    st.write(f"### {title}")
    chart_data = data[[x_col, y_col]].copy()
    chart_data = chart_data.set_index(x_col)
    st.bar_chart(chart_data)


# ==========================================================
# Load Data
# ==========================================================

buyer_df, buyer_path = load_csv([
    "buyer_profile_with_segments.csv",
    "csv_files/buyer_profile_with_segments.csv",
    "Project_Work_01_Final_Outputs/csv_files/buyer_profile_with_segments.csv"
])

insights_df, insights_path = load_csv([
    "final_segment_insights.csv",
    "csv_files/final_segment_insights.csv",
    "Project_Work_01_Final_Outputs/csv_files/final_segment_insights.csv"
])

segment_summary_df, segment_summary_path = load_csv([
    "segment_summary.csv",
    "csv_files/segment_summary.csv",
    "Project_Work_01_Final_Outputs/csv_files/segment_summary.csv"
])

visual_summary_df, visual_summary_path = load_csv([
    "visual_summary.csv",
    "csv_files/visual_summary.csv",
    "Project_Work_01_Final_Outputs/csv_files/visual_summary.csv"
])

kmeans_eval_df, kmeans_eval_path = load_csv([
    "kmeans_evaluation_results.csv",
    "csv_files/kmeans_evaluation_results.csv",
    "Project_Work_01_Final_Outputs/csv_files/kmeans_evaluation_results.csv"
])

if buyer_df is None:
    st.error(
        "No buyer data file found. Please upload buyer_profile_with_segments.csv "
        "or keep it inside csv_files/."
    )
    st.stop()

# ==========================================================
# Detect Whether Data is Full Buyer Profile or Summary File
# ==========================================================

required_full_columns = {
    "client_id",
    "country",
    "region",
    "client_type",
    "acquisition_purpose",
    "loan_applied",
    "buyer_segment",
    "total_investment",
    "avg_property_price",
    "total_properties_bought"
}

is_full_buyer_profile = required_full_columns.issubset(set(buyer_df.columns))

summary_columns = {
    "buyer_segment",
    "buyer_count",
    "avg_total_investment",
    "avg_property_price",
    "avg_properties_bought"
}

is_summary_file = summary_columns.issubset(set(buyer_df.columns))

st.sidebar.header("📁 Loaded Files")
st.sidebar.write(f"Main file: `{buyer_path}`")

if insights_path:
    st.sidebar.write(f"Insights file: `{insights_path}`")

if segment_summary_path:
    st.sidebar.write(f"Segment summary: `{segment_summary_path}`")

# ==========================================================
# Mode 1: Full Buyer Profile Dashboard
# ==========================================================

if is_full_buyer_profile:

    st.success("Full buyer profile loaded successfully.")

    df = buyer_df.copy()

    # ------------------------------------------------------
    # Sidebar Filters
    # ------------------------------------------------------

    st.sidebar.header("🔎 Filters")

    selected_countries = st.sidebar.multiselect(
        "Country",
        sorted(df["country"].dropna().unique()),
        default=sorted(df["country"].dropna().unique())
    )

    selected_regions = st.sidebar.multiselect(
        "Region",
        sorted(df["region"].dropna().unique()),
        default=sorted(df["region"].dropna().unique())
    )

    selected_purposes = st.sidebar.multiselect(
        "Acquisition Purpose",
        sorted(df["acquisition_purpose"].dropna().unique()),
        default=sorted(df["acquisition_purpose"].dropna().unique())
    )

    selected_client_types = st.sidebar.multiselect(
        "Client Type",
        sorted(df["client_type"].dropna().unique()),
        default=sorted(df["client_type"].dropna().unique())
    )

    filtered_df = df[
        (df["country"].isin(selected_countries)) &
        (df["region"].isin(selected_regions)) &
        (df["acquisition_purpose"].isin(selected_purposes)) &
        (df["client_type"].isin(selected_client_types))
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    # ------------------------------------------------------
    # Key Metrics
    # ------------------------------------------------------

    st.subheader("📌 Key Business Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Buyers", f"{filtered_df.shape[0]:,}")
    col2.metric("Total Investment", safe_currency(filtered_df["total_investment"].sum()))
    col3.metric("Avg Property Price", safe_currency(filtered_df["avg_property_price"].mean()))
    col4.metric("Avg Properties Bought", safe_number(filtered_df["total_properties_bought"].mean()))

    st.markdown("---")

    # ------------------------------------------------------
    # Buyer Segment Distribution
    # ------------------------------------------------------

    st.subheader("📊 Buyer Segmentation Overview")

    segment_counts = (
        filtered_df["buyer_segment"]
        .value_counts()
        .reset_index()
    )
    segment_counts.columns = ["buyer_segment", "buyer_count"]

    st.dataframe(segment_counts, use_container_width=True)
    show_bar_chart(segment_counts, "buyer_segment", "buyer_count", "Buyer Segment Distribution")

    st.markdown("---")

    # ------------------------------------------------------
    # Segment Investment Summary
    # ------------------------------------------------------

    st.subheader("💰 Investor Behavior Dashboard")

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

    show_bar_chart(
        segment_summary.sort_values("avg_total_investment", ascending=False),
        "buyer_segment",
        "avg_total_investment",
        "Average Total Investment by Segment"
    )

    show_bar_chart(
        segment_summary.sort_values("avg_property_price", ascending=False),
        "buyer_segment",
        "avg_property_price",
        "Average Property Price by Segment"
    )

    show_bar_chart(
        segment_summary.sort_values("avg_properties_bought", ascending=False),
        "buyer_segment",
        "avg_properties_bought",
        "Average Properties Bought by Segment"
    )

    st.markdown("---")

    # ------------------------------------------------------
    # Loan and Acquisition Pattern
    # ------------------------------------------------------

    st.subheader("🏦 Loan and Acquisition Behavior")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Loan Pattern by Segment")
        loan_table = pd.crosstab(
            filtered_df["buyer_segment"],
            filtered_df["loan_applied"],
            normalize="index"
        ).mul(100).round(2)

        st.dataframe(loan_table, use_container_width=True)

    with col2:
        st.write("### Acquisition Purpose by Segment")
        purpose_table = pd.crosstab(
            filtered_df["buyer_segment"],
            filtered_df["acquisition_purpose"],
            normalize="index"
        ).mul(100).round(2)

        st.dataframe(purpose_table, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Geographic Buyer Analysis
    # ------------------------------------------------------

    st.subheader("🌍 Geographic Buyer Analysis")

    top_countries = (
        filtered_df["country"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_countries.columns = ["country", "buyer_count"]

    top_regions = (
        filtered_df["region"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_regions.columns = ["region", "buyer_count"]

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(top_countries, use_container_width=True)
        show_bar_chart(top_countries, "country", "buyer_count", "Top 10 Buyer Countries")

    with col2:
        st.dataframe(top_regions, use_container_width=True)
        show_bar_chart(top_regions, "region", "buyer_count", "Top 10 Buyer Regions")

    st.markdown("---")

    # ------------------------------------------------------
    # Segment Insights and Recommendations
    # ------------------------------------------------------

    st.subheader("🧠 Segment Insights and Recommendations")

    selected_segment = st.selectbox(
        "Select Buyer Segment",
        sorted(filtered_df["buyer_segment"].dropna().unique())
    )

    if insights_df is not None and "buyer_segment" in insights_df.columns:
        selected_info = insights_df[insights_df["buyer_segment"] == selected_segment]

        if not selected_info.empty:
            row = selected_info.iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Buyer Count", int(row.get("buyer_count", 0)))
            col2.metric("Avg Total Investment", safe_currency(row.get("avg_total_investment", 0)))
            col3.metric("Avg Property Price", safe_currency(row.get("avg_property_price", 0)))

            st.write("### Segment Profile")
            st.info(row.get("segment_profile", "No profile available."))

            st.write("### Business Opportunity")
            st.success(row.get("business_opportunity", "No business opportunity available."))

            st.write("### Recommended Strategy")
            st.warning(row.get("recommended_strategy", "No recommendation available."))

            st.write("### Marketing Channel Focus")
            st.write(row.get("marketing_channel_focus", "No channel focus available."))
        else:
            st.info("No detailed insight found for this segment.")
    else:
        st.info("Detailed insight file not found. Showing basic segment description.")

    st.markdown("---")

    # ------------------------------------------------------
    # Raw Data Preview and Download
    # ------------------------------------------------------

    st.subheader("📄 Filtered Buyer Data Preview")
    st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_buyer_segments.csv",
        mime="text/csv"
    )

# ==========================================================
# Mode 2: Summary-Only Dashboard
# ==========================================================

else:

    st.warning(
        "Full buyer-level data was not detected. "
        "Running dashboard in summary-only mode."
    )

    # Choose best available summary table
    if insights_df is not None and summary_columns.issubset(set(insights_df.columns)):
        summary_df = insights_df.copy()
        source_name = insights_path
    elif visual_summary_df is not None and summary_columns.issubset(set(visual_summary_df.columns)):
        summary_df = visual_summary_df.copy()
        source_name = visual_summary_path
    elif segment_summary_df is not None and "buyer_segment" in segment_summary_df.columns:
        summary_df = segment_summary_df.copy()
        source_name = segment_summary_path
    elif is_summary_file:
        summary_df = buyer_df.copy()
        source_name = buyer_path
    else:
        st.error(
            "Could not find usable summary columns. "
            "Please upload buyer_profile_with_segments.csv or final_segment_insights.csv."
        )
        st.write("Available columns in current file:")
        st.write(list(buyer_df.columns))
        st.stop()

    st.info(f"Using summary file: `{source_name}`")

    st.subheader("📌 Segment Summary Metrics")

    total_buyers = summary_df["buyer_count"].sum() if "buyer_count" in summary_df.columns else "N/A"

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Buyers", f"{total_buyers:,}" if total_buyers != "N/A" else "N/A")

    if "avg_total_investment" in summary_df.columns:
        col2.metric("Highest Avg Investment", safe_currency(summary_df["avg_total_investment"].max()))
    else:
        col2.metric("Highest Avg Investment", "N/A")

    if "avg_property_price" in summary_df.columns:
        col3.metric("Highest Avg Property Price", safe_currency(summary_df["avg_property_price"].max()))
    else:
        col3.metric("Highest Avg Property Price", "N/A")

    st.markdown("---")

    st.subheader("📊 Buyer Segment Summary")
    st.dataframe(summary_df, use_container_width=True)

    if "buyer_count" in summary_df.columns:
        show_bar_chart(
            summary_df.sort_values("buyer_count", ascending=False),
            "buyer_segment",
            "buyer_count",
            "Buyer Count by Segment"
        )

    if "avg_total_investment" in summary_df.columns:
        show_bar_chart(
            summary_df.sort_values("avg_total_investment", ascending=False),
            "buyer_segment",
            "avg_total_investment",
            "Average Total Investment by Segment"
        )

    if "avg_property_price" in summary_df.columns:
        show_bar_chart(
            summary_df.sort_values("avg_property_price", ascending=False),
            "buyer_segment",
            "avg_property_price",
            "Average Property Price by Segment"
        )

    if "avg_properties_bought" in summary_df.columns:
        show_bar_chart(
            summary_df.sort_values("avg_properties_bought", ascending=False),
            "buyer_segment",
            "avg_properties_bought",
            "Average Properties Bought by Segment"
        )

    st.markdown("---")

    st.subheader("🧠 Segment Insights")

    if "buyer_segment" in summary_df.columns:
        selected_segment = st.selectbox(
            "Select Buyer Segment",
            sorted(summary_df["buyer_segment"].dropna().unique())
        )

        selected_row = summary_df[summary_df["buyer_segment"] == selected_segment].iloc[0]

        st.write(f"## {selected_segment}")

        if "segment_profile" in summary_df.columns:
            st.info(selected_row.get("segment_profile", ""))

        if "business_opportunity" in summary_df.columns:
            st.success(selected_row.get("business_opportunity", ""))

        if "recommended_strategy" in summary_df.columns:
            st.warning(selected_row.get("recommended_strategy", ""))

        if "marketing_channel_focus" in summary_df.columns:
            st.write("### Marketing Channel Focus")
            st.write(selected_row.get("marketing_channel_focus", ""))

    st.markdown("---")

    csv = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Segment Summary as CSV",
        data=csv,
        file_name="segment_summary_dashboard.csv",
        mime="text/csv"
    )

# ==========================================================
# Optional: K-Means Evaluation Section
# ==========================================================

if kmeans_eval_df is not None and {"k", "inertia", "silhouette_score"}.issubset(set(kmeans_eval_df.columns)):
    st.markdown("---")
    st.subheader("📈 K-Means Evaluation Results")

    st.dataframe(kmeans_eval_df, use_container_width=True)

    show_bar_chart(
        kmeans_eval_df,
        "k",
        "silhouette_score",
        "Silhouette Score by Number of Clusters"
    )
