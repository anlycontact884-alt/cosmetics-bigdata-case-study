import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Cosmetics E-commerce Big Data Dashboard",
    page_icon="💄",
    layout="wide",
)

DATA_DIR = os.path.dirname(__file__)

def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        st.error(f"Missing data file: {name}")
        st.stop()
    return pd.read_csv(path)

@st.cache_data
def load_all():
    return {
        "kpi": load_csv("result_kpi.csv"),
        "events": load_csv("result_event_behavior.csv"),
        "funnel": load_csv("result_conversion_funnel.csv"),
        "top_views": load_csv("result_top_view_products.csv"),
        "top_purchases": load_csv("result_top_purchase_products.csv"),
        "brands": load_csv("result_top_brands.csv"),
        "segments": load_csv("result_customer_segments.csv"),
        "kmeans": load_csv("result_kmeans_metrics.csv"),
    }

data = load_all()
kpi = data["kpi"]

def kpi_value(label):
    row = kpi.loc[kpi["KPI"] == label, "Value"]
    return row.iloc[0] if not row.empty else None

st.title("💄 Cosmetics E-commerce Big Data Dashboard")
st.caption(
    "Customer Shopping Behavior Analysis in an Online Cosmetics Store "
    "using PySpark + Spark MLlib"
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose analysis",
    [
        "📊 Overview",
        "🛒 Conversion Funnel",
        "🧴 Product & Brand",
        "👥 Customer Segmentation",
        "📌 Business Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Data source: eCommerce Events History in Cosmetics Shop — Kaggle. "
    "Dashboard values are generated from the full-dataset PySpark analysis."
)

if page == "📊 Overview":
    st.subheader("Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", f"{int(float(kpi_value('Total events'))):,}")
    c2.metric("Unique Users", f"{int(float(kpi_value('Unique users'))):,}")
    c3.metric("Purchase Users", f"{int(float(kpi_value('Users with purchase'))):,}")
    c4.metric("Purchase Revenue", f"{float(kpi_value('Total purchase revenue')):,.2f}")

    st.markdown("### Customer Event Behavior")
    events = data["events"].copy()
    fig = px.bar(
        events,
        x="event_type",
        y="event_count",
        text_auto=".2s",
        title="Distribution of Customer Events",
    )
    fig.update_layout(xaxis_title="Event Type", yaxis_title="Number of Events")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Findings")
    st.write(
        "The dataset contains more than 20 million customer events. "
        "Viewing is the dominant interaction, while purchase events represent "
        "a much smaller share of total activity. This indicates a substantial "
        "gap between product interest and completed transactions."
    )

elif page == "🛒 Conversion Funnel":
    st.subheader("Customer Conversion Funnel")

    funnel = data["funnel"].copy()
    fig = px.funnel(
        funnel,
        y="stage",
        x="users",
        title="View → Cart → Purchase",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    users = dict(zip(funnel["stage"], funnel["users"]))
    view = users.get("View", 0)
    cart = users.get("Cart", 0)
    purchase = users.get("Purchase", 0)

    c1.metric("View Users", f"{int(view):,}")
    c2.metric("View → Cart", f"{cart / view * 100:.2f}%" if view else "0%")
    c3.metric("View → Purchase", f"{purchase / view * 100:.2f}%" if view else "0%")

    st.markdown("### Funnel Data")
    display_funnel = funnel.copy()
    display_funnel["users"] = display_funnel["users"].map(lambda x: f"{int(x):,}")
    display_funnel["conversion_from_previous"] = display_funnel[
        "conversion_from_previous"
    ].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
    st.dataframe(display_funnel, use_container_width=True, hide_index=True)

    st.warning(
        "Method note: this is a user-based funnel. It checks whether a user "
        "has each event type; it does not enforce the exact chronological order "
        "of View → Cart → Purchase."
    )

elif page == "🧴 Product & Brand":
    st.subheader("Product & Brand Analysis")

    tab1, tab2, tab3 = st.tabs(
        ["Top Viewed Products", "Top Purchased Products", "Top Brands"]
    )

    with tab1:
        df = data["top_views"].copy()

        # Treat product_id as categorical rather than numeric
        df["product_id"] = df["product_id"].astype(str)

        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df.sort_values("view_count"),
            x="view_count",
            y="product_id",
            orientation="h",
            text_auto=".2s",
            title="Top 10 Viewed Products",
        )

        fig.update_layout(
            xaxis_title="Number of Views",
            yaxis_title="Product ID",
            yaxis=dict(type="category"),
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df = data["top_purchases"].copy()

        # Treat product_id as categorical
        df["product_id"] = df["product_id"].astype(str)

        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df.sort_values("revenue"),
            x="revenue",
            y="product_id",
            orientation="h",
            text_auto=".2f",
            title="Top 10 Products by Purchase Revenue",
        )

        fig.update_layout(
            xaxis_title="Purchase Revenue",
            yaxis_title="Product ID",
            yaxis=dict(type="category"),
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df = data["brands"].copy()
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = px.bar(
            df.sort_values("revenue"),
            x="revenue",
            y="brand",
            orientation="h",
            text_auto=".2f",
            title="Top 10 Brands by Purchase Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "👥 Customer Segmentation":
    st.subheader("Customer Segmentation — K-Means")

    metrics = data["kmeans"].copy()
    best_k = int(kpi_value("Best K"))

    c1, c2 = st.columns(2)
    c1.metric("Selected K", best_k)
    best_sil = metrics.loc[metrics["k"] == best_k, "silhouette"]
    c2.metric(
        "Best Silhouette",
        f"{float(best_sil.iloc[0]):.4f}" if not best_sil.empty else "N/A",
    )

    st.markdown("### Model Selection")
    fig = px.line(
        metrics,
        x="k",
        y="silhouette",
        markers=True,
        title="Silhouette Score by K",
    )
    fig.update_layout(xaxis_title="K", yaxis_title="Silhouette Score")
    st.plotly_chart(fig, use_container_width=True)

   st.markdown("### Segment Profile")

seg = data["segments"].copy()

# Hide the event-based conversion rate because it can exceed 100%
# when calculated from purchase events / view events.
display_seg = seg.drop(
    columns=["avg_conversion_rate"],
    errors="ignore"
)

st.dataframe(
    display_seg,
    use_container_width=True,
    hide_index=True
)
    fig = px.bar(
        seg,
        x="cluster",
        y="avg_spending",
        text_auto=".2f",
        title="Average Spending by Customer Segment",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation")
    if len(seg) == 2:
        low = seg.sort_values("avg_spending").iloc[0]
        high = seg.sort_values("avg_spending").iloc[-1]
        st.write(
            f"**Cluster {int(low['cluster'])}** represents the lower-engagement/lower-value "
            f"group, with average spending of {low['avg_spending']:.2f}. "
            f"**Cluster {int(high['cluster'])}** represents the highly engaged/high-value "
            f"group, with average spending of {high['avg_spending']:.2f}."
        )

elif page == "📌 Business Insights":
    st.subheader("Business Insights & Recommendations")

    events = data["events"]
    funnel = data["funnel"]
    brands = data["brands"]
    seg = data["segments"]

    view_events = events.loc[events["event_type"] == "view", "event_count"].sum()
    purchase_events = events.loc[
        events["event_type"] == "purchase", "event_count"
    ].sum()
    total_events = events["event_count"].sum()

    view_users = funnel.loc[funnel["stage"] == "View", "users"].iloc[0]
    cart_users = funnel.loc[funnel["stage"] == "Cart", "users"].iloc[0]
    purchase_users = funnel.loc[funnel["stage"] == "Purchase", "users"].iloc[0]

    st.markdown("### 1. Engagement vs. Purchase")
    st.write(
        f"View events account for approximately {view_events / total_events * 100:.2f}% "
        f"of all events, while purchase events account for only "
        f"{purchase_events / total_events * 100:.2f}%. This indicates a large "
        f"gap between product interest and completed purchases."
    )

    st.markdown("### 2. Conversion Opportunity")
    st.write(
        f"The View → Cart rate is {cart_users / view_users * 100:.2f}%, while "
        f"the View → Purchase rate is {purchase_users / view_users * 100:.2f}%. "
        "The business should prioritize reducing abandonment between product "
        "interest, cart addition, and final purchase."
    )

    st.markdown("### 3. Brand Prioritization")
    if not brands.empty:
        top_brand = brands.iloc[0]
        st.write(
            f"{top_brand['brand']} is the highest-revenue brand in the Top 10 list, "
            f"with revenue of {top_brand['revenue']:,.2f}. This brand can be "
            "considered for promotional, cross-selling, and retention campaigns."
        )

    st.markdown("### 4. Customer Segmentation")
    if not seg.empty:
        high = seg.loc[seg["avg_spending"].idxmax()]
        st.write(
            f"Cluster {int(high['cluster'])} is the highest-value segment, with "
            f"average spending of {high['avg_spending']:,.2f}, average purchases "
            f"of {high['avg_purchases']:.2f}, and average views of "
            f"{high['avg_views']:.2f}. This segment should receive personalized "
            "retention and loyalty strategies."
        )

    st.markdown("### Recommended Actions")
    st.markdown(
        """
        - **Reduce cart abandonment:** test free-shipping thresholds, coupons,
          price incentives, and a simpler checkout flow.
        - **Retarget high-intent users:** use View/Cart behavior for personalized
          remarketing.
        - **Develop high-value customers:** loyalty benefits, bundles, early-access
          promotions, and personalized recommendations.
        - **Prioritize high-revenue brands:** allocate promotional budget based on
          both purchase volume and revenue contribution.
        """
    )

st.markdown("---")
st.caption(
    "Big Data Case Study | PySpark + Spark MLlib + Streamlit | "
    "Full-dataset results"
)
