import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from data_loader import load_and_prepare_data


st.set_page_config(page_title="Crime Analysis Dashboard", layout="wide")
st.title("Crime Against Women Dashboard (2001–2021)")


@st.cache_data
def get_data():
    return load_and_prepare_data()


data1, long_df, crime_columns = get_data()


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

year_min = int(data1["Year"].min())
year_max = int(data1["Year"].max())

year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)

selected_states = st.sidebar.multiselect(
    "States",
    options=sorted(data1["State"].unique()),
    default=[]
)

selected_types = st.sidebar.multiselect(
    "Crime types for trend charts",
    options=crime_columns,
    default=crime_columns
)

n_clusters = st.sidebar.slider("Clusters (KMeans)", 2, 8, 4, 1)

show_elbow = st.sidebar.checkbox("Show elbow chart", value=False)


filtered_data = data1[(data1["Year"] >= year_range[0]) & (data1["Year"] <= year_range[1])].copy()
filtered_long = long_df[(long_df["Year"] >= year_range[0]) & (long_df["Year"] <= year_range[1])].copy()

if len(selected_states) > 0:
    filtered_data = filtered_data[filtered_data["State"].isin(selected_states)]
    filtered_long = filtered_long[filtered_long["State"].isin(selected_states)]

if len(selected_types) > 0:
    filtered_long = filtered_long[filtered_long["Type of Crime"].isin(selected_types)]


# -----------------------------
# KPI SECTION
# -----------------------------
st.header("Key Metrics")

# Custom column widths
col1, col2, col3, col4 = st.columns([1.2, 1.2, 2.2, 1.4])

total_crimes = int(filtered_data["Total Crimes"].sum())

yearly_total = filtered_data.groupby("Year")["Total Crimes"].sum()
highest_year = int(yearly_total.idxmax()) if len(yearly_total) > 0 else None

state_total = filtered_data.groupby("State")["Total Crimes"].sum()
top_state = state_total.idxmax() if len(state_total) > 0 else None

avg_yearly = int(yearly_total.mean()) if len(yearly_total) > 0 else 0


col1.metric("Total Crimes", f"{total_crimes:,}")

col2.metric("Highest Crime Year",
            "-" if highest_year is None else str(highest_year))

col3.metric("Highest Crime State",
            "-" if top_state is None else str(top_state))

col4.metric("Average per Year",
            f"{avg_yearly:,}")


# -----------------------------
# DATA PREVIEW
# -----------------------------
st.header("Dataset Preview")
st.dataframe(filtered_data.head(20))


# -----------------------------
# TOTAL CRIMES BY YEAR
# -----------------------------
st.header("National Crime Trend")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(yearly_total.index, yearly_total.values, marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Total Crimes")

years = list(yearly_total.index)
ax.set_xticks(years[::2] if len(years) > 10 else years)

plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# TOP STATES
# -----------------------------
st.header("Top States")

top_states = filtered_data.groupby("State")["Total Crimes"].sum().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
top_states.plot(kind="bar", ax=ax)
ax.set_xlabel("State")
ax.set_ylabel("Total Crimes")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# CRIME TYPE DISTRIBUTION
# -----------------------------
st.header("Crime Type Distribution")

type_totals = filtered_data[crime_columns].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
type_totals.plot(kind="bar", ax=ax)
ax.set_xlabel("Crime Type")
ax.set_ylabel("Total Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# CRIME TREND BY TYPE
# -----------------------------
st.header("Crime Trend Over Time by Type")

trend_types = sorted(filtered_long["Type of Crime"].unique())

fig, ax = plt.subplots(figsize=(12, 6))
for t in trend_types:
    series = filtered_long[filtered_long["Type of Crime"] == t].groupby("Year")["Crime Count"].sum()
    ax.plot(series.index, series.values, marker="o", label=t)

trend_years = sorted(filtered_long["Year"].unique())
ax.set_xticks(trend_years[::2] if len(trend_years) > 10 else trend_years)

ax.set_xlabel("Year")
ax.set_ylabel("Crime Count")
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# CRIME GROWTH RATE
# -----------------------------
st.header("Crime Growth Rate Over Time")

growth_df = filtered_long.sort_values(["Type of Crime", "Year"]).copy()
growth_df["Crime Growth Rate"] = growth_df.groupby("Type of Crime")["Crime Count"].pct_change() * 100

fig, ax = plt.subplots(figsize=(12, 6))
for t in trend_types:
    series = growth_df[growth_df["Type of Crime"] == t].groupby("Year")["Crime Growth Rate"].mean()
    ax.plot(series.index, series.values, marker="o", label=t)

growth_years = sorted(growth_df["Year"].unique())
ax.set_xticks(growth_years[::2] if len(growth_years) > 10 else growth_years)

ax.set_xlabel("Year")
ax.set_ylabel("Growth Rate (%)")
ax.legend(loc="upper left")
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.header("Correlation Heatmap")

corr = filtered_data[crime_columns].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# CLUSTERING
# -----------------------------
st.header("State Clustering")

state_profile = (
    filtered_data.groupby("State")[crime_columns]
    .mean()
    .reset_index()
)

if len(state_profile) < n_clusters:
    st.write("Not enough states in the filter for the selected cluster count.")
else:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(state_profile[crime_columns])

    if show_elbow:
        wcss = []
        max_k = min(10, len(state_profile))
        ks = list(range(1, max_k + 1))

        for k in ks:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(scaled)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, wcss, marker="o")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    state_profile["Cluster"] = kmeans.fit_predict(scaled)

    st.subheader("Cluster Assignments")
    st.dataframe(state_profile.sort_values(["Cluster", "State"]))

    st.subheader("Cluster Profiles (Mean)")
    cluster_summary = state_profile.groupby("Cluster")[crime_columns].mean()
    st.dataframe(cluster_summary)

    st.subheader("Cluster Scatter View")
    x_feat = st.selectbox("X axis", crime_columns, index=0)
    y_feat = st.selectbox("Y axis", crime_columns, index=min(1, len(crime_columns) - 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=state_profile,
        x=x_feat,
        y=y_feat,
        hue="Cluster",
        s=100,
        ax=ax
    )
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------
# FORECASTING
# -----------------------------
st.header("Forecast (Linear Regression)")

yearly_total_full = filtered_data.groupby("Year")["Total Crimes"].sum()

if len(yearly_total_full) < 3:
    st.write("Not enough yearly points for forecasting.")
else:
    X = yearly_total_full.index.values.reshape(-1, 1)
    y = yearly_total_full.values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.array(range(int(yearly_total_full.index.max()) + 1, int(yearly_total_full.index.max()) + 6)).reshape(-1, 1)
    predictions = model.predict(future_years)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly_total_full.index, yearly_total_full.values, marker="o", label="Historical")
    ax.plot(future_years.flatten(), predictions, marker="o", linestyle="--", label="Forecast")

    all_years = np.concatenate([yearly_total_full.index, future_years.flatten()])
    ax.set_xticks(all_years[::2] if len(all_years) > 10 else all_years)

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Crimes")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Forecast Values")
    forecast_table = pd.DataFrame({"Year": future_years.flatten(), "Predicted Total Crimes": predictions.astype(int)})
    st.dataframe(forecast_table)