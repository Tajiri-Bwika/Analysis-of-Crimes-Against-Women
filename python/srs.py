import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the datasets
data1 = pd.read_csv('CrimesOnWomenData.csv')
data2 = pd.read_csv('description.csv')

# Data inspection
print(data1.head())
print(data1.info())
print(data1.describe())
print(data2.head())
print(data2.info())


# Data Cleaning (Preprocessing)
# Rename Unnamed column
data1 = data1.rename(columns={"Unnamed: 0": "NO"})

# Rename short crime codes using description dataset
column_mapping = dict(zip(data2["Column Names"], data2["Explanation"]))
data1 = data1.rename(columns=column_mapping)

print(data1.head())

# Check for missing values
print("Missing values in data1:")
print(data1.isnull().sum())


# Feature Engineering
crime_columns = data1.columns[3:]  # crime columns start after NO, State, Year

# Create Total Crimes column
data1["Total Crimes"] = data1[crime_columns].sum(axis=1)

print(data1.head())

# Convert from wide to long format
long_df = data1.melt(
    id_vars=["NO", "State", "Year"],
    value_vars=crime_columns,
    var_name="Type of Crime",
    value_name="Crime Count"
)

print(long_df.head())


# Overall Analysis

total_crimes = data1["Total Crimes"].sum()
print(f"Total Crimes (2001-2021): {total_crimes}")

total_crimes_by_type = long_df.groupby("Type of Crime")["Crime Count"].sum().sort_values(ascending=False)
print("Total Crimes by Type:")
print(total_crimes_by_type)

total_crimes_by_year = data1.groupby("Year")["Total Crimes"].sum()
plt.figure(figsize=(12,6))
plt.plot(total_crimes_by_year.index,
         total_crimes_by_year.values,
         marker='o')
plt.xticks(total_crimes_by_year.index, rotation=45)
plt.title("Total Crimes by Year")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.grid()
plt.tight_layout()
plt.show()


# Top 5 crime types
top_5_crimes = total_crimes_by_type.head(5)
print("Top 5 Crime Types:")
print(top_5_crimes)



# Crime Distribution by State
crimes_by_state = data1.groupby("State")["Total Crimes"].sum().sort_values(ascending=False)
plt.figure(figsize=(14,8))
crimes_by_state.head(15).plot(kind='bar')
plt.title("Top 15 States by Total Crimes")
plt.xlabel("State")
plt.ylabel("Total Crimes")
plt.xticks(rotation=60, ha='right')
plt.grid()
plt.tight_layout()
plt.show()



# Crime Trend Over Time by Type
plt.figure(figsize=(14,7))

for crime_type in long_df["Type of Crime"].unique():
    crime_trend = long_df[long_df["Type of Crime"] == crime_type] \
        .groupby("Year")["Crime Count"].sum()
    plt.plot(crime_trend.index,
             crime_trend.values,
             marker='o',
             label=crime_type)
years = sorted(long_df["Year"].unique())
plt.xticks(years, rotation=45)
plt.title("Crime Trend Over Time by Type")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Highest crime year
highest_crime_year = total_crimes_by_year.idxmax()
print(f"Year with the highest crime: {highest_crime_year}")


# Crime Growth Rate
long_df["Crime Growth Rate"] = long_df.groupby("Type of Crime")["Crime Count"].pct_change() * 100

plt.figure(figsize=(14,7))
for crime_type in long_df["Type of Crime"].unique():

    growth = long_df[long_df["Type of Crime"] == crime_type] \
        .groupby("Year")["Crime Growth Rate"].mean()

    plt.plot(growth.index,
             growth.values,
             marker='o',
             label=crime_type)

years = sorted(long_df["Year"].unique())
plt.xticks(years, rotation=45)
plt.title("Crime Growth Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()


# Correlation Analysis
correlation_matrix = data1[crime_columns].corr()

plt.figure(figsize=(8,6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# -----------------------------
# Cluster Analysis (State-Level)
# -----------------------------

# Create state profiles (ONE ROW PER STATE)
state_profile = (
    data1.groupby("State")[crime_columns]
    .mean()
    .reset_index()
)

# Standardize data
scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    state_profile[crime_columns]
)

# Elbow Method
wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, random_state=42)

    kmeans.fit(scaled_features)

    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10,6))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method for Optimal Clusters")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid()

plt.show()


# Choose optimal clusters
kmeans = KMeans(n_clusters=4, random_state=42)

clusters = kmeans.fit_predict(scaled_features)

state_profile["Cluster"] = clusters


print("\nCluster Assignment:")

print(state_profile.sort_values("Cluster"))


# -----------------------------
# Cluster Characteristics
# -----------------------------

cluster_summary = state_profile.groupby("Cluster")[crime_columns].mean()

print("\nCluster Profiles:")

print(cluster_summary)


# -----------------------------
# Cluster Visualization
# -----------------------------

plt.figure(figsize=(10,6))

sns.scatterplot(

    x=state_profile["No. of Rape cases"],

    y=state_profile["Domestic violence"],

    hue=state_profile["Cluster"],

    s=100

)

plt.title("State Clusters Based on Crime Patterns")

plt.xlabel("Average Rape Cases")

plt.ylabel("Average Domestic Violence Cases")

plt.grid()

plt.show()


# Predictive Modeling (Linear Regression)
yearly_total = data1.groupby("Year")["Total Crimes"].sum()

# Prepare training data
X = yearly_total.index.values.reshape(-1, 1)
y = yearly_total.values

# Train model
model = LinearRegression()
model.fit(X, y)

# Create future years
future_years = np.array(range(2022, 2027)).reshape(-1, 1)

# Predict
predictions = model.predict(future_years)

# Plot results
plt.figure(figsize=(12, 6))

# Historical data
plt.plot(yearly_total.index,
         yearly_total.values,
         marker='o',
         label="Historical")

# Forecasted data
plt.plot(future_years.flatten(),
         predictions,
         marker='o',
         linestyle='--',
         label="Forecast")

# Force integer year ticks (Fix for 2027.5 issue)
all_years = list(yearly_total.index) + list(future_years.flatten())
plt.xticks(all_years, rotation=45)

plt.title("Crime Forecast (Linear Regression)")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.legend()
plt.grid()
plt.show()

# Print predictions
print("Predicted Crimes for 2022–2027:")
for year, value in zip(future_years.flatten(), predictions):
    print(f"{year}: {int(value)}")

