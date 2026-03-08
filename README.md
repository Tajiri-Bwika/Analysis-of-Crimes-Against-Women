# Crime Against Women Analysis and Dashboard

## Project Overview

This project analyzes crimes against women using a dataset covering the years 2001 to 2021. The work includes data cleaning, statistical analysis, machine learning, and an interactive dashboard.

The project produces three main outputs.

• Exploratory data analysis
• Machine learning analysis and forecasting
• Interactive dashboard for visual exploration

The dataset includes crimes such as rape, kidnapping, dowry deaths, assault, modesty crimes, domestic violence, and trafficking.

The goal focuses on understanding crime patterns, identifying high-risk regions, and forecasting trends.

---

## Repository Structure

```
repository
│
README.md
│
Dataset
   CrimesOnWomenData.csv
   description.csv
│
Python codes
   srs.py
   data_loader.py
   apps.py
```

### Dataset

Contains the raw data used in the analysis.

• **CrimesOnWomenData.csv**
Main dataset containing yearly crime statistics.

• **description.csv**
Provides explanations for abbreviated column names.

---

### Python Codes

#### srs.py

Main analysis script.

Functions performed inside the script:

• Data loading and inspection
• Data cleaning and preprocessing
• Feature engineering
• Statistical analysis
• Data visualization
• Cluster analysis using KMeans
• Forecasting using Linear Regression

The script loads the dataset, renames columns, calculates total crimes, and converts the dataset into long format for analysis. 

Key analysis produced:

• total crimes across years
• top crime categories
• state crime distribution
• crime trends over time
• correlation heatmap
• clustering of states based on crime patterns
• future crime predictions

---

#### data_loader.py

Utility module used for loading and preparing the dataset for the dashboard.

The script performs several preprocessing steps:

• reads the dataset
• renames columns
• cleans state names
• calculates total crimes
• converts data into long format

The function `load_and_prepare_data()` returns three objects.

• cleaned dataset
• long format dataset
• list of crime columns 

This module improves code organization and avoids repeating preprocessing steps.

---

#### apps.py

This script runs the interactive dashboard using Streamlit.

The dashboard includes:

• filtering by year and state
• crime trend visualization
• state comparison charts
• correlation heatmap
• clustering visualization
• crime growth rate analysis
• crime forecasting

The dashboard loads processed data through the data loader module and generates charts dynamically. 

Users interact with filters in the sidebar to explore crime patterns.

---

## Key Analytical Methods

### 1. Data Cleaning

The dataset required preprocessing before analysis.

Steps performed:

• renamed unclear column names
• handled formatting issues
• standardized state names
• calculated total crimes per record

---

### 2. Data Visualization

Charts help identify patterns and trends.

Visualizations include:

• yearly crime trend
• crime distribution by type
• top states by crime count
• crime growth rate

Libraries used:

• Matplotlib
• Seaborn

---

### 3. Cluster Analysis

KMeans clustering groups states with similar crime patterns.

Steps:

• calculate average crime values per state
• standardize features
• apply KMeans clustering
• visualize clusters

Clusters reveal states with similar crime characteristics.

---

### 4. Forecasting

Linear regression predicts future crime counts.

Steps:

• compute yearly total crimes
• train regression model
• predict future crime levels

Forecast helps estimate crime trends for upcoming years.

---

## Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit

---

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/yourusername/repository-name.git
```

Enter the project folder.

```
cd repository-name
```

---

### 2. Install required packages

```
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

---

### 3. Run the analysis script

```
python "Python codes/srs.py"
```

The script performs full data analysis and displays graphs.

---

### 4. Run the dashboard

```
streamlit run "Streamlit codes/apps.py"
```

Streamlit launches the dashboard in your browser.

You interact with filters and explore the dataset visually.

---

## Expected Output

The project generates:

• statistical insights on crime trends
• visual charts explaining patterns
• machine learning clustering results
• predicted crime levels
• interactive dashboard

---

## Future Improvements

Possible enhancements include:

• adding deep learning prediction models
• integrating real-time crime datasets
• deploying the dashboard online
• adding geographic crime mapping

---

## Author

Project created for data science analysis of crimes against women using machine learning and interactive visualization tools.

