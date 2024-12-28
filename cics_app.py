import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit UI Design
st.set_page_config(page_title="Car Insurance EDA", layout="wide")

st.title("🚗 Car Insurance EDA Analysis")
st.markdown("""
This application performs an **Exploratory Data Analysis (EDA)** on car insurance claim data.  
Explore correlations, distributions, and relationships between variables.
""")

# Load Predefined Dataset
@st.cache
def load_data():
    # Replace this with the path to your dataset if it's stored locally
    data = pd.read_csv("car_insurance_claim.csv")  # Ensure this file is in the same directory or provide a full path.
    return data

# Load data
data = load_data()

# Preprocessing Function
def preprocess_data(df):
    def clean_currency_columns(df, columns):
        for column in columns:
            df[column] = df[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)
        return df

    # Clean numeric columns
    currency_columns = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    df = clean_currency_columns(df, currency_columns)

    # Convert BIRTH to age
    current_year = 2024
    df['BIRTH_YEAR'] = np.where(df['BIRTH'].str[-2:].astype(int) > 23,
                                1900 + df['BIRTH'].str[-2:].astype(int),
                                2000 + df['BIRTH'].str[-2:].astype(int))
    df['CALCULATED_AGE'] = current_year - df['BIRTH_YEAR']
    df.drop(columns=['BIRTH', 'BIRTH_YEAR'], inplace=True)

    # Convert categorical variables
    binary_columns = ['PARENT1', 'MSTATUS', 'GENDER', 'RED_CAR', 'REVOKED']
    df[binary_columns] = df[binary_columns].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

    # Handle missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df_encoded

# Preprocess data
preprocessed_data = preprocess_data(data)

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(preprocessed_data.corr(), cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# Distribution of Target Variable (CLM_AMT)
st.subheader("Distribution of Claim Amount (CLM_AMT)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(preprocessed_data['CLM_AMT'], kde=True, bins=30, ax=ax)
ax.set_title("Distribution of CLM_AMT")
st.pyplot(fig)

# Pairplot for Selected Variables
st.subheader("Pairplot for Selected Variables")
selected_cols = st.multiselect("Choose columns for pairplot", options=preprocessed_data.columns, default=['CLM_AMT', 'INCOME', 'BLUEBOOK', 'CALCULATED_AGE'])
if len(selected_cols) > 1:
    pairplot_fig = sns.pairplot(preprocessed_data[selected_cols], diag_kind='kde', corner=True)
    st.pyplot(pairplot_fig)

# Outlier Handling
st.subheader("Outliers in Target Variable (CLM_AMT)")
Q1 = preprocessed_data['CLM_AMT'].quantile(0.25)
Q3 = preprocessed_data['CLM_AMT'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = preprocessed_data[(preprocessed_data['CLM_AMT'] < lower_bound) | (preprocessed_data['CLM_AMT'] > upper_bound)]

st.write(f"Number of outliers: {len(outliers)}")
st.write(f"Outliers are values outside the range [{lower_bound:.2f}, {upper_bound:.2f}].")

# Show Outliers
if st.checkbox("Show Outlier Rows"):
    st.dataframe(outliers)

# Distribution of Numerical Columns
st.subheader("Distribution of Numerical Variables")
numeric_columns = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns
selected_numeric = st.selectbox("Choose a numerical column", options=numeric_columns, index=0)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(preprocessed_data[selected_numeric], kde=True, bins=30, ax=ax)
ax.set_title(f"Distribution of {selected_numeric}")
st.pyplot(fig)
