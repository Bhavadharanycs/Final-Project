import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI Design
st.set_page_config(page_title="Insurance Claim Prediction", layout="wide")

st.title("🚗 Insurance Claim Prediction")
st.markdown("""
This application performs an **Exploratory Data Analysis (EDA)** and predicts the **amount of claim** (`CLM_AMT`)  
based on the given insurance data.
""")

# Load Predefined Dataset
@st.cache
def load_data():
    data = pd.read_csv("car_insurance_claim.csv")  # Ensure the dataset file is in the same directory
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

# Model Training Section
st.header("Insurance Claim Prediction")

# Split data into features and target
features = preprocessed_data.drop(columns=['CLM_AMT', 'ID'])
target = preprocessed_data['CLM_AMT']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write("### Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# User Inputs for Prediction
st.write("### Predict Claim Amount")
input_data = {}
for col in features.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X_train[col].median()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"**Predicted Claim Amount:** ${prediction[0]:,.2f}")
