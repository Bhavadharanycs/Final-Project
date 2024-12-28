import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit UI Design
st.set_page_config(page_title="Car Insurance Claim Prediction", layout="wide")

st.title("🚗 Car Insurance Claim Prediction")
st.markdown("""
This application helps analyze car insurance claim data, handle missing values and outliers, and predict insurance claim amounts.  
Interact with the app to discover insights and make predictions.
""")

# Load Predefined Dataset
@st.cache
def load_data():
    # Replace this with the path to your dataset if it's stored locally
    data = pd.read_csv("car_insurance_claim.csv")  # Ensure this file is in the same directory or provide a full path.
    return data

# Load data
st.header("1️⃣ Data Overview")
data = load_data()
st.write("### Dataset Sample:")
st.dataframe(data.head())

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
st.write("### Preprocessed Dataset:")
st.dataframe(preprocessed_data.head())

# EDA Section
st.header("2️⃣ Exploratory Data Analysis (EDA)")
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(preprocessed_data.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("### Distribution of Target Variable (CLM_AMT)")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(preprocessed_data['CLM_AMT'], kde=True, ax=ax, bins=30)
ax.set_title("Distribution of CLM_AMT")
st.pyplot(fig)

# Outlier Handling
Q1 = preprocessed_data['CLM_AMT'].quantile(0.25)
Q3 = preprocessed_data['CLM_AMT'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
preprocessed_data['CLM_AMT'] = np.clip(preprocessed_data['CLM_AMT'], lower_bound, upper_bound)

st.write("### Outliers Handled in Target Variable")
st.write(f"Values outside [{lower_bound:.2f}, {upper_bound:.2f}] have been capped.")

# Prediction Section
st.header("3️⃣ Prediction")
features = preprocessed_data.drop(columns=['CLM_AMT', 'ID'])
target = preprocessed_data['CLM_AMT']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
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

# Predict on new input
st.write("### Predict Claim Amount")
input_data = {}
for col in features.columns:
    input_data[col] = st.number_input(f"{col}", value=float(X_train[col].median()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"**Predicted Claim Amount:** ${prediction[0]:,.2f}")
