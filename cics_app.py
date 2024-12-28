import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to clean numeric columns with "$" and "," symbols
def clean_currency_columns(df, columns):
    for column in columns:
        df[column] = df[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)
    return df

# Function to preprocess the dataset
def preprocess_data(file_path):
    df = pd.read_csv('car_insurance_claim.csv')

    # Clean currency columns
    currency_columns = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    df = clean_currency_columns(df, currency_columns)

    # Convert date of birth to age
    current_year = 2024
    df['BIRTH_YEAR'] = np.where(df['BIRTH'].str[-2:].astype(int) > 23, 
                                1900 + df['BIRTH'].str[-2:].astype(int), 
                                2000 + df['BIRTH'].str[-2:].astype(int))
    df['CALCULATED_AGE'] = current_year - df['BIRTH_YEAR']
    df.drop(columns=['BIRTH', 'BIRTH_YEAR'], inplace=True)

    # Convert binary categorical columns
    binary_columns = ['PARENT1', 'MSTATUS', 'GENDER', 'RED_CAR', 'REVOKED']
    df[binary_columns] = df[binary_columns].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

    # Handle missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df_encoded

# Load and preprocess dataset
st.title("Car Insurance Claim Prediction")
st.write("Upload the dataset to start analysis and make predictions.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    data = preprocess_data(uploaded_file)
    st.write("Dataset after preprocessing:")
    st.dataframe(data.head())

    # Splitting data into features and target
    features = data.drop(columns=['CLM_AMT', 'ID'])
    target = data['CLM_AMT']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Evaluation Metrics")
    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
    st.write(f"**R² Score:** {r2}")

    # Predict for new input
    st.write("### Predict Insurance Claim Amount")
    input_data = {}
    for col in features.columns:
        input_data[col] = st.number_input(f"{col}", value=float(X_train[col].median()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.write(f"**Predicted Claim Amount:** ${prediction[0]:,.2f}")
