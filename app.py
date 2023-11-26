import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load CPI data
cpi_data = pd.read_csv('CPI.csv')
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Streamlit app
st.title('Inflation and Financial Data Analysis')

# Upload file
uploaded_file = st.file_uploader("Upload Quarterly Financial Data File", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read uploaded file
    financial_data = pd.read_excel(uploaded_file, index_col=0)

    # Merge data based on the common date index
    merged_data = pd.merge(financial_data, cpi_data, how='inner', left_index=True, right_index=True)

    # Calculate correlations
    correlation_data = merged_data[['Total Revenue/Income', 'Total Operating Expense', 'Income/Profit Before Tax', 'Net Income', 'Inflation']]
    correlation_matrix = correlation_data.corr()

    # Plot and save correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    st.image('correlation_matrix.png')

    # Regression analysis
    features = ['Inflation']
    target_columns = ['Total Revenue/Income', 'Net Income']

    for target_col in target_columns:
        X = merged_data[features]
        y = merged_data[target_col]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate and display Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error for {target_col}: {mse:.2f}')

        # Plot and save regression line
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        plt.title(f'Regression Analysis for {target_col}')
        plt.xlabel('Inflation')
        plt.ylabel(target_col)
        plt.legend()
        plt.savefig(f'regression_{target_col.lower().replace(" ", "_")}.png')
        st.image(f'regression_{target_col.lower().replace(" ", "_")}.png')