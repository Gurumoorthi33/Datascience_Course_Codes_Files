

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def load_and_explore_data(filepath):
    """Load and perform initial data exploration"""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nSample Data:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    return df


def preprocess_data(df):
    """Preprocess the dataset"""
    # Create copy to avoid modifying original data
    df_processed = df.copy()

    # Separate features by type
    categorical_features = ['Brand', 'Model']
    numerical_features = ['Year', 'Mileage', 'EngineSize', 'Horsepower',
                          'FuelEfficiency_MPG', 'Accidents', 'MaintenanceScore']

    # Scale numerical features
    scaler = StandardScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = pd.DataFrame(
        encoder.fit_transform(df_processed[categorical_features]),
        columns=encoder.get_feature_names_out(),
        index=df_processed.index
    )

    # Combine processed features
    df_final = pd.concat([
        df_processed.drop(categorical_features + ['Price'], axis=1),
        encoded_cats,
        df_processed['Price']
    ], axis=1)

    return df_final


def create_visualizations(df):
    """Create and display important visualizations"""
    # Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Price by brand boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Brand', y='Price', data=df)
    plt.title('Car Prices by Brand')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numerical_cols = ['Price', 'Year', 'Mileage', 'EngineSize', 'Horsepower',
                      'FuelEfficiency_MPG', 'Accidents', 'MaintenanceScore']
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.show()

    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sns.scatterplot(data=df, x='Mileage', y='Price', hue='Brand', ax=axes[0, 0])
    axes[0, 0].set_title('Price vs Mileage by Brand')

    sns.scatterplot(data=df, x='Year', y='Price', hue='Brand', ax=axes[0, 1])
    axes[0, 1].set_title('Price vs Year by Brand')

    sns.scatterplot(data=df, x='Horsepower', y='Price', hue='Brand', ax=axes[1, 0])
    axes[1, 0].set_title('Price vs Horsepower by Brand')

    sns.scatterplot(data=df, x='MaintenanceScore', y='Price', hue='Brand', ax=axes[1, 1])
    axes[1, 1].set_title('Price vs Maintenance Score by Brand')

    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """Train and evaluate multiple models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        ),
        'Linear Regression': LinearRegression()
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'Cross Val Score': np.mean(cross_val_score(
                model, X, y, cv=5, scoring='r2'
            ))
        }

        # Feature importance for Random Forest
        if name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 10 Most Important Features ({name}):")
            print(feature_importance.head(10))

    return results


def main():
    try:
        # Load and explore data
        print("Loading and exploring data...")
        df = load_and_explore_data('car_prices.csv')

        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df)

        # Preprocess data
        print("\nPreprocessing data...")
        df_processed = preprocess_data(df)

        # Prepare features and target
        X = df_processed.drop('Price', axis=1)
        y = df_processed['Price']

        # Train and evaluate models
        print("\nTraining and evaluating models...")
        results = train_and_evaluate_models(X, y)

        # Print results
        print("\nModel Evaluation Results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure that:")
        print("1. The 'car_prices.csv' file exists in the same directory as this script")
        print("2. All required libraries are installed (pandas, numpy, matplotlib, seaborn, scikit-learn)")
        print("3. You have read permissions for the CSV file")


if __name__ == "__main__":
    main()