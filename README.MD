# Simple Linear Regression Model

This repository contains a Python implementation of a custom **Linear Regression** model that predicts a target variable based on input features, such as engine size and fuel consumption. The model computes the weights and intercept manually, without using external machine learning libraries like `scikit-learn`.

## Features
- **Custom Linear Regression**: Implements the linear regression algorithm from scratch using numpy.
- **Flexible Prediction**: Can predict any target variable based on the provided input features.
- **No External Libraries**: The model doesn't rely on any external machine learning libraries, offering insights into the mathematical workings behind linear regression.

## How It Works
1. **Data Loading**: The dataset (e.g., `FuelConsumptionCo2.csv`) is loaded, and the relevant features (e.g., engine size, fuel consumption) are extracted.
2. **Training**: The model computes the weights for each feature using the covariance and variance formulas. The intercept is calculated as well.
3. **Prediction**: The model uses the computed weights and intercept to predict the target variable.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SimpleLinearRegression.git
