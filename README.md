# Stock Prediction model

A prediction model for stock in India.

# Project Overview

This project is about building a predictive model that will take any stock's historical data and will give a predictive model of that stock.

# Tools

1. Google colab to show proper working with visualization.
2. Sci-kit learn library for predictive model.
3. Pandas for data cleaning.
4. Numpy for array creation.
5. yahoo finance api for extracting dataset.
6. Pickle for saving the model.

# Resources

1. [Yahoo Finance API](https://pypi.org/project/yfinance/).
2. [Google Colab](https://colab.research.google.com/).
3. [NSE](https://www.nseindia.com/).

# Data Preparation

In order to make the model we need a dataset of historical stock prices. To get the data, I did following steps:

1. Download Security available for equity segment from this [Equity available for trade](https://www.nseindia.com/market-data/securities-available-for-trading).
2. Use Pandas to extract the symbol column from the dataset.
3. Use the symbol column to extract data from yahoo finance API.
4. Finally save the dataset of companies using Pandas. (optional)

Import the dataset to a variable.(optional)
Note: Use dropna() function to drop any NaN values.

Now we need to remove the 'Close' column because 'Adj Close' column provides the real closing value.

To prepare the dataset for the model we need to perform few task:

1. Create a variable X which has the data without 'Adj Close' column. X is an independent variable.
2. Create a variable Y which has 'Adj Close' column. Y is a dependent variable.
3. Use [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) fucntion and scale the X variable using the StandardScaler().
4. Use [train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to split data into training and testing data.

# Preparing the model

In order to prepare the model we need to perform few tasks:

1. We are going to use [RandomForestRegressor()](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) for predicting the data. It is the best regression model for predicting continuous data.
2. Fit the data.
3. Use Pickle library to extract the model and save it.



