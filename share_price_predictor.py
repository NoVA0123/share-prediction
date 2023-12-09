import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklear.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle


# saving the data from yahoo finance
def stock_data_generator(stock_name):

    data = yf.download(stock_name + '.NS')
    data.to_csv('/data/' + stock_name + '.csv')
    return data


'''
Two ways we can achieve importing our data:
1. Either by importing the data
2. By downloading the data
'''


def implementation(data, name, dataset=False):

    if dataset:

        df = pd.read_csv('/data/' + name + 'csv')

    else:

        df = yf.download(name + '.NS')

    # dropping any NaN values and deleting 'Close' column
    df.dropna(inplace=True)
    data = pd.DataFrame()
    data = df.drop('Close', axis=1)

    '''Creating X variable for independent values
       and y variable for dependent values'''
    X = data.drop('Adj Close', axis=1).to_numpy()
    y = data['Adj Close'].to_numpy()

    # Creating a scaler to scale the values of X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    # making a model and fitting the data
    rfr = RandomForestRegressor(n_estimators=100,
                                random_state=3,
                                max_samples=0.5,
                                max_features=0.75,
                                max_depth=15)

    rfr.fit(x_train, y_train)

    # Creating a file that stores the model
    pickle.dump(rfr, open(f'/models/{name}.pkl', 'wb'))


# importing the names of the companies
equity_data = pd.read_csv('EQUITY_L.csv')
equity_symbols = equity_data['SYMBOL'].tolist()

# downloading the data
for x in equity_symbols:

    data = stock_data_generator(x)
    
# Creating the models
    if len(data) >= 63:
        implementation(x)
