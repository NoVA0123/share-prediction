import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklear.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# saving the data from yahoo finance
def stock_data_generator(stock_name):

    data = yf.download(stock_name + '.NS')
    data.to_csv('/data/' + stock_name + '.csv')


def implemetaion(dataset = False, name):

    if dataset:

        df = pd.read_csv('/data/' + name + 'csv')


