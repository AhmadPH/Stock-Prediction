import pandas_datareader.data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd

yf.pdr_override()

def get_stock_data(ticker,start,end):
    stock_data = pdr.get_data_yahoo(ticker, start, end)
    stock_data.to_csv("./dataset/stock_data.csv")
    # stock_data['Close'].plot()
    # plt.title('shoupan')
    # plt.show()


if __name__ == "__main__":
    get_stock_data("AAPL", "2000-01-01", "2019-10-01")
