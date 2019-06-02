import pandas_datareader.data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib.pylab import date2num
import datetime
import pandas as pd
plt.rcParams["figure.figsize"] = [12.8, 9.6]
yf.pdr_override()

def get_stock_data(ticker,start,end):
    stock_data = pdr.get_data_yahoo(ticker, start, end)
    stock_data.to_csv("./dataset/stock_data.csv")


if __name__ == "__main__":
    get_stock_data("AAPL", "2000-01-01", "2019-10-01")
    stock = pd.read_csv("./dataset/stock_data.csv")
    quotes = []

    for row in range(50):
        if row == 0:
            sdate = str(stock.loc[row,'Date'])
            sdate_change_format = sdate
            sdate_num = date2num(datetime.datetime.strptime(sdate_change_format,'%Y-%m-%d'))
            sdate_plt = sdate_num
        else:
            sdate_plt = sdate_num + row
        
        sopen = stock.loc[row,'Open']
        shigh = stock.loc[row,'High']
        slow = stock.loc[row,'Low']
        sclose = stock.loc[row,'Close']
        datas = (sdate_plt,sopen,shigh,slow,sclose)
        quotes.append(datas)

    fig, ax = plt.subplots(facecolor=(1, 1, 1),figsize=(12,8))
    fig.subplots_adjust(bottom=0.1)
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.title('AAPI stock')
    plt.xlabel('time')
    plt.ylabel('price')
    candlestick_ohlc(ax,quotes,width=0.7,colorup='r',colordown='green') 
    plt.grid(True)
    plt.show()
