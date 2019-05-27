import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

dataset = pd.read_csv('./test/stock_data.csv',index_col='Date',parse_dates=True)

print(dataset.head())