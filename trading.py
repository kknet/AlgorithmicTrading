"""
Test a learner.  (c) 2023 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn
import pandas as pd
import os
import matplotlib.pyplot as plt

def symbol_to_path(symbol, base_dir=os.path.join(".", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_adj_closing(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""

    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df = df.dropna()
    return df

def Volatility(prices):
    return pd.rolling_std(prices, window=20)

def Momentum(prices):
    momentum = (prices/prices.shift(-20))-1
    return momentum

def BollingerBand(prices):
    sma = pd.rolling_mean(prices, window=20)
    std = pd.rolling_std(prices, window=20)
    df = (prices - sma)/(2*std)
    return df

def Y(prices):
    return (prices.shift(-5)/prices) - 1.0

def getData(symbols, dates):
    price_df = get_adj_closing(symbols, dates)
    Y_df = Y(price_df).fillna(method='ffill')
    BB_df = BollingerBand(price_df).fillna(0)
    V_df = Volatility(price_df).fillna(0)
    M_df = Momentum(price_df).fillna(0)
    total_rows = len(Y_df)
    data = np.zeros(shape=(total_rows,4))
    price_array = np.zeros(shape=(total_rows,1))
    #prices = price_df.fillna(method='ffill')

    for i in xrange(total_rows):
       data[i,0]=BB_df.ix[i]
       data[i,1]=V_df.ix[i]
       data[i,2]=M_df.ix[i]
       data[i,3]=Y_df.ix[i]

       price_array[i,0] = price_df.ix[i]

    return data , price_array , price_df


def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    File contains - list orders
    Load historical data
    Execute the orders in the past
    At any instant we shall have a portfolio

    earlier we pretended - buying stocks in the beginning and holding them forever
    now continuously buying and selling .. later on it will help to 'formulate trading strategy using a technical indicator'
the straegy will then generate orders and be executed thru this Market Simulator ...

now back to current assignments ....


    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """

    # Reference : http://quantsoftware.gatech.edu/images/a/a2/Marketsim-guidelines.pdf

    # Step 1 : read the dates and symbols from order file and sort the data by date

    orders_df = pd.read_csv(orders_file,header=0, index_col=["Date"])
    #orders_df = pd.DataFrame.sort(orders_df,ascending=[1, 0])

    #print orders_df

    order_date_range = orders_df.index.tolist()
    symbols = list(set(orders_df["Symbol"]))

    # Step 2 : read actual values

    # order_date_range[0] , order_date_range[-1]

    prices_df = get_adj_closing(symbols, pd.date_range(start_date, end_date ))
    #print " ************ "
    #print "Orders : "
    #print orders_df
    #print order_date_range
    #print "Share Prices : "
    #print prices_df
    #print " ************ "

    # Step 3 : create the matrix of shares
    # Create a dataframe which has all values as zero with index as dates and columns as symbols.
    # Trade Matrix
    '''
    Date AAPL MSFT
    12/1 0 0
    '''
    #print "************"

    trading_df = pd.DataFrame(0, index=prices_df.index, columns=symbols)

    # Step 4 : calculate cash timeseriese and trading timeseriese
    '''
    For each order subtract the cash used in that trade.
    - Selling actually gives you cash.

    Date Cash
    12/1 10000
    12/2 -500
    '''
    cash_ts = pd.DataFrame(0, index=prices_df.index, columns=['Cash'])

    for date_index, row in orders_df.iterrows():

        #print "     >>  current order " , row["Shares"] , row["Symbol"] , date_index

        if row["Order"] == 'BUY':
            trading_df.ix[date_index][row["Symbol"]] += row["Shares"]
            cash_ts.ix[date_index] -= row["Shares"]*prices_df.ix[date_index][row["Symbol"]]

        elif row["Order"] == 'SELL':
            trading_df.ix[date_index][row["Symbol"]] -= row["Shares"]
            cash_ts.ix[date_index] += row["Shares"]*prices_df.ix[date_index][row["Symbol"]]

    print " ####### "
    #print trading_df
    print " ####### "

    # Step 5 : calculate funds timeseriese
    cash_ts.ix[0] += start_val
    funds_ts = cash_ts.cumsum()

    # Step 6 :
    '''
    Use cummulative sum to convert the trade matrix into holding matrix.
    so that the holdings reflect the correct ammount of daily shares and funds show

    Example :

    PRICE >>
    Date\Sym AAPL MSFT
    12/1     400.0 30.0

    Holdings >>
    Date\Shares AAPL MSFT
    12/1        50.0 200.0

    Funds >>
    Date\Cash
    12/1       1000.0

    Date Value
    12/1 27,000.0  (400*50+30*200+1000)
    '''

    holdings_df = trading_df.cumsum()

    '''
    Now we have both the price and holding matrix.
    Use dot product to calculate value of portfolio on each date.
    '''

    port_value = pd.DataFrame(0, columns=['Value'], index=prices_df.index)
    port_value['Value'] = (prices_df * holdings_df).sum(axis=1)
    port_value['Value'] = (funds_ts['Cash']+port_value['Value'])

   # print " ************ "
    #print "Daily value of portfolio : "
    #print port_value
    #print " ************ "

    return port_value


def get_portfolio_value(prices, allocs, start_val=1):
    """Compute daily portfolio value given stock prices, allocations and starting value.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio
        allocs: initial allocations, as fractions that sum to 1
        start_val: total starting value invested in portfolio (default: 1)

    Returns
    -------
        port_val: daily portfolio value
    """
    normed_vals = prices / prices.ix[0]
    allocated_vals = normed_vals*allocs
    pos_val = allocated_vals*start_val
    port_val = pos_val.sum(axis=1)

    return port_val

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    """Calculate statistics on given portfolio values.

    Parameters
    ----------
        port_val: daily portfolio value
        daily_rf: daily risk-free rate of return (default: 0%)
        samples_per_year: frequency of sampling (default: 252 trading days)

    Returns
    -------
        cum_ret: cumulative return
        avg_daily_ret: average of daily returns
        std_daily_ret: standard deviation of daily returns
        sharpe_ratio: annualized Sharpe ratio
    """
    k = np.sqrt(samples_per_year)
    daily_returns = (port_val / port_val.shift(1))-1

    #print daily_returns

    cum_ret = (port_val[-1] /  port_val[0]) - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret =  daily_returns.std()
    daily_returns[0] = 0
    sharpe_ratio = k * (np.mean(daily_returns - daily_rf) / np.std(daily_returns))

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio


def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.

    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    df = df / df.ix[0]
    df.plot(figsize=(8, 5))
    plt.show()



def generateOrders(price_df, symbol, predY):

    #ax = plt.plot(price_array)
    ax = price_df[symbol].plot(title="Trading Strategies", label=symbol)

    orders = pd.DataFrame(index=np.arange(price_df.size),columns=['Date','Symbol','Order','Shares'])

    '''
    long_entries = np.zeros(shape=(price_array.size,1))
    short_entries = np.zeros(shape=(price_array.size,1))
    long_exits = np.zeros(shape=(price_array.size,1))
    short_exits = np.zeros(shape=(price_array.size,1))
    '''
    long_entries = pd.DataFrame(index=price_df.index, columns=[symbol])
    short_entries = pd.DataFrame(index=price_df.index, columns=[symbol])

    long_exits = pd.DataFrame(index=price_df.index, columns=[symbol])
    short_exits = pd.DataFrame(index=price_df.index, columns=[symbol])

    last_position = 'NA'
    i = 0 # range(0, df.shape[0])
    j = -1
    s = -1
    t = 0
    long_entry_band = 5
    short_entry_band = 5
    N = len(price_df)

    normedPredY = predY*0.01
    price_df = price_df*0.01

    diff = (price_df[symbol] - normedPredY)
    avgDiff = diff.mean()
    print avgDiff


    for date_index , row  in price_df.iterrows():

        index_date=date_index.date()

        if(i<N):

            if (abs(price_df.irow(i)[symbol] - normedPredY[i]) < avgDiff and  long_entry_band > 0):
                short_entry_band = 5
                long_entry_band = long_entry_band - 1
                #long_entries[i,0] = price_array[s,0]
                long_entries.irow(i)[symbol] = price_df.irow(i)[symbol]
                last_position = 'LONG_ENTRY'
                j = j + 1
                orders.irow(j)['Date']=index_date
                orders.irow(j)['Symbol']=symbol
                orders.irow(j)['Order']='BUY'
                orders.irow(j)['Shares']=100

                plt.vlines(x = index_date, colors = 'g', ymin= 0, ymax= 140)

            elif (abs(normedPredY[i] - price_df.irow(i)[symbol]) > avgDiff and short_entry_band > 0):
                long_entry_band = 5
                short_entry_band = short_entry_band - 1
                #short_entries[i,0] = price_array[s,0]
                short_entries.irow(i)[symbol] = price_df.irow(i)[symbol]

                last_position = 'SHORT_ENTRY'
                j = j + 1
                orders.irow(j)['Date']=index_date
                orders.irow(j)['Symbol']=symbol
                orders.irow(j)['Order']='SELL'
                orders.irow(j)['Shares']=100

                plt.vlines(x = index_date, colors = 'r', ymin= 0, ymax= 140)

            elif (last_position == 'LONG_ENTRY' and long_entry_band < 1) :
                long_entry_band = 5
                #long_exits[i,0] = price_array[s,0]
                long_exits.irow(i)[symbol] = price_df.irow(i)[symbol]

                last_position = 'LONG_EXIT'

                j = j + 1
                orders.irow(j)['Date']=index_date
                orders.irow(j)['Symbol']=symbol
                orders.irow(j)['Order']='SELL'
                orders.irow(j)['Shares']=100

                plt.vlines(x = index_date, colors = 'k', ymin= 0, ymax= 140)

            elif (last_position == 'SHORT_ENTRY' and short_entry_band < 1) :
                #short_exits[i,0] = price_array[s,0]
                short_entry_band = 5
                short_exits.irow(i)[symbol] = price_df.irow(i)[symbol]

                last_position = 'SHORT_EXIT'

                j = j + 1
                orders.irow(j)['Date']=index_date
                orders.irow(j)['Symbol']=symbol
                orders.irow(j)['Order']='BUY'
                orders.irow(j)['Shares']=100

                plt.vlines(x = index_date, colors = 'k', ymin= 0, ymax= 140)

        i = i + 1
        t = t + 1

    print '--------------------'
    orders = orders.dropna(subset=["Symbol"]).sort_index()
    #print 'save data into orders file.'
    orders.to_csv('orders.txt')
    #print 'Orders' , orders

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower center')
    plt.show()

    return orders

def simulateMarket(orders_file , start_date , end_date):
    """Driver function."""
    start_val = 10000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a SPY-only reference portfolio to get stats
    prices_SPX = get_adj_closing(['SPY'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against SPY
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, portvals_SPX], keys=['Portfolio', 'SPY'], axis=1)
    #print "**********" , df_temp
    plot_normalized_data(df_temp, title="Daily portfolio value and SPY")


if __name__=="__main__":

    #SYMBOL = 'ML4T-399'
    SYMBOL = 'IBM'
    train_data , price_array, price_df1 = getData([SYMBOL], pd.date_range('2008-01-01', '2009-12-31'))
    test_data  , price_array, price_df2 = getData([SYMBOL], pd.date_range('2010-01-01', '2010-12-31'))

    # compute how much of the data is training and testing
    train_rows = math.floor(train_data.shape[0])
    test_rows = math.floor(test_data.shape[0])

    # separate out training and testing data
    trainX = train_data[:train_rows,0:-1]
    trainY = train_data[:train_rows,-1]

    testX = test_data[:test_rows,0:-1]
    testY = test_data[:test_rows,-1]

    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner
    #learner = knn.KNNLearner(k = 3) # constructor
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])

    price_array = np.zeros(shape=(len(price_df1),1))
    i=0
    for ind in price_df1.index:
        price_array[i] =  price_df1.ix[i]*0.0001
        i = i+1
    plt.plot(price_array)
    plt.plot(trainY)
    plt.plot(predY)
    plt.legend(('originalPrice','trainY', 'predY'), loc='upper right', shadow=True)
    plt.show()

    orders = generateOrders(price_df1, SYMBOL,  predY)
    #print predY
    #mkt.run('orders.txt' , '2008-01-01' , '2009-12-31')
    simulateMarket('orders.txt' , '2008-01-01' , '2009-12-31')

    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    plt.plot(price_array)
    plt.plot(testY)
    plt.plot(predY)
    plt.legend(('originalPrice','trainY', 'predY'), loc='upper right', shadow=True)
    plt.show()
    orders = generateOrders(price_df2, SYMBOL,  predY)
    #print predY
    #mkt.run('orders.txt' , '2008-01-01' , '2009-12-31')
    simulateMarket('orders.txt' , '2010-01-01', '2010-12-31')
