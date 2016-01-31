__author__ = 'kmanda1'

import portifolioEvaluator as pve
import predictorGenerator as helper
import pandas as pd

def simulateMarket(orders_file , start_date , end_date):
    """Driver function."""
    start_val = 10000

    # Process orders
    portvals = pve.compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = pve.get_portfolio_stats(portvals)

    # Simulate a SPY-only reference portfolio to get stats
    prices_SPX = helper.get_adj_closing(['SPY'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]  # remove SPY
    portvals_SPX = pve.get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = pve.get_portfolio_stats(portvals_SPX)

    # Compare portfolio against SPY
    print
    print "Initial Portfolio Value: 10000"
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
    #plot_normalized_data(df_temp, title="Daily portfolio value and SPY")