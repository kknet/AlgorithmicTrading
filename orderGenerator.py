__author__ = 'kmanda1'

import pandas as pd
import numpy as np

def generateOrders(price_df, symbol, predY):

    print "******** Generate Orders based on Normalized Predictions:"
    print "******** start"

    #ax = plt.plot(price_array)
    ax = price_df[symbol].plot(title="Trading Strategies", label=symbol)

    orders = pd.DataFrame(index=np.arange(price_df.size),columns=['Date','Symbol','Order','Shares'])

    print "*** 1"

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

    print "*** 2"

    last_position = 'NA'
    i = 0 # range(0, df.shape[0])
    j = -1
    s = -1
    t = 0
    long_entry_band = 5
    short_entry_band = 5

    normedPredY = predY*100
    N = len(normedPredY)



   # print "*** 3 -> price_df", price_df.size

    price_df = price_df*0.01

   # print "*** 4" , normedPredY.size

    for date_index , row  in price_df.iterrows():

        index_date=date_index.date()

        if(i<N):

            if(isinstance(normedPredY, pd.DataFrame)):
                currentDiff = price_df.irow(i)[symbol] - normedPredY.irow(i)[1]
            else:
                currentDiff = price_df.irow(i)[symbol] - normedPredY[1]

            #print currentDiff

            if (currentDiff < 1.5 and  long_entry_band > 0):
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

                #plt.vlines(x = index_date, colors = 'g', ymin= 0, ymax= 140)

            elif (currentDiff > 0.2 and short_entry_band > 0):
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

                #plt.vlines(x = index_date, colors = 'r', ymin= 0, ymax= 140)

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

                #plt.vlines(x = index_date, colors = 'k', ymin= 0, ymax= 140)

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

                #plt.vlines(x = index_date, colors = 'k', ymin= 0, ymax= 140)

        i = i + 1
        t = t + 1

    print '--------------------'
    orders = orders.dropna(subset=["Symbol"]).sort_index()
    print 'save data into orders file.'
    orders.to_csv('orders.txt')
    print 'Orders' , orders

    # Add axis labels and legend
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='lower center')
    #plt.show()

    return orders
