import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
import h2o
import tabulate
import operator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import predictorGenerator as helper
import orderGenerator as orderGenerator
import tradingSimulator as simulator

if __name__=="__main__":

    #h2OServer = h2o.init(ip="10.0.0.218",port="54321")
    h2o.init()
    #SYMBOL = 'IBM'
    SYMBOL = 'APPLE'
    train_data , price_array, price_df1 = helper.getData([SYMBOL], pd.date_range('2008-01-01', '2009-12-31'))
    test_data  , price_array, price_df2 = helper.getData([SYMBOL], pd.date_range('2010-01-01', '2010-12-31'))

    tndf = pd.DataFrame(train_data);
    tsdf = pd.DataFrame(test_data)

    tndf.to_csv('/tmp/training.csv')
    tsdf.to_csv('/tmp/testing.csv')

    #h2o.upload_file("training.csv")
    #h2o.upload_file("testing.csv")

    print "**********"

    h20_train = h2o.import_file("/tmp/training.csv")
    h20_test = h2o.import_file("/tmp/testing.csv")

    print "**********"
    train_rows = math.floor(train_data.shape[0])
    test_rows = math.floor(test_data.shape[0])

    # separate out training and testing data
    trainX = train_data[:train_rows,0:-1]
    trainY = train_data[:train_rows,-1]


    learner = H2OGeneralizedLinearEstimator(family = "gaussian")
    learner.train(x=list(range(0,2)), y=3, training_frame=h20_train)


    preds = learner.predict(h20_test)
    print "***************  PREDICTION RESULTS :"

    pd.DataFrame(preds.as_data_frame(use_pandas=True)).to_csv("preds.csv")
    predsDF = pd.read_csv("preds.csv", skiprows=1)

    print "***********" ,learner.model_performance(h20_test)

    '''
    price_array = np.zeros(shape=(len(price_df1),1))
    i=0
    for ind in price_df1.index:
        price_array[i] =  price_df1.ix[i]*0.0001
        i = i+1
    plt.plot(price_array)
    plt.plot(trainY)
    plt.plot(preds)
    plt.legend(('originalPrice','trainY', 'predY'), loc='upper right', shadow=True)
    plt.show()
    '''

    # Now Generate Orders Algorithmically based on the predicted values
    orders = orderGenerator.generateOrders(price_df1, SYMBOL,  predsDF)
    simulator.simulateMarket('orders.txt' , '2008-01-01' , '2009-12-31')
