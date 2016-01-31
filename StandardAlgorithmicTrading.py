import math
import pandas as pd
import numpy as np
import NumpyLinRegLearner as lrl
import matplotlib.pyplot as plt
import predictorGenerator as helper
import orderGenerator as orderGenerator
import tradingSimulator as simulator

if __name__=="__main__":

    SYMBOL = 'IBM'
    train_data , price_array, price_df1 = helper.getData([SYMBOL], pd.date_range('2008-01-01', '2009-12-31'))
    test_data  , price_array, price_df2 = helper.getData([SYMBOL], pd.date_range('2010-01-01', '2010-12-31'))

    train_rows = math.floor(train_data.shape[0])
    test_rows = math.floor(test_data.shape[0])

    # separate out training and testing data
    trainX = train_data[:train_rows,0:-1]
    trainY = train_data[:train_rows,-1]

    testX = test_data[:test_rows,0:-1]
    testY = test_data[:test_rows,-1]
    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner

    print "***************  PREDICTION RESULTS :"

    learner.addEvidence(trainX, trainY) # train it
    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])

    #print predY

   # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
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

    # Now Generate Orders Algorithmically based on the predicted values

    #pd.DataFrame(preds.as_data_frame(use_pandas=True)).to_csv("preds.csv")
    #predsDF = pd.read_csv("preds.csv", skiprows=1)

    orders = orderGenerator.generateOrders(price_df1, SYMBOL, predY)
    simulator.simulateMarket('orders.txt' , '2008-01-01' , '2009-12-31')
