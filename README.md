# Algorithmic Trading
Select a supervised algorithm that can predict stock prices of historical data based on the predictors 
i.e. statistical indicators - like Volatility, Bollinger Bands etc.
> H2OAlgorithmicTrading.py

Accordingly formulate a trading strategy based on predicted values to generate orders on same historical training set to backtest 
how much portfolio would have increased.
> tradingSimulator.py , orderGeneraor.py

Select the combination of Machine learning algorithm and Trading strategy to maximize gain for future orders,
placed automatically via the program.

### Performance of H2OGeneralizedLinearEstimator(family = "gaussian")

//////////////////////////
Dataset:
Symbol : IBM
Training Data: pd.date_range('2008-01-01', '2009-12-31')
Testing Data: pd.date_range('2010-01-01', '2010-12-31')

MSE: 0.0011164438007
R^2: -0.0556747669549
Mean Residual Deviance: 0.0011164438007
Null degrees of freedom: 251
Residual degrees of freedom: 249
Null deviance: 0.269232404456
Residual deviance: 0.281343837775
AIC: -989.851898711

////////////////////////
Initial Portfolio Value: 100000
Data Range: 2008-01-01 to 2009-12-31

Sharpe Ratio of Fund: -0.868324247687
Sharpe Ratio of SPY: -0.137143768005

Cumulative Return of Fund: 32.3673
Cumulative Return of SPY: -0.194324631101

Standard Deviation of Fund: 1.98264249049
Standard Deviation of SPY: 0.0219321223021

Average Daily Return of Fund: -0.108449237284
Average Daily Return of SPY: -0.000189476626317

Final Portfolio Value: 333673.0

### Performance of numpy standard Linear Regression Classifier - np.linalg.lstsq(dataX, dataY)

The main observation is variation of predictions from actual value considerably high and as a result the particular trading strategy does not work for this standard algorithm very well.

//////////////////////////////
In sample results
RMSE:  0.0371178709308
corr:  0.499644155508

Out of sample results
RMSE:  0.0199253542973
corr:  0.488509112306

/////////////////////////////
Initial Portfolio Value: 100000
Data Range: 2008-01-01 to 2009-12-31

Sharpe Ratio of Fund: -0.311980176564
Sharpe Ratio of SPY: -0.137143768005

Cumulative Return of Fund: -38.1258
Cumulative Return of SPY: -0.194324631101

Standard Deviation of Fund: 1.12516858177
Standard Deviation of SPY: 0.0219321223021

Average Daily Return of Fund: -0.0221127948848
Average Daily Return of SPY: -0.000189476626317

Final Portfolio Value: -371258.0
