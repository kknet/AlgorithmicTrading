# Algorithmic Trading
### Goal:
Select a supervised algorithm that can predict stock prices of historical data based on the predictors 
i.e. statistical indicators - like Volatility, Bollinger Bands etc.
> H2OAlgorithmicTrading.py

Accordingly formulate a trading strategy based on predicted values to generate orders on same historical training set to backtest 
how much portfolio would have increased.
> tradingSimulator.py , orderGeneraor.py

Select the combination of Machine learning algorithm and Trading strategy to maximize gain for future orders,
placed automatically via the program.

### Run the Application:

python H2OAlgorithmicTrading.py
python StandardAlgorithmicTrading.py

##### Assumptions:
1) h2o python modules installed (see Referenced at the bottom of the page)

2) write permission on /tmp/ folder and write permission on current working folder 

3) faced problems downloading files from H2O server and converting pd frames from/into H2O frames

4) will work only in python 2.7

5) if one wants to point to local / remote h2O server, then comment out h2O.init() and uncomment h2O.init(ip=..,port=...)

#### Performance of H2OGeneralizedLinearEstimator(family = "gaussian")

##### Algorithm Metrics:
Dataset:
Link : https://drive.google.com/file/d/0ByhSuUifwO07bms4NmFNNjVwbTQ/view
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

##### Trading Results:
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

#### Performance of standard numpy Linear Regression Classifier - np.linalg.lstsq(dataX, dataY)

The main observation is variation of predictions from actual value considerably high and as a result the particular trading strategy does not work for this standard algorithm very well.

Strong prediction automatically bossts up the Trading as the decision to but / sell is based on a threshold on the difference between actual and predicted values. Definitely we can make this strtagey more robust.  

##### Algorithm Metrics:
In sample results
RMSE:  0.0371178709308
corr:  0.499644155508

Out of sample results
RMSE:  0.0199253542973
corr:  0.488509112306

##### Trading Results:
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

### Issues:
1. we shouldn't save intermediate results into files, rather should upload into h2o server and download from it
2. 

### Scope of Improvement:
1. use cross-validations / bagging along with Lin Regression and check if performance further improves.

### References:

H2O Quick Setup and Demo - https://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/8/docs-website/h2o-docs/booklets/Python_booklet.pdf

Generalized Linear Modelling with H2O - http://h2o-release.s3.amazonaws.com/h2o/rel-slater/8/docs-website/h2o-docs/booklets/GLM_Vignette.pdf

Useful COde reference - http://h2o-release.s3.amazonaws.com/h2o/master/3340/docs-website/h2o-py/docs/intro.html

Demos : https://github.com/h2oai/h2o-3/tree/master/h2o-py/demos

Algo Syntax and Semantics - http://docs.h2o.ai/h2oclassic/datascience/glm.html

API docs - http://h2o-release.s3.amazonaws.com/h2o/master/3065/docs-website/h2o-py/docs/frame.html
