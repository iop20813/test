from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
# TODO
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data, columns=['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']) #有13個feature
# TODO
# MEDV即預測目標向量
# TODO
x = df[['CRIM','ZN','INDUS','CHAS','NOX','RM' ,'AGE','DIS','RAD','TAX', 'PTRATIO','B','LSTAT']]
target=pd.DataFrame(boston.target,columns=['target'])
y=target['target']

#分出20%的資料作為test set
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=1)
# TODO
le=linear_model.LinearRegression()
le.fit(xtrain,ytrain)
#Fit linear model 配適線性模型
pred=le.predict(xtest)
mae=mean_absolute_error(pred,ytest)
mse=mean_squared_error(pred,ytest)

# TODO
print('MAE:' ,mae             )
print('MSE:' ,mse             )
print('RMSE:'   ,mse**0.5          )

X_new=([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]])
prediction = le.predict(X_new)
print( prediction)
