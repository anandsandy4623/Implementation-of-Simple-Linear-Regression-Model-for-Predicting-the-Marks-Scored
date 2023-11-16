# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard Libraries.

2. Set variables for assigning dataset values.
   
3. Import linear regression from sklearn.
 
4. Assign the points for representing in the graph.
 
5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
...
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ananthavarthan S
RegisterNumber:  212221043001
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="yellow")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
...
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
# df.head()
![Screenshot 2023-08-24 141423](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/26905b7a-453e-4e02-b1ec-16b149b5b3c5)
# df.tail()
![Screenshot 2023-08-24 141815](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/39c302f6-2926-4dd5-a6e3-95ff255a97d4)

# Array value of X
![Screenshot 2023-08-24 141826](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/88e7978e-23a1-4e0f-9ef4-da9250e6b3ed)

# Array value of Y
![Screenshot 2023-08-24 141837](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/5f72cf98-ceaf-43ec-9482-d9b036cd6a2e)

# Values of Y prediction
![Screenshot 2023-08-24 141847](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/0d41e29e-9f78-4490-857e-9c912d2c62bf)

# Array values of Y test
![Screenshot 2023-08-24 141854](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/8706b56e-f276-41f2-96e2-bd2d98274e5f)

# Training set graph
![Screenshot 2023-08-24 141903](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/c277e823-cb79-4bca-ac0e-00963d8f0795)

# Test set graph
![Screenshot 2023-08-24 162012](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/96bfd221-3ac2-4197-a254-178f581c241a)

# Values of MSE,MAE and RMSE
![Screenshot 2023-08-24 141920](https://github.com/chandramohan3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142579775/56c42b36-03cb-4fc6-8a25-4c84fbee14d3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
