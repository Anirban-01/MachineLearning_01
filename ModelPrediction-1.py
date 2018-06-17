import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

 # Declare the columns names
columns = 'age sex bmi map tc ldl hdl tch ltg glu'.split()
# call diabetes datasets from sklearn
diabetes = datasets.load_diabetes()
#load the datasets
datasets = pd.DataFrame(diabetes.data, columns=columns)
#loading target variable
y=diabetes.target

#creating train_set and test_set
x_train, x_test, y_train, y_test = train_test_split(datasets, y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#fit the model
classifier=linear_model.LinearRegression()
model = classifier.fit(x_train, y_train)
#predicting
predict = classifier.predict(x_test)
#accuracy score
print('score', model.score(x_test, y_test))

