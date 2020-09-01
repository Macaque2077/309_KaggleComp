import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import metrics


train_path = "data/electricity_labelled.csv"

test_path = "data/electricity_data_unlabelled.csv"

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# fill missing data with mean for column
data = data.fillna(data.mean())
test_data = test_data.fillna(data.mean())
# df = pd.DataFrame(data=f)
# print(Data.head())

# get x and y from dataframe


train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

y_train = train.iloc[:,-1]
x_train = train.iloc[:,:-1] 

y_test = test.iloc[:,-1]
x_test = test.iloc[:,:-1]

print(train.head())
print(x_train.head())
print("----------------------------")
print(y_train.head())

train_path = 'data/train.tsv'
test_path = 'data/test.tsv'
train.to_csv(train_path, sep='\t', index=False)
test.to_csv(test_path, sep='\t', index=False)

# perform regression
logreg = LogisticRegression()
linreg = LinearRegression()

# x = train[]
linreg.fit(x_train, y_train)
# logreg.fit(x_train,y_train)

predictions = linreg.predict(x_test)
print(test_data.head())
test_predictions = linreg.predict(test_data)

print("on y_test")
print("Mean squared error: ", metrics.mean_squared_error(y_test, predictions))
print("R2 score: ", metrics.r2_score(y_test, predictions))

print("on test data")
print(test_predictions)


import csv 

counter = 1464
testPredictions = []
header = ["Id", "Price"]
testPredictions.append(header)
for pred in test_predictions:
    print(pred)
    line = [counter, pred]
    testPredictions.append(line)
    counter += 1

f = open('test_preds.csv', 'w')

with f:
    writer = csv.writer(f)
    writer.writerows(testPredictions)
