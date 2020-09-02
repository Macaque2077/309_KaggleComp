import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt 


def preprocessData(path):
    data = pd.read_csv(path)
    data.drop(['Id'], axis =1, inplace = True)
    data = data.fillna(data.median())

    return data

train_path = "data/electricity_labelled.csv"

test_path = "data/electricity_data_unlabelled.csv"

data = preprocessData(train_path)
test_data = preprocessData(test_path)


# get x and y from dataframe --------------------- for kfold cross val
# X = data.iloc[:,:-1]
# y = data.iloc[:,-1]
# # kfold 
# kf = KFold(n_splits=2)
# kf.get_n_splits(X)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns and colname != "Price":
                    del dataset[colname]
                    del test_data[colname]
                    print(colname)
    # for col in col_corr:
    #     if col != "Price":
            
    #     print(col)
    # print(dataset)

# to removae values with correlation 
correlation(data, 0.99)

# normal train test split 
train, test = train_test_split(data, test_size=0.01, random_state=0, shuffle=True)

y_train = train.iloc[:,-1]
x_train = train.iloc[:,:-1] 

y_test = test.iloc[:,-1]
x_test = test.iloc[:,:-1]

# train_path = 'data/train.tsv'
# test_path = 'data/test.tsv'
# train.to_csv(train_path, sep='\t', index=False)
# test.to_csv(test_path, sep='\t', index=False)

# perform regression
# linreg = LinearRegression()
rf = RandomForestRegressor(n_estimators= 1000, max_depth=200, random_state=0)
rf.fit(x_train, y_train)
# x = train[]
# linreg.fit(x_train, y_train)

# logreg.fit(x_train,y_train)

predictions = rf.predict(x_test)

test_predictions = rf.predict(test_data)

print("on y_test")
print("Mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print("R2 score: ", metrics.r2_score(y_test, predictions))

print("on test data")
print(test_predictions)


import csv 

counter = 1464
testPredictions = []
header = ["Id", "Price"]
testPredictions.append(header)
for pred in test_predictions:
    # print(pred)
    line = [counter, abs(pred)]
    testPredictions.append(line)
    counter += 1

f = open('test_preds.csv', 'w')

with f:
    writer = csv.writer(f)
    writer.writerows(testPredictions)
