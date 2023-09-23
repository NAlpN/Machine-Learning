import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("veriler.csv")

print(data)

x = data.iloc[:,1:4]
y = data.iloc[:,4:]
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)


y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)