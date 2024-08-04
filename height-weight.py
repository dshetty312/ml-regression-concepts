import pandas as pd
df = pd.read_csv("height-weight-data.csv")
print(df.head())

# EDA
# Scatterplot
import seaborn as sns
import numpy as np


sns.lmplot(x="Height-Inches", y="Weight-Pounds", data=df)

#Descriptive Statistics
print(df.describe())

#Kernel Density Distribution
sns.jointplot(x="Height-Inches", y="Weight-Pounds", data=df, kind="kde");

# Modeling
# Sklearn Regression Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


y = df['Weight-Pounds'].values  # Target
y = y.reshape(-1, 1)
X = df['Height-Inches'].values  # Feature(s)
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_predicted = lm.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

#RMSE Root Mean Squared Error
rms = sqrt(mean_squared_error(y_predicted, y_test))
print(f'Model rms: {rms}')



