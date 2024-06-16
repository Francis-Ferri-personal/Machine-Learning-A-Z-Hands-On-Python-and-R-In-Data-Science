# Multiple Linear Regresion

# Importing the Ubraries
import numpy as np
import matplotlib.pyplot as pit
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independant Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
# Apply OneHotEncoder using ColumnTransformer
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [3])
    ],
    remainder='passthrough'
)

x = column_transformer.fit_transform(x)

# avoiding the dummy variable trap 
x = x[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#%% Fitting Multiple Linerar Regression to the training set
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train, y_train)

# %%
# Prediction of the test set results
y_pred = regresor.predict(x_test)

# %%
# Build optimal moedl using backwars elimination
import statsmodels.formula.api as sm
from statsmodels.regression.linear_model import OLS

# We need to add one column of ones for the b0*x0 value
x = np.append(arr=np.ones((x.shape[0], 1)).astype(int), values=x, axis=1)

# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()

# P>|t|
# x2 = 0.990 (we have to remove it)

# %%
from statsmodels.regression.linear_model import OLS
# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 1, 3, 4, 5]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()
# %%
from statsmodels.regression.linear_model import OLS
# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 1, 3, 4, 5]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()

# %%
from statsmodels.regression.linear_model import OLS
# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 3, 4, 5]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()

# %%
from statsmodels.regression.linear_model import OLS
# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 3, 5]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()

# %%
from statsmodels.regression.linear_model import OLS
# x opt are the optimal variables. Variables that have only high impact
x_opt = x[:, [0, 3]]
x_opt = x_opt.astype(float)
 
regresor_ols = OLS(endog=y, exog=x_opt).fit()

regresor_ols.summary()