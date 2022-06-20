
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

veriler = pd.read_csv('maaslar_yeni.csv')

data = veriler.iloc[:,2:5].values
maas = veriler.iloc[:,-1:].values

#backwards elemination
'''
lin_reg = LinearRegression()
lin_reg.fit (data,maas)
import statsmodels.api as sm
r_ols = sm.OLS(lin_reg.predict(data),data)
r = r_ols.fit()
print(r.summary())
'''
#x2 and x3 has high p value so we eleminate it
data = veriler.iloc[:,[2]].values

#polynominal regression
poly_reg = PolynomialFeatures (degree=4)
d_poly = poly_reg.fit_transform (data)
lin_reg = LinearRegression()
lin_reg.fit(d_poly,maas)

print('poly reg pred')
print(lin_reg.predict(poly_reg.fit_transform([[10]])))

#SVR
sc1 = StandardScaler()
xsc = sc1.fit_transform(data)
sc2 = StandardScaler()
ysc = sc2.fit_transform(maas)

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(xsc,ysc)

print('svr pred')
print('rf pred')
sc3 = StandardScaler()
pred1 = svr_reg.predict([[10]])
sc3.fit([[10]])
pred1 = sc3.inverse_transform([[10]])
print(pred1)

#DT

#RF
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)

sc1 = StandardScaler()
xsc = sc1.fit_transform(data)
sc2 = StandardScaler()
ysc = sc2.fit_transform(maas)

rf_reg = SVR(kernel = 'rbf')
rf_reg.fit(xsc,ysc)

print('rf pred')
sc3 = StandardScaler()
pred1 = rf_reg.predict([[10]])
sc3.fit([[10]])
pred1 = sc3.inverse_transform([[10]])
print(pred1)
