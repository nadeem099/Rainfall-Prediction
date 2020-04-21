import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib as pyplot
import requests
import math
import pickle

data = pd.read_csv("NewDataset.csv")

X = data.iloc[:, :22].values
Y = data.iloc[:, 22].values
Y = Y.reshape(-1, 1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Linear Regression Model
Regressor = LinearRegression()
Regressor = Regressor.fit(X_train, Y_train)
Y_predict = Regressor.predict(X_test)

#Plynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
poly_reg.fit(X_poly_train, Y_train)
Regressor1 = LinearRegression()
Regressor1 = Regressor1.fit(X_poly_train, Y_train)
Y_poly_predict = Regressor1.predict(X_poly_test)
train_score =  Regressor1.score(X_poly_train, Y_train)
test_score = Regressor1.score(X_poly_test, Y_test)

#saving model to disk or dumping model in pkl extension
pickle.dump(Regressor1,open('modelPR.pkl', 'wb'))

#Data from weather api
api_address = "https://api.openweathermap.org/data/2.5/weather?appid=8758f651bc9f419b2156cf8d7b566282&q="
city = input("Enter City Name:")
url = api_address+city
json_data = requests.get(url).json() 
L=[]
#temperature 
kelvin_temp = json_data["main"]["temp_min"]
celcisus_temp = kelvin_temp - 273.15
fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
formatted_data1 = math.floor(fahrenheit_temp)
L.append(formatted_data1)
kelvin_temp = json_data["main"]["temp_max"]
celcisus_temp = kelvin_temp - 273.15
fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
formatted_data2 = math.floor(fahrenheit_temp)
L.append(formatted_data2)
kelvin_temp = json_data["main"]["temp"]
celcisus_temp = kelvin_temp - 273.15
fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
formatted_data3 = math.floor(fahrenheit_temp)
L.append(formatted_data3)
#Humidity
L.append(40)
L.append(80)
formatted_data4 = json_data["main"]["humidity"]
L.append(formatted_data4)
#sea level pressure
L.append(990)
L.append(1010)
formatted_data5 = json_data["main"]["pressure"]
L.append(formatted_data5)
#cloud cover
formatted_data6 = json_data["clouds"]["all"]
L.append(formatted_data6)
L.append(formatted_data6)
L.append(formatted_data6)
L.append(formatted_data6)
#sunshine duration
L.append(0)
#shortwave radiation
L.append(0)
#wind speed
L.append(0)
L.append(3)
formatted_data7 = json_data["wind"]["speed"]
L.append(formatted_data7)
#wind direction dominant
L.append(0)
#wind gust
L.append(0)
L.append(0)
L.append(0)
sample = np.transpose([L])
sample= sample.reshape(1, -1)
sample_poly = poly_reg.fit_transform(sample)


#predicting real time precipitation
output = Regressor1.predict(sample_poly)
final_value = output[0][0]
print(final_value)


#evaluation
#for linear regression
from sklearn import metrics
print(metrics.mean_absolute_error(Y_test, Y_predict))
print(metrics.mean_squared_error(Y_test, Y_predict))
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))

#for polynomial regression
from sklearn import metrics
print(metrics.mean_absolute_error(Y_test, Y_poly_predict))
print(metrics.mean_squared_error(Y_test, Y_poly_predict))
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_poly_predict)))




