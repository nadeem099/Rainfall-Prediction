#Linear Regression
#importing libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

#importing dataset
data = pd.read_csv("NewDataset.csv")

#separating X and Y
X = data.drop(['Precipitation Total'], axis=1)
Y = data['Precipitation Total']
Y = Y.values.reshape(-1, 1)

#splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.2, random_state = 0)

#list of days
day_index = 798
days = [i for i in range(Y.size)]

#training model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#predicting test data
Y_pred = regressor.predict(X_test)

#saving model to disk or dumping model in pkl extension
pickle.dump(regressor,open('modelLR.pkl', 'wb'))

#Data from weather api
import requests
import math
api_address = "https://api.openweathermap.org/data/2.5/weather?appid=8758f651bc9f419b2156cf8d7b566282&q="
city = input("Enter city Name:")
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
sample2 = np.transpose([L])
sample2= sample2.reshape(1, -1)

#predicting real time precipitation
output = regressor.predict(sample2)
final_value = output[0][0]
print(final_value)

#plotting the graph
plt.scatter(days, Y)
plt.scatter(days[day_index], Y[day_index], color='r')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")

plt.show()
x_vis = X.filter(['Temperature Avg', 'Relative Humidity Avg', 'Mean Sea Level Pressure', 'Cloud Cover Total', 'Wind Speed Avg', 'Wind Gust Avg'], axis=1)

for i in range(x_vis.columns.size):
	plt.subplot(3,2,i+1)
	plt.scatter(days, x_vis[x_vis.columns.values[i]])
	plt.scatter(days[day_index], x_vis[x_vis.columns.values[i]][day_index], color='r')
	plt.title(x_vis.columns.values[i])

plt.show()

#Evaluation
train_score = regressor.score(X_train, Y_train)
test_score = regressor.score(X_test, Y_test) 

from sklearn import metrics
print(metrics.mean_absolute_error(Y_test, Y_pred)) 
print(metrics.mean_squared_error(Y_test, Y_pred)) 
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))











