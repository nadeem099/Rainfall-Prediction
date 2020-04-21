#multiple Linear regression
#importing libraries
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

#importing dataset
data = pd.read_csv("NewDatasetBE.csv")

#separating X and Y
X = data.iloc[:, :16].values
Y = data.iloc[:, 16].values
Y = Y.reshape(-1, 1)

#splitting dataset
X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Training model
Regressor = LinearRegression()
Regressor = Regressor.fit(X_train, Y_train)

#saving model to disk or dumping model in pkl extension
pickle.dump(Regressor,open('modelMLR.pkl','wb'))

#predicting test results
Y_pred = Regressor.predict(X_test)

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
#Humidity
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
#shortwave radiation
L.append(0)
#wind speed
L.append(0)
formatted_data7 = json_data["wind"]["speed"]
L.append(formatted_data7)
#wind gust
L.append(0)
L.append(0)
sample = np.transpose([L])
sample= sample.reshape(1, -1)

#predicting real time precipitation
op = Regressor.predict(sample)
p = op[0][0]
#if (p < 2):
#    p = 0
final_pred = print(p)

#condition of weather:
if(p < 2):
    print("No Rain")
elif(p >= 2 and p < 10):
    print("Drizzles")
elif(p >= 10 and p < 20):
    print("Moderate Rains")
else:
    print("Heavy Rains")
    
#Evaluation
train_score = Regressor.score(X_train, Y_train)
test_score = Regressor.score(X_test, Y_test)

from sklearn import metrics
print(metrics.mean_absolute_error(Y_test, Y_pred)) 
print(metrics.mean_squared_error(Y_test, Y_pred)) 
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

    


    
    
