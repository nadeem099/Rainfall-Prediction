#Logistic Regression
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import metrics, datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
data = pd.read_csv("NewDataset.csv")
X = data.drop(['Precipitation Total'], axis=1)
Y = data['Precipitation Total']
Y = Y.values.reshape(-1, 1)

Y2 = []

x1 = pd.DataFrame(columns=X.columns.values)
x2 = pd.DataFrame(columns=X.columns.values)
x3 = pd.DataFrame(columns=X.columns.values)
x4 = pd.DataFrame(columns=X.columns.values)
for i in range(Y.size):
	if(Y[i]<2):
		Y2.append(1)
		x1.loc[i] = X.loc[i]
	elif(Y[i]>=2 and Y[i]<15):
		Y2.append(2)
		x2.loc[i] = X.loc[i]
	elif(Y[i]>=15 and Y[i]<25):
		Y2.append(3)
		x3.loc[i] = X.loc[i]
	else:
		Y2.append(4)
		x4.loc[i] = X.loc[i]

Y2 = np.array(Y2).reshape(len(Y2), )

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y2 , test_size = 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

logr = LogisticRegression(multi_class='ovr', solver='liblinear').fit(X_train, Y_train)

Y_pred = logr.predict(X_test)

##Data from weather api
#import requests
#import math
#api_address = "https://api.openweathermap.org/data/2.5/weather?appid=8758f651bc9f419b2156cf8d7b566282&q="
#city = input("Enter city Name:")
#url = api_address+city
#json_data = requests.get(url).json() 
#L=[]
##temperature 
#kelvin_temp = json_data["main"]["temp_min"]
#celcisus_temp = kelvin_temp - 273.15
#fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
#formatted_data1 = math.floor(fahrenheit_temp)
#L.append(formatted_data1)
#kelvin_temp = json_data["main"]["temp_max"]
#celcisus_temp = kelvin_temp - 273.15
#fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
#formatted_data2 = math.floor(fahrenheit_temp)
#L.append(formatted_data2)
#kelvin_temp = json_data["main"]["temp"]
#celcisus_temp = kelvin_temp - 273.15
#fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
#formatted_data3 = math.floor(fahrenheit_temp)
#L.append(formatted_data3)
##Humidity
#L.append(40)
#L.append(80)
#formatted_data4 = json_data["main"]["humidity"]
#L.append(formatted_data4)
##sea level pressure
#L.append(990)
#L.append(1010)
#formatted_data5 = json_data["main"]["pressure"]
#L.append(formatted_data5)
##cloud cover
#formatted_data6 = json_data["clouds"]["all"]
#L.append(formatted_data6)
#L.append(formatted_data6)
#L.append(formatted_data6)
#L.append(formatted_data6)
##sunshine duration
#L.append(0)
##shortwave radiation
#L.append(0)
##wind speed
#L.append(0)
#L.append(3)
#formatted_data7 = json_data["wind"]["speed"]
#L.append(formatted_data7)
##wind direction dominant
#L.append(0)
##wind gust
#L.append(0)
#L.append(0)
#L.append(0)
#sample = np.transpose([L])
#sample= sample.reshape(1, -1)
#
##predicting real time precipitation
#classes = ['None', 'No Rain', 'Drizzles', 'Moderate Rains', 'Heavy Rains']
#print(classes[int(logr.predict(sample))])
#
#pickle.dump(logr,open('modelLogR.pkl','wb'))
#



x1 = x1.filter(['Temperature Avg', 'Relative Humidity Avg', 'Mean Sea Level Pressure', 'Cloud Cover Total', 'Wind Speed Avg', 'Wind Gust Avg'], axis=1)
x2 = x2.filter(['Temperature Avg', 'Relative Humidity Avg', 'Mean Sea Level Pressure', 'Cloud Cover Total', 'Wind Speed Avg', 'Wind Gust Avg'], axis=1)
x3 = x3.filter(['Temperature Avg', 'Relative Humidity Avg', 'Mean Sea Level Pressure', 'Cloud Cover Total', 'Wind Speed Avg', 'Wind Gust Avg'], axis=1)
x4 = x4.filter(['Temperature Avg', 'Relative Humidity Avg', 'Mean Sea Level Pressure', 'Cloud Cover Total', 'Wind Speed Avg', 'Wind Gust Avg'], axis=1)


for i in range(6):
	plt.subplot(3,2,i+1)
	plt.scatter(x1.index.values, x1[x1.columns.values[i]], color='b')
	plt.scatter(x2.index.values, x2[x2.columns.values[i]], color='r')
	plt.scatter(x3.index.values, x3[x3.columns.values[i]], color='g')
	plt.scatter(x4.index.values, x4[x4.columns.values[i]], color='y')
	plt.title(x1.columns.values[i])

blue_patch = mpatches.Patch(color='blue', label='No rains')
red_patch = mpatches.Patch(color='red', label='Drizzles')
green_patch = mpatches.Patch(color='green', label='Moderate rains')
yellow_patch = mpatches.Patch(color='yellow', label='Heavy rains')
plt.legend(handles=[blue_patch, red_patch, green_patch, yellow_patch], bbox_to_anchor=(1.05, 2), loc=2, borderaxespad=0.)

plt.show()

#Evaluation
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#scores
train_score = logr.score(X_train, Y_train)
test_score = logr.score(X_test, Y_test)

#accuracy score 
from sklearn.metrics import accuracy_score, classification_report
ac = accuracy_score(Y_test, Y_pred)
cf = classification_report(Y_test, Y_pred)


