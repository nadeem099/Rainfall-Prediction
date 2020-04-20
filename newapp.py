#Flask applicatoin
from flask import Flask, request, jsonify, render_template
import requests 
import math
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
modelLR = pickle.load(open('modelLR.pkl', 'rb'))
modelMLR = pickle.load(open('modelMLR.pkl','rb'))
modelPR = pickle.load(open('modelPR.pkl','rb'))
modelLogR = pickle.load(open('modelLogR.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/algos', methods = ['GET','POST'])
def algos():
    city = request.form['location']
    city = city.upper() 
    api_address = "https://api.openweathermap.org/data/2.5/weather?appid=8758f651bc9f419b2156cf8d7b566282&q="
    url = api_address+city
    json_data = requests.get(url).json()
    #country code
    country_code = json_data["sys"]["country"]
    #for Multiple linear Regression
    L1=[]
    #temperature 
    kelvin_temp = json_data["main"]["temp_min"]
    celcisus_temp = kelvin_temp - 273.15
    fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
    formatted_data1 = math.floor(fahrenheit_temp)
    L1.append(formatted_data1)
    kelvin_temp = json_data["main"]["temp_max"]
    celcisus_temp = kelvin_temp - 273.15
    fahrenheit_temp = celcisus_temp * ( 9 / 5 ) + 32
    formatted_data2 = math.floor(fahrenheit_temp)
    L1.append(formatted_data2)
    #Humidity
    L1.append(80)
    formatted_data4 = json_data["main"]["humidity"]
    L1.append(formatted_data4)
    #sea level pressure
    L1.append(990)
    L1.append(1010)
    formatted_data5 = json_data["main"]["pressure"]
    L1.append(formatted_data5)
    #cloud cover
    formatted_data6 = json_data["clouds"]["all"]
    L1.append(formatted_data6)
    L1.append(formatted_data6)
    L1.append(formatted_data6)
    L1.append(formatted_data6)
    #shortwave radiation
    L1.append(0)
    #wind speed
    L1.append(0)
    formatted_data7 = json_data["wind"]["speed"]
    L1.append(formatted_data7)   
    #wind gust
    L1.append(0)
    L1.append(0)
    sample = np.array(L1)
    sample= sample.reshape(1, -1)
    
    #for rest of the algorithms
    L2=[]
    #temperature 
    kelvin = json_data["main"]["temp_min"]
    celcisus = kelvin - 273.15
    fahrenheit = celcisus * ( 9 / 5 ) + 32
    formatted1 = math.floor(fahrenheit)
    L2.append(formatted1)
    kelvin = json_data["main"]["temp_max"]
    celcisus = kelvin - 273.15
    fahrenheit = celcisus * ( 9 / 5 ) + 32
    formatted2 = math.floor(fahrenheit)
    L2.append(formatted2)
    kelvin = json_data["main"]["temp"]
    celcisus = kelvin - 273.15
    fahrenheit = celcisus * ( 9 / 5 ) + 32
    formatted3 = math.floor(fahrenheit)
    L2.append(formatted3)
    #Humidity
    L2.append(40)
    L2.append(80)
    formatted4 = json_data["main"]["humidity"]
    L2.append(formatted4)
    #sea level pressure
    L2.append(990)
    L2.append(1010)
    formatted5 = json_data["main"]["pressure"]
    L2.append(formatted5)
    #cloud cover
    formatted6 = json_data["clouds"]["all"]
    L2.append(formatted6)
    L2.append(formatted6)
    L2.append(formatted6)
    L2.append(formatted6)
    #sunshine duration
    L2.append(0)
    #shortwave radiation
    L2.append(0)
    #wind speed
    L2.append(0)
    L2.append(3)
    formatted7 = json_data["wind"]["speed"]
    L2.append(formatted7)
    #wind direction dominant
    L2.append(0)
    #wind gust
    L2.append(0)
    L2.append(0)
    
    L2.append(0)
    sample2 = np.array(L2)
    sample2= sample2.reshape(1, -1)
    
    #LR
    prediction1 = modelLR.predict(sample2)
    prediction1 = prediction1[0][0]
    prediction1 = math.floor(prediction1)
    if (prediction1 < 0):
        prediction1 = 0
    #condition of weather:
    if(prediction1 < 2):
        p1 = "No Rain"
    elif(prediction1 >= 2 and prediction1 < 10):
        p1 = "Drizzles"
    elif(prediction1 >= 10 and prediction1 < 20):
        p1 = "Moderate Rains"
    else:
        p1 = "Heavy Rains"
        
    #MLR
    prediction2 = modelMLR.predict(sample)
    prediction2 =  prediction2[0][0]
    prediction2 = math.floor(prediction2)
    if (prediction2 < 0):
        prediction2 = 0
    #condition of weather:
    if(prediction2 < 2):
        p2 = "No Rain"
    elif(prediction2 >= 2 and prediction2 < 10):
        p2 = "Drizzles"
    elif(prediction2 >= 10 and prediction2 < 20):
        p2 = "Moderate Rains"
    else:
        p2 = "Heavy Rains"
        
    #PR
    poly_reg = PolynomialFeatures(degree=2)
    sample_poly = poly_reg.fit_transform(sample2)
    prediction3 = modelPR.predict(sample_poly)
    prediction3 = prediction3[0][0]
    prediction3 = math.floor(prediction3)
    if (prediction3 < 0):
        prediction3 = 0
    #condition of weather:
    if(prediction3 < 2):
        p3 = "No Rain"
    elif(prediction3 >= 2 and prediction3 < 10):
        p3 = "Drizzles"
    elif(prediction3 >= 10 and prediction3 < 20):
        p3 = "Moderate Rains"
    else:
        p3 = "Heavy Rains"
    
    #LogR
    classes = ['None', 'No Rain', 'Drizzles', 'Moderate Rains', 'Heavy Rains']
    prediction4 = classes[int(modelLogR.predict(sample2))]
    return render_template("algos2.html", 
                           city = "{}".format(city), 
                           country_code = "{}".format(country_code),
                           prediction_text1 = "Precipitation: {}".format(prediction1), 
                           pt1 = "{}".format(p1),
                           prediction_text2 = "Precipitation: {}".format(prediction2), 
                           pt2 = "{}".format(p2),
                           prediction_text3 = "Precipitation: {}".format(prediction3),
                           pt3 = "{}".format(p3),
                           prediction_text4 = "There is possibility of {} today.".format(prediction4),
                           temperature = "{}".format(formatted3),
                           wind_speed = "{}".format(formatted7),
                           humidity = "{}".format(formatted4),
                           pressure = "{}".format(formatted5))

if __name__ == "__main__":
    app.run(debug = False)