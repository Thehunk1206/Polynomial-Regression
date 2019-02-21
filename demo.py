import numpy as np
from numpy import polyfit
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pt

import time


def main():
    data = pd.read_csv('CleanData.csv')

    data.drop(['Unnamed: 0'],axis=1, inplace=True)
    data.drop(data.index[[0,1,5,9,73,74,75,710,711,712,713,714,715,716,717,718,719]], inplace=True)

    print(data.info())
    print(data.describe())

    fig = pt.figure()

    graph1 = fig.add_subplot(1,1,1)


    #all data
    x = data['Decimal Date'].tolist()
    y = data['Carbon Dioxide (ppm)'].tolist()
    z = data['Carbon Dioxide Fit (ppm)'].tolist()
    a = data['Seasonally Adjusted CO2 (ppm)'].tolist()
    b = data['Seasonally Adjusted CO2 Fit (ppm)'].tolist()

    #training data
    x_train = x[0:600]
    a_train = a[0:600]

    #testing data
    x_test =  x[601:703]#[2017.0411,2018.0411,2019.0411,2020.0411,2021.0411,2022.0411,2023.0411,2024.0411,2025.0411]
    #actual data
    a_actual = a[601:703]

    degree = 2
    startTime = time.time()
    coef = polyfit(x_train,a_train,degree)
    coefList = coef.tolist()
    #print("coef: %s" %coef)
    print(coefList)

    #training model
    training_curve = list()

    for i in range(len(x_train)):
        values = coef[-1]
        for d in range(degree):
            values += x_train[i]**(degree-d) * coef[d]
        training_curve.append(values)

    #first graph
    graph1.scatter(x_train,a_train,color='red',label='Carbon Dioxide (ppm)')
    graph1.plot(x_train,training_curve,color='blue',label='Regression Line')
    graph1.legend(loc='upper left')

    #coumpute error for training curve
    training_error = 0
    for i in range(len(training_curve)):
        training_error+= ((a_train[i]- (training_curve[i]))**2)/len(training_curve)
    print("==================Error During training=======================")
    print("MSE = ", training_error)


    #predicting concentration of CO2
    predicted_list = list()
    for i in range(len(x_test)):
        predict_val = coef[-1]
        for deg in range(degree):
            predict_val += x_test[i]**(degree-deg) * coef[deg]
        predicted_list.append(predict_val)

    #execution time
    print("========================================================")
    endTime = time.time()
    print("Approx execution time: ", endTime-startTime,"Seconds")

    print("==================prediction===========================")
    for i in range(len(x_test)):
        print("Predicted value of CO2(in ppm) for Year:",x_test[i],predicted_list[i])

    #coumpute error after prediction
    prediction_error = 0
    for i in range(len(predicted_list)):
        prediction_error+= ((a_actual[i]- (predicted_list[i]))**2)/len(predicted_list)
    print("==================PRediction Error=======================")
    print("MSE FOR Prediction = ", prediction_error)



    pt.show()

main()
