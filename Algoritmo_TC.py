# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:22:44 2016

@author: george
"""

import graphlab
import matplotlib.pyplot as plt

def CalculateNumeratorForK(errors):
    product = 1
    for i in errors:
        product = product * i
    return product
    
def CalculateDenominatorForK(errors):
    suma = 0
    product = 1
    for i, v1 in enumerate(errors):
        for j, v2 in enumerate(errors):
            if j != i:            
                product = product * v2
        suma = suma + product
        product = 1
    return suma

def CalculatePredict(sbs,day,daysModelRegression):
    weights = []
    predictions = []
    errors = []
    for i in daysModelRegression:
        rang = sbs[day - i : day]
        model = graphlab.linear_regression.create(rang, target='SBS', features=['CodigoFecha'],validation_set=None)
        errors.append(model.evaluate(rang)['rmse'])
        predictions.append(model.predict(rang[rang['CodigoFecha']==day])[0])  
    
    #Hallamos la cariable K
    numerator = CalculateNumeratorForK(errors)
    denominator = CalculateDenominatorForK(errors)
    k = numerator/ denominator
    
    #Calculamos los pesos
    for e in errors: 
        try:
            weights.append(k/e) 
        except ZeroDivisionError:
            weights.append(1)                        
    
    #Calculamos el tipo de cambio predecido
    tcPredict = sum([a*b for a, b in zip(weights, predictions)])     
    
    return (tcPredict, weights, predictions)    
    
def CalculateError(sbs,day,daysModelRegression,tcReal):
    tcPredict, weights, predictions = CalculatePredict(sbs,day,daysModelRegression)
    row = {'error': [], 'a1': [], 'a2': [], 'a3': [], 'a4': [], 'real': [], 'predict': []}    
    row['error'].append((tcReal - tcPredict)**2)
    row['a1'].append(weights[0])
    row['a2'].append(weights[1])
    row['a3'].append(weights[2])
    row['a4'].append(weights[3])
    row['real'].append(tcReal)
    row['predict'].append(tcPredict)
    return row 

def ComputeCost(theta, coefficients):
    return coefficients[0] + coefficients[1]*theta[0] + coefficients[2]*theta[1] + coefficients[3]*theta[2] + coefficients[4]*theta[3] 
    
def GradientDescent(theta, alpha, numIters, coefficients):
    JHistory = [] 
    iteracion = numIters
    for i in range(1,numIters):
        if ComputeCost(theta,coefficients)>=0:        
            JHistory.append(ComputeCost(theta,coefficients))
            auxTheta = [0,0,0,0]
            auxTheta[0] = theta[0] - alpha * coefficients[1]
            auxTheta[1] = theta[1] - alpha * coefficients[2]
            auxTheta[2] = theta[2] - alpha * coefficients[3]
            auxTheta[3] = theta[3] - alpha * coefficients[4]
            theta[0] = auxTheta[0]
            theta[1] = auxTheta[1]
            theta[2] = auxTheta[2]
            theta[3] = auxTheta[3]
        else:
            iteracion = i            
            break
    return (iteracion,theta,JHistory)

def CalculateWeightsHistoric(tableError):
    avgWeights = []
    dataError = graphlab.SFrame(tableError) #Convertimos a SFrame la tabla de errores
    avgWeights.append(float(sum(dataError['a1']))/len(dataError) if len(dataError) > 0 else float('nan'))
    avgWeights.append(float(sum(dataError['a2']))/len(dataError) if len(dataError) > 0 else float('nan'))
    avgWeights.append(float(sum(dataError['a3']))/len(dataError) if len(dataError) > 0 else float('nan'))
    avgWeights.append(float(sum(dataError['a4']))/len(dataError) if len(dataError) > 0 else float('nan'))
    return avgWeights

def CalculateDayPredictNewModel(sbs,daysModelRegression,dayModel,weights,dayPredict):
    predictions = []    
    for i in daysModelRegression:
        rang = sbs[dayModel - i : dayModel]
        model = graphlab.linear_regression.create(rang, target='SBS', features=['CodigoFecha'],validation_set=None)
        predictions.append(model.predict(graphlab.SFrame(dayPredict))[0])
    return sum([a*b for a, b in zip(weights, predictions)])


"""***********************************************************
Programa Principal
***********************************************************"""

sbs = graphlab.SFrame('Diarias-20160430-000800_maker.csv')
day = len(sbs)-1 #Comenzaremos a analizar desde el dias mas actual
daysModelRegression = [2,7,30,365] #Lista con los diferentes dias de antiguedad con los que se calcularan los modelos
tableError = {'error': [], 'a1': [], 'a2': [], 'a3': [], 'a4': [], 'real': [], 'predict': []}

#Recorremos cada uno de los valores de la data llamada 'sbs'.
#En este caso solo leeremos 50 valores, para realizar pruebas.
for i in range (0,50):
    tcReal = sbs['SBS'][day-i]
    rowError = CalculateError(sbs,day-i,daysModelRegression,tcReal)    
    for key in rowError.keys():
        tableError[key].append(rowError[key][0])
    
#Convertimos a SFrame la tabla de errores    
dataError = graphlab.SFrame(tableError)
#Calculamos el modelo de regresion lineal de la tabla de errores
modelError = graphlab.linear_regression.create(dataError, target='error', features=['a1','a2','a3','a4'],validation_set=None)

#Obtenemos los coeficientes del modelo de regresion lineal de los errores
coefficients = []
for i,v in enumerate(modelError.get('coefficients')):  
    coefficients.append(v['value'])

"""*************************************
Metodo 1: Calculo de a1, a2, a3 y a4 como los promedios de estos mismos valores en la tabla de errores
*************************************"""
avgWeights = []
avgWeights.append(float(sum(dataError['a1']))/len(dataError) if len(dataError) > 0 else float('nan'))
avgWeights.append(float(sum(dataError['a2']))/len(dataError) if len(dataError) > 0 else float('nan'))
avgWeights.append(float(sum(dataError['a3']))/len(dataError) if len(dataError) > 0 else float('nan'))
avgWeights.append(float(sum(dataError['a4']))/len(dataError) if len(dataError) > 0 else float('nan'))


"""*************************************
Metodo 2: Calculo de a1, a2, a3 y a4 tal que permitan que el error sea cercano a cero. Se usa el algoritmo del gradiente de descenso.
*************************************"""

JHistory = []
theta = [0,0,0,0]
alpha = 0.01
numIters = 600

InitialJHistory = ComputeCost(theta, coefficients)
iteracion, theta, JHistory = GradientDescent(theta, alpha, numIters, coefficients)
#plt.plot(range(1,iteracion),JHistory,'.')

"""
#Calculamos un factor tal que los theta's encontrados sumen 1.
calculo = 0
for i in theta:
    calculo = calculo + (i)

factor = 1 / calculo
for i,t in enumerate(theta):
    theta[i] = factor * (t)
"""

#En esta parte calculamos el tipo de cambio del dia 02/05
lastDay = len(sbs)-1 #Hace referencia al ultimo dia de la data, equivalente al 29/04
dayPredict = {'CodigoFecha' : [lastDay + 1]} #Dia a predecir el ultimo dia + 1
predictions = []

for i in daysModelRegression:
    rang = sbs[lastDay - i : lastDay]
    model = graphlab.linear_regression.create(rang, target='SBS', features=['CodigoFecha'],validation_set=None)
    predictions.append(model.predict(graphlab.SFrame(dayPredict))[0])

tcDayPredictMetodo1 = sum([a*b for a, b in zip(avgWeights, predictions)])   
tcDayPredictMetodo2 = sum([a*b for a, b in zip(theta, predictions)])/sum(theta)


"""*************************************
Calculo de predicciones del tipo de cambio, dado un periodo por el usuario definido en la variable daysPredict. 
Se usa el metodo 1 que calcula el promedio de las ponderaciones

Estos son los dos valores que debes modificar para probar el modelo de prediccion vs los valores reales. 
Probar con dias de meses pasados. Si prueba con el dia actual = len(sbs) entonces en el grafico los valores reales seran cero,
ya que no existe data real del futuro. Solo se mostrar el grafico de las predicciones."""

size = len(sbs) #El maximo valor permitido es len(sbs). Es el dia inicial desde el cual se desea calcular las predicciones
daysPredict = 90 #Cantidad de dias a predecir

"""****************************************************************************************************************************"""
"""****************************************************************************************************************************"""

lastDay = size - 1
copy = graphlab.SFrame.copy(sbs)
copy = copy[0:size]
weights = []
newPredictions = {'CodigoFecha': [], 'TcDayPredict': []}
for x in range(1,daysPredict+1):
    dayPredict = {'CodigoFecha' : [lastDay + x]} #Dia a predecir el ultimo dia + 1
    weights = CalculateWeightsHistoric(tableError)
    dayModel = lastDay + x - 1 #El modelo de prediccion de cada nuevo dia se realizar con el modelo del dia anterior
    tcDayPredict = CalculateDayPredictNewModel(copy,daysModelRegression,dayModel,weights,dayPredict)
    newPredictions['CodigoFecha'].append(lastDay + x + 1)
    newPredictions['TcDayPredict'].append(tcDayPredict)    
    copy = copy.append(graphlab.SFrame({'Fecha': ['2016'], 'CodigoFecha': [lastDay + x + 1], 'SBS': [tcDayPredict]}))
    rowError = CalculateError(copy,lastDay + x,daysModelRegression,tcDayPredict)
    for key in rowError.keys():
        tableError[key].append(rowError[key][0])

aux = sbs['SBS'][lastDay+1:lastDay+daysPredict+1]
if len(aux) == 0:
    aux = graphlab.SArray([])
if len(aux) < daysPredict:
    for x in range(len(aux)+1,daysPredict+1):
        aux = aux.append(graphlab.SArray([0.0]))
        
plt.gca().set_color_cycle(['red', 'blue'])
plt.plot(range(1,daysPredict+1),newPredictions['TcDayPredict'],'.')
plt.plot(range(1,daysPredict+1),aux,'.')
plt.legend(['Prediccion', 'Real'], loc='lower right')
plt.show()

rtn = graphlab.SFrame(newPredictions)
rtn.save('predictions.csv', format='csv')

