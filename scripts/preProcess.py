import numpy as np

def scale(x):
    # Set min/max attributes to reuse them for prediction
    xMin = x.min(axis=0)
    xMax = x.max(axis=0)
    # Scale x
    return (x - xMin) / (xMax - xMin), (xMin/xMax-1)*(xMax-xMin)*(-1)

def prepareData(forcing,streamflow,days):
    counter=days
    inputcounter=days
    inputTensor=np.ones((forcing.shape[0]*(365-days),counter,forcing.shape[2],forcing.shape[3],forcing.shape[4]), dtype='float32')
    targetTensor=np.ones((forcing.shape[0]*(365-days),1))
    year=0
    inputcounter=15
    while year<forcing.shape[0]:
        counter=days
        while counter<forcing.shape[1]:  
            rain = forcing[year,counter-days:counter]
            stream=streamflow[year,counter]
            inputTensor[inputcounter-days]=rain
            targetTensor[inputcounter-days]=stream
            counter+=1
            inputcounter+=1
        year+=1
    return inputTensor,targetTensor

def prepareTensor(forcing_numpy,streamflow_numpy,days):
    counter=days
    inputTensor = np.zeros((len(forcing_numpy)-counter,counter,5),float)
    targetTensor = streamflow_numpy[:len(forcing_numpy)-counter]
    while counter<len(forcing_numpy):
        rain = forcing_numpy[counter-days:counter]
        inputTensor[counter-days:counter-(days-1),:]=rain
        counter+=1
    inputTensor= np.reshape(inputTensor, (inputTensor.shape[0], inputTensor.shape[1],5))
    return inputTensor,targetTensor

def shuffle(forcingshuffle,streamflowshuffle):
    randomize = np.arange(len(forcingshuffle))
    np.random.shuffle(randomize)
    forcingshuffle = forcingshuffle[randomize]
    streamflowshuffle = streamflowshuffle[randomize]
    return forcingshuffle,streamflowshuffle
