import numpy as np
import pandas as pd

def nse(ypred, ytest):
    zahlersum=[]
    nennersum=[]
    qo=np.sum(ytest)/len(ytest)
    counter=0
    while counter < len(ypred):

        zahler = (ypred[counter][0]-ytest[counter])**2
        zahlersum.append(zahler)
        nenner = (ytest[counter]-qo)**2
        nennersum.append(nenner)
        counter+=1
    zahlersum = np.sum(zahlersum)
    nennersum = np.sum(nennersum)
    nse = 1 - (zahlersum/nennersum)
    return nse

