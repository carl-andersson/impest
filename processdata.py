'''
Created on 18 jan. 2017

@author: carl
'''

from sklearn import preprocessing
import numpy as np
import scipy.io

def getData(name:str,path:str= "data/"):
    mat = scipy.io.loadmat(path+name)
    
    U = mat['U'];
    G = mat["G"];
    Y = mat["Y"];
    
    G = G / np.std(U, 1,keepdims=True) / np.std(Y, 1,keepdims=True);
    U = U / np.std(U, 1,keepdims=True);
    Y = Y / np.std(Y, 1,keepdims=True);
    
    if ('SNR' in mat):
        SNR = mat["SNR"];
    else:
        SNR = np.ones([np.shape(G)[0],1])*10 ;
    
    return {"g":G,"u":mat['U'],"y":Y,"SNR":SNR};


