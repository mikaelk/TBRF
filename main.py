#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 15:00:00 2017
main.py
Main script which loads data, fits a ML regressor and makes
predictions.
Change log:
    v1: adapted from main_TBDT_v4.py, added all regressors implemented so far
    v2: add new features from Wang 2016: A Priori Assessment of Prediction Confidence
        v2_2: added writing tensor basis coefficient scalar fields
        v2_4: training/test data can be reloaded separately
    v3: cleaned up
        v3_2: analyze whether the trees can be used for U.Q.
        v3_3: new BFS case added + save pickled instance of the data
    v4: cleaned, added amount of features used for splitting in DT algorithm (tbdt_v6)
    v5: feature importance
    v6: fixed the decision tree when selecting random variables for splitting (tbdr_v8 & tbrf_v4)
    v7: getFeaturesWang2(): corrected turbulent/shear time scale
    v8: getFeaturesWang3(): added last wang feature + adjusted normalization (TreeFiles_1612)

@author: Mikael Kaandorp
"""

from __future__ import division
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys

from sklearn.ensemble import RandomForestRegressor
from tbnn import NetworkStructure, TBNN                                        #Tensor Basis Neural Network, see https://github.com/tbnn/tbnn
from tbdt_v8 import TBDT                                                       #Tensor Basis Decision Tree algorithm
from tbrf_v4 import TBRF                                                       #Tensor Basis Random Forest algorithm
import PyFOAM as pyfoam                                                        #functions for turbulence modelling and OpenFOAM
from importAndInterpolate_v6 import fn_importAndInterpolate                    #contains functions to read DNS and RANS data     
import helperFunctions as helperFn                                             #contains functions to write/plot data



def filterField(inputData, std, filter_spatial='Gaussian'):
    """
    Filter a field (e.g. predicted b_ij) spatially using a gaussian or median filter
    """
    if len(inputData.shape) == 4:
        outputData = np.zeros(inputData.shape)
        for i1 in range(inputData.shape[0]):
            for i2 in range(inputData.shape[1]):
                if filter_spatial == 'Gaussian':
                    outputData[i1,i2,:,:] = ndimage.gaussian_filter(inputData[i1,i2,:,:], 
                              std, order=0, output=None, mode='nearest', cval=0.0, truncate=10.0)
                elif filter_spatial == 'Median':
                    outputData[i1,i2,:,:] = ndimage.median_filter(inputData[i1,i2,:,:], 
                              size=std, mode='nearest')
    
    else: #TODO: other input shapes
        pass

    return outputData


def medianFilter(signal,threshold,bounds=False):
    """
    Median filter used to filter out outliers in the random forest (i.e. decision trees
    which are far away from the expected value)
    bounds: anisotropy values cannot be larger than 0.6, thus these values can be filtered out as well if set to true
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = np.zeros(signal.shape) if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    
    if bounds:
        mask2 = np.abs(signal) > 0.6
        
        # cancel if all values evaluate to true
        if np.all(mask2):
            mask2 = np.zeros(signal.shape,dtype=bool)
            
        # combine masks by using or    
        mask = (mask|mask2)

    # return the values which are not outliers
    outputSignal = signal[~mask]
    return outputSignal


def medianFilteredField(inputData,threshold,bounds=False):
    """
    Return a median filtered field for a random forest prediction
    """
    outputData = np.zeros([inputData.shape[0],inputData.shape[1]])
    for i1 in range(inputData.shape[0]):
        for i2 in range(inputData.shape[1]):
            outputData[i1,i2] = np.mean(medianFilter(inputData[i1,i2,:],threshold,bounds=bounds))
    return outputData
    

def normalizeTrainingFeatures(X_training,cap):
    """
    normalize training features, output the normalized features and mu/std original data
    which will be used to normalize the test features
    """
    std_inv = np.zeros(X_training.shape[0])
    mu_inv = np.zeros(X_training.shape[0]) 
    for i1 in range(X_training.shape[0]):
        # process consists of two parts: normalizing and removing outliers,
        # and normalizing and keeping the std/mean for later processing
        std_temp = np.std(X_training[i1,:])
        mu_temp = np.mean(X_training[i1,:])
        
        #normalize
        X_training[i1,:] = (X_training[i1,:] - mu_temp) / std_temp
        
        #cap
        X_training[i1,:][X_training[i1,:]> cap] = cap
        X_training[i1,:][X_training[i1,:]< -cap] = -cap
        
        #de-normalize
        X_training[i1,:] = (X_training[i1,:]*std_temp) + mu_temp
        
        # data with outliers removed: again get the mean and std
        std_inv[i1] = np.std(X_training[i1,:])
        mu_inv[i1] = np.mean(X_training[i1,:])
        
        #normalize the data with no outliers
        X_training[i1,:] = (X_training[i1,:] - mu_inv[i1]) / std_inv[i1]
    return X_training, std_inv, mu_inv
        
def normalizeTestFeatures(X_test, std_training, mu_training, Cap_inv): 
    """
    normalize test features according to the std/mu from the training features + remove outliers
    """
    for i1 in range(X_test.shape[0]): #rescale invariants according to training data
        X_test[i1,:] = (X_test[i1,:]-mu_training[i1])/std_training[i1]
        X_test[i1,:][X_test[i1,:]>Cap_inv] = Cap_inv
        X_test[i1,:][X_test[i1,:]<-Cap_inv] = -Cap_inv
    return X_test


def randomSampling(X,Y,TB,fraction,replace):
    """
    take random samples with/without replacement from training data, 
    N_samples = fraction*length(array)
    """
    
    size_out = np.round(fraction*X.shape[1])
    #samples from the columns:
    idx = np.random.choice(X.shape[1],int(size_out),replace=replace)
    
    X_out = X[:,idx]
    Y_out = Y[:,idx]
    TB_out = TB[:,:,idx]
    
    return X_out,Y_out,TB_out


def make_realizable(labels):
    """
    From Ling et al. (2016), see https://github.com/tbnn/tbnn:
    This function is specific to turbulence modeling.
    Given the anisotropy tensor, this function forces realizability
    by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
    Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
    constraints from first step, so this function should be called iteratively to get convergence
    to a realizable state.
    :param labels: the predicted anisotropy tensor (num_points X 9 array)
    """
    numPoints = labels.shape[0]
    A = np.zeros((3, 3))
    for i in range(numPoints):
        # Scales all on-diags to retain zero trace
        if np.min(labels[i, [0, 4, 8]]) < -1./3.:
            labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
        if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
            labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
            labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
        if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
            labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
            labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
        if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
            labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
            labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

        # Enforce positive semidefinite by pushing evalues to non-negative
        A[0, 0] = labels[i, 0]
        A[1, 1] = labels[i, 4]
        A[2, 2] = labels[i, 8]
        A[0, 1] = labels[i, 1]
        A[1, 0] = labels[i, 1]
        A[1, 2] = labels[i, 5]
        A[2, 1] = labels[i, 5]
        A[0, 2] = labels[i, 2]
        A[2, 0] = labels[i, 2]
        evalues, evectors = np.linalg.eig(A)
        if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
            evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]
        if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
            evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
            A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
            for j in range(3):
                labels[i, j] = A[j, j]
            labels[i, 1] = A[0, 1]
            labels[i, 5] = A[1, 2]
            labels[i, 2] = A[0, 2]
            labels[i, 3] = A[0, 1]
            labels[i, 7] = A[1, 2]
            labels[i, 6] = A[0, 2]

    return labels

def analyzeBarycentricMap(eigVals,model,plot_stressType=True):
    C1c = np.zeros([eigVals.shape[1],eigVals.shape[2]])
    C2c = np.zeros([eigVals.shape[1],eigVals.shape[2]])
    C3c = np.zeros([eigVals.shape[1],eigVals.shape[2]])
    
    for i1 in range(eigVals.shape[1]):
        for i2 in range(eigVals.shape[2]):

            C1c[i1,i2] = eigVals[0,i1,i2] - eigVals[1,i1,i2]
            C2c[i1,i2] = 2*(eigVals[1,i1,i2]-eigVals[2,i1,i2])
            C3c[i1,i2] = 3*eigVals[2,i1,i2] + 1
            
    baryMap = pyfoam.barycentricMap(eigVals)
    helperFn.plotBaryMap(baryMap,title=('Barycentric map, %s' % model))           
            
    if plot_stressType:
        # RGB map barycentric map
        helperFn.plotStressType(meshRANS[plotIndices[testParam['flowCase']]['x'],:,:],meshRANS[plotIndices[testParam['flowCase']]['y'],:,:],
                       2500,500,C1c,C2c,C3c,title=('Stress type in the flow domain, %s' % model))          
                      
#%% Set variables

np.random.seed(12345)

home                = './turbulenceData/' #Home folder in which the OpenFOAM cases and DNS data sets are located
Scale_SR            = 1 #scale the strain rate / rotation rate tensors using k and epsilon
Scale_TB            = 1 #scale the tensor basis according to Ling et al. (2016)
Scale_inv           = 1 #scale the invariants to std 1 and mean 0 
Cap_inv             = 5 # filter the invariants for std>Cap, std<-Cap
Realizable          = False # make the output realizable
Regressor           = 'TBRF' #TBDT, TBRF, TBNN, RF
reload_training     = True #reload training data
reload_test         = True #reload test data, can be set to false to quickly evaluate ML hyperparameters
read_treefile       = True #TBDT/TBRF: read tree file of specified tree name if file exists
SecondaryFeatures   = ['k','Wang'] #whether to use more features than just S and R: use 'k' for grad(k) features, Wang for Wang(2016) features
apriori             = False #plot a priori confidence parameters
write_results       = False #write results for b_ij to a data file for later processing
plot_stressType     = True #plot stress type from barycentric map in RGB
write_apriori       = False #write a-priori confidence data for further processing
plot_tensorBasis    = False #plot TB coefficients, and g*mag(T) for further analysis

#DeleteFeatures      = [] #Features which need to be deleted
DeleteFeatures      = [2,3,5,9,10,11,12,13,15,16,17]    # for k-omega

if reload_training:
    trainingParam   = {} #dict with all training parameters (lists)
if reload_test:
    testParam       = {} #dict with all test parameters (one parameters per dict)


#----------------------------------------- Possible training sets: ----------------------------------------------------------
#trainingParam['Re']         = [3500]
#trainingParam['turbModel']  = ['kOmega']
#trainingParam['flowCase']   = ['SquareDuct']
#trainingParam['time_end']   = [50000]
#trainingParam['Nx']         = [50]
#trainingParam['Ny']         = [50]
#trainingParam['frac']       = 1                    # how much of the total available training data should be used for training
#trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data
#trainingParam['nu']         = [0.00013770285714285714]

#trainingParam['Re']         = [3200,2900]
#trainingParam['turbModel']  = ['kOmega','kOmega']
#trainingParam['flowCase']   = ['SquareDuct','SquareDuct']
#trainingParam['time_end']   = [50000,50000]
#trainingParam['Nx']         = [50,50]
#trainingParam['Ny']         = [50,50]
#trainingParam['frac']       = 0.5                     # how much of the total available training data should be used for training
#trainingParam['replace']    = False    
#
#trainingParam['Re']         = [3200,2900,2600,2400]
#trainingParam['turbModel']  = ['kOmega','kOmega','kOmega','kOmega']
#trainingParam['flowCase']   = ['SquareDuct','SquareDuct','SquareDuct','SquareDuct']
#trainingParam['time_end']   = [50000,50000,50000,50000]
#trainingParam['Nx']         = [50,50,50,50]
#trainingParam['Ny']         = [50,50,50,50]
#trainingParam['frac']       = 0.25                     # how much of the total available training data should be used for training
#trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data
#trainingParam['nu']         = [0.0001506125,0.00016619310344827585, 0.00018536923076923077,0.00020081666666666668]

trainingParam['Re']         = [10595]
trainingParam['turbModel']  = ['kOmega']
trainingParam['flowCase']   = ['PeriodicHills']
trainingParam['time_end']   = [30000]
trainingParam['Nx']         = [140]
trainingParam['Ny']         = [150]
trainingParam['frac']       = 0.25                    # how much of the total available training data should be used for training
trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data
trainingParam['nu']         = [9.438414346389807e-05] 

#trainingParam['Re']         = [10595,5600,12600]
#trainingParam['turbModel']  = ['kOmega','kOmega','kOmega']
#trainingParam['flowCase']   = ['PeriodicHills','PeriodicHills','ConvDivChannel']
#trainingParam['time_end']   = [30000,30000,7000]
#trainingParam['Nx']         = [140,140,140]
#trainingParam['Ny']         = [150,150,100]
#trainingParam['frac']       = 0.375                    # how much of the total available training data should be used for training
#trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data
#trainingParam['nu']         = [9.438414346389807e-05,1.7857142857142857e-04,7.936507936507937e-05] 

#trainingParam['Re']             = [40000]
#trainingParam['turbModel']      = ['kOmega']
#trainingParam['flowCase']       = ['BackwardFacingStep3_after']
#trainingParam['time_end']       = [20000]
#trainingParam['Nx']             = [100]
#trainingParam['Ny']             = [140]
#trainingParam['nu']             = [8.750000000000001e-07]
#trainingParam['frac']       = 0.375                    # how much of the total available training data should be used for training
#trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data

#-------------------------------------- Possible test cases: -----------------------------------------------------

#testParam['Re']             = 3500
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'SquareDuct'
#testParam['time_end']       = 50000
#testParam['Nx']             = 50
#testParam['Ny']             = 50
#testParam['nu']             = 0.00013770285714285714

#testParam['Re']             = 3200
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'SquareDuct'
#testParam['time_end']       = 50000
#testParam['Nx']             = 50
#testParam['Ny']             = 50
#testParam['nu']             = 0.0001506125

testParam['Re']             = 5600
testParam['turbModel']      = 'kOmega'
testParam['flowCase']       = 'PeriodicHills'
testParam['time_end']       = 30000
testParam['Nx']             = 140
testParam['Ny']             = 150
testParam['nu']             = 1.7857142857142857e-04
         
#testParam['Re']             = 10595
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'PeriodicHills'
#testParam['time_end']       = 30000
#testParam['Nx']             = 140
#testParam['Ny']             = 150
#testParam['nu']             = 9.438414346389807e-05
#
#testParam['Re']             = 12600
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'ConvDivChannel'
#testParam['time_end']       = 7000
#testParam['Nx']             = 140
#testParam['Ny']             = 100
#testParam['nu']             = 7.936507936507937e-05 #necessary for creating the Wang features
##
#testParam['Re']             = 13700
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'CurvedBackwardFacingStep'
#testParam['time_end']       = 3000
#testParam['Nx']             = 140
#testParam['Ny']             = 150
#testParam['nu']             = 7.299270072992701e-05

#testParam['Re']             = 2800
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'PeriodicHills'
#testParam['time_end']       = 30000
#testParam['Nx']             = 140
#testParam['Ny']             = 150
#testParam['nu']             = 3.5714285714285714e-04 #necessary for creating the Wang features

#testParam['Re']             = 5100
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'BackwardFacingStep_after'
#testParam['time_end']       = 20000
#testParam['Nx']             = 150
#testParam['Ny']             = 140
#testParam['nu']             = 0.000196078431372549

#testParam['Re']             = 5100
#testParam['turbModel']      = 'kOmega'
#testParam['flowCase']       = 'BackwardFacingStep_before'
#testParam['time_end']       = 20000
#testParam['Nx']             = 150
#testParam['Ny']             = 90
#testParam['nu']             = 0.000196078431372549

#-------------------------------- further information, ignore -----------------------

# indices for x and y axes when plotting
plotIndices = {}
plotIndices['PeriodicHills'] = {}
plotIndices['PeriodicHills']['x'] = 0
plotIndices['PeriodicHills']['y'] = 1
plotIndices['ConvDivChannel'] = {}
plotIndices['ConvDivChannel']['x'] = 0
plotIndices['ConvDivChannel']['y'] = 1
plotIndices['CurvedBackwardFacingStep'] = {}
plotIndices['CurvedBackwardFacingStep']['x'] = 0
plotIndices['CurvedBackwardFacingStep']['y'] = 1
plotIndices['SquareDuct'] = {}
plotIndices['SquareDuct']['x'] = 1
plotIndices['SquareDuct']['y'] = 2
plotIndices['BackwardFacingStep_before'] = {}
plotIndices['BackwardFacingStep_before']['x'] = 0
plotIndices['BackwardFacingStep_before']['y'] = 1
plotIndices['BackwardFacingStep_after'] = {}
plotIndices['BackwardFacingStep_after']['x'] = 0
plotIndices['BackwardFacingStep_after']['y'] = 1

#%% Obtain training and test data
if reload_training: #reload data; can be set to false when quickly evaluating ML fit
    
    # -------------------------------------get training data-----------------------
    for i1 in range(len(trainingParam['Re'])):
        
        # get the RANS and DNS data for a given flow case
        dataRANS_training,dataDNS_training,meshRANS = fn_importAndInterpolate(home,trainingParam['flowCase'][i1],
                                                                              trainingParam['Re'][i1],trainingParam['turbModel'][i1],
                                                                              trainingParam['time_end'][i1],trainingParam['Nx'][i1],
                                                                              trainingParam['Ny'][i1],0,0,SecondaryFeatures)
        # obtain the RANS strain rate tensors 'S' and 'R'
        dataRANS_training['S'],dataRANS_training['R'] = pyfoam.getSRTensors(dataRANS_training['gradU'], 
                         Scale_SR,dataRANS_training['k'],dataRANS_training['epsilon'])
        
        # get all invariant input features for the machine learning algorithm.
        # check if more features are required, e.g. grad(k), or the features from Wang et al. (2016)
        if 'k' in SecondaryFeatures:
            dataRANS_training['A_k'] = pyfoam.getTkeFeatures(dataRANS_training['gradTke'],Scale_SR,dataRANS_training['k'],dataRANS_training['epsilon'])
            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R'],dataRANS_training['A_k']])
#            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R']])
        else:
            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R']])
        if 'Wang' in SecondaryFeatures:
            dataRANS_training['invariants'] = pyfoam.getInvariantsWang3(dataRANS_training,trainingParam['nu'][i1])
            
        # Get from the training data the anisotropy tensor bij, and decompose it so that the eigenvalues
        # and eigenvectors can be used for further analysis if necessary:
        dataDNS_training['eigValMat'],dataDNS_training['eigVecMat'] = pyfoam.eigenDecomposition(dataDNS_training['bij'])
        dataDNS_training['phi'] = pyfoam.eigenvectorToEuler(dataDNS_training['eigVecMat'])
        
        # get the basis tensors in the flow field
        dataRANS_training['tb'] = pyfoam.getTensorBasis(dataRANS_training['S'],dataRANS_training['R'],Scale_TB)
        
        #----------------- reshape data and stack different training cases
        temp_X_training = np.reshape(dataRANS_training['invariants'],[np.shape(dataRANS_training['invariants'])[0],trainingParam['Nx'][i1]*trainingParam['Ny'][i1]])
        
        #Delete specified features if specified
        if DeleteFeatures:
            temp_X_training = np.delete(temp_X_training,DeleteFeatures,axis=0)

        temp_Y_training = np.reshape(dataDNS_training['bij'],[9,trainingParam['Nx'][i1]*trainingParam['Ny'][i1]])
        temp_TB_training = np.reshape(dataRANS_training['tb'],[9,dataRANS_training['tb'].shape[2],dataRANS_training['tb'].shape[3]*dataRANS_training['tb'].shape[4]])
        
        # plot features and response for verification
        helperFn.plotFeatures2(temp_X_training, trainingParam['Nx'][i1], trainingParam['Ny'][i1],meshRANS,trainingParam['Re'][i1],'Training',plotIndices[trainingParam['flowCase'][i1]]['x'],plotIndices[trainingParam['flowCase'][i1]]['y'])
        helperFn.plotAnisotopy(temp_Y_training, trainingParam['Nx'][i1], trainingParam['Ny'][i1],meshRANS,trainingParam['Re'][i1],'Training',plotIndices[trainingParam['flowCase'][i1]]['x'],plotIndices[trainingParam['flowCase'][i1]]['y'])
        helperFn.plotBaryMap(dataDNS_training['baryMap'])
        
        #stack training data
        if i1 == 0:
            X_training = np.zeros(temp_X_training.shape)
            Y_training = np.zeros(temp_Y_training.shape)
            TB_training = np.zeros(temp_TB_training.shape)
            X_training = X_training + temp_X_training
            Y_training = Y_training + temp_Y_training
            TB_training = TB_training + temp_TB_training
        else:
            X_training = np.hstack([X_training,temp_X_training])
            Y_training = np.hstack([Y_training,temp_Y_training])
            TB_training = np.dstack([TB_training,temp_TB_training])

    # remove features with low variance, in case no features are specified to be removed
    if not DeleteFeatures:
        list_remove = []
        for i1 in range(X_training.shape[0]):
            if np.var(X_training[i1,:]) < 1e-10:
                list_remove.append(i1)
        X_training = np.delete(X_training,list_remove,axis=0)    
        
    # normalize input features to std 1 and mu 0, remove outliers above |std| > Cap_inv
    X_training, trainingParam['std'], trainingParam['mu'] = normalizeTrainingFeatures(X_training,Cap_inv)
        
    # random sampling of data points
    x_training,y_training,tb_training = randomSampling(X_training,Y_training,TB_training,trainingParam['frac'],trainingParam['replace'])
    

if reload_test: 
    #--------------------------------------get test data---------------------------
    dataRANS_test,dataDNS_test,meshRANS = fn_importAndInterpolate(home,testParam['flowCase'],testParam['Re'],
                                                                  testParam['turbModel'],testParam['time_end'],testParam['Nx'],testParam['Ny'],0,0,SecondaryFeatures)
    dataRANS_test['S'],dataRANS_test['R'] = pyfoam.getSRTensors(dataRANS_test['gradU'], 
                 Scale_SR,dataRANS_test['k'],dataRANS_test['epsilon'])
    dataRANS_test['invariants'] = pyfoam.getInvariants([dataRANS_test['S'],dataRANS_test['R']])
    #check if more features are required, e.g. grad(k)
    if 'k' in SecondaryFeatures:
        dataRANS_test['A_k'] = pyfoam.getTkeFeatures(dataRANS_test['gradTke'],Scale_SR,dataRANS_test['k'],dataRANS_test['epsilon'])
        dataRANS_test['invariants'] = pyfoam.getInvariants([dataRANS_test['S'],dataRANS_test['R'],dataRANS_test['A_k']])
#            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R']])
    else:
        dataRANS_test['invariants'] = pyfoam.getInvariants([dataRANS_test['S'],dataRANS_test['R']])
    if 'Wang' in SecondaryFeatures:
        # add invariants to dataRANS_training['invariants']        
        dataRANS_test['invariants'] = pyfoam.getInvariantsWang3(dataRANS_test,testParam['nu'])
        
    # Get from the test data bij, and decompose it so that the eigenvalues
    # and eigenvectors can be used for further analysis if necessary:
    dataRANS_test['eigValMat'],dataRANS_test['eigVecMat'] = pyfoam.eigenDecomposition(dataRANS_test['bij'])
    dataDNS_test['eigValMat'],dataDNS_test['eigVecMat'] = pyfoam.eigenDecomposition(dataDNS_test['bij'])
    
    # get basis tensors 
    dataRANS_test['tb'] = pyfoam.getTensorBasis(dataRANS_test['S'],dataRANS_test['R'],Scale_TB)
    
    X_test = np.reshape(dataRANS_test['invariants'],[np.shape(dataRANS_test['invariants'])[0],testParam['Nx']*testParam['Ny']])
    
    # remove specified input features
    if DeleteFeatures:
        X_test = np.delete(X_test,DeleteFeatures,axis=0)

    Y_test = np.reshape(dataDNS_test['bij'],[9,testParam['Nx']*testParam['Ny']])
    TB_test = np.reshape(dataRANS_test['tb'],[9,dataRANS_test['tb'].shape[2],dataRANS_test['tb'].shape[3]*dataRANS_test['tb'].shape[4]])
    
    # plot test data for verification
    helperFn.plotFeatures2(X_test, testParam['Nx'], testParam['Ny'],meshRANS,testParam['Re'],'Test',plotIndices[testParam['flowCase']]['x'],plotIndices[testParam['flowCase']]['y'])
    helperFn.plotAnisotopy(Y_test, testParam['Nx'], testParam['Ny'],meshRANS,testParam['Re'],'Test',plotIndices[testParam['flowCase']]['x'],plotIndices[testParam['flowCase']]['y'])
    helperFn.plotBaryMap(dataDNS_test['baryMap'])    
    
    # remove zero features
    if not DeleteFeatures:
        X_test = np.delete(X_test,list_remove,axis=0)  
    # normalize test features according to std and mu from training data, remove outliers |std| > Cap_inv
    X_test = normalizeTestFeatures(X_test, trainingParam['std'],trainingParam['mu'], Cap_inv)
   
    if apriori:
    # plot a priori prediction confidence parameters:
        print('Plotting a priori confidence parameters...')
        try: #singular matrices are possible in which case D_kde cannot be plotted
            D_kde_scott = helperFn.plotKDEDistance(x_training,X_test,meshRANS,plotIndices[testParam['flowCase']]['x'],plotIndices[testParam['flowCase']]['y'],kernelwidth='scott')
            D_kde_scott_mean = np.mean(D_kde_scott)            
            print('D_kde, mean: %f' % D_kde_scott_mean)
        except:
            print('Cannot plot D_kde_scott, possibly singular matrix')
        D_mahalanobis = helperFn.plotMahalanobisDistance(x_training,X_test,meshRANS,plotIndices[testParam['flowCase']]['x'],plotIndices[testParam['flowCase']]['y'])
        D_mahalanobis_mean = np.mean(D_mahalanobis)
        
        print('D_mah, mean: %f' % D_mahalanobis_mean)
        
        
#%% Train ML regressor and predict
if Regressor == 'TBDT':
    # ------------- TBDT hyperparameters: --------------------------------------
    regularization = True
    regularization_lambda = 1e-15
    write_g = False #write tensor basis coefficient fields
    splitting_features='all'
    min_samples_leaf = 1
    
    # ---------- set up the tree filename which can be read later on if necessary
    tree_filename_ReCase = ''
    for i in range(len(trainingParam['Re'])):
        tree_filename_ReCase = tree_filename_ReCase + trainingParam['flowCase'][i] + str(trainingParam['Re'][i]) + '_'        
    tree_filename_var = '%s_%ifeat_Regul%iLambda%2.1e_ScaleTB%i' % (trainingParam['turbModel'][0],X_training.shape[0],regularization,regularization_lambda,Scale_TB)

    # ------------ path where the tree file needs to be saved ------------------
    tree_filename = './TreeFiles/TBDT_' + tree_filename_ReCase + tree_filename_var
    
    # ------------ set-up the TBDT with the appropriate parameters -------------
    tbdt = TBDT(tree_filename=tree_filename,regularization=regularization,splitting_features=splitting_features,
                regularization_lambda=regularization_lambda, optim_split=True,optim_threshold=100,
                min_samples_leaf=min_samples_leaf)
    
    # ------------ fit and predict ---------------------------------------------
    tree = tbdt.fit(x_training,y_training,tb_training) 
    
    y_predict, g_tree = tbdt.predict(X_test,TB_test,tree)
    
    # write data to OpenFOAM format if necessary
    if write_g:
        for i in range(g_tree.shape[0]):
            helperFn.writeScalarField(('g%i'%i),g_tree[i,:],home,testParam['flowCase'],testParam['Re'],'kOmega',testParam['Nx'],testParam['Ny'],testParam['time_end'],'')


elif Regressor == 'TBRF':
    # ------------- TBRF hyperparameters: --------------------------------------
    regularization = True
    regularization_lambda = 1e-12
    write_g = False #write tensor basis coefficient fields
    splitting_features=11 # 'all', 'div3', 'sqrt', or an integer value
    min_samples_leaf = 9
    n_trees = 8
    
    # ----------- set up the tree filename which can be read later on if necessary 
    # The tree filenames contain the training data parameters, as well as the random
    # forest parameters such as min. features per node etc.
    tree_filename_ReCase = ''
    for i in range(len(trainingParam['Re'])):
        tree_filename_ReCase = tree_filename_ReCase + trainingParam['flowCase'][i] + str(trainingParam['Re'][i]) + '_'        
    tree_filename_var = '%s_%ifeat_Regul%iLambda%2.1e_ScaleTB%i' % (trainingParam['turbModel'][0],X_training.shape[0],regularization,regularization_lambda,Scale_TB)
 
    if splitting_features == 'all':
        pass
    elif isinstance(splitting_features,int):
        tree_filename_var = tree_filename_var + '_Split%i' % splitting_features
    if min_samples_leaf > 1:
        tree_filename_var = tree_filename_var + '_Sampl%i' % min_samples_leaf
    
    
    # ------------ path where the tree files need to be saved ------------------
    tree_filename = './TreeFiles/TBRF_TREE%i_' + tree_filename_ReCase + tree_filename_var
    
    # ------------ set-up the TBRF with the appropriate parameters -------------
    tbrf = TBRF(min_samples_leaf=min_samples_leaf,tree_filename=tree_filename,n_trees=n_trees,regularization=regularization,
                regularization_lambda=regularization_lambda, splitting_features=splitting_features,
                optim_split=True,optim_threshold=100,read_from_file=read_treefile)
    
    # ------------ fit and predict ---------------------------------------------
    forest = tbrf.fit(x_training,y_training,tb_training)
    
    print('Making predictions...')
    y_predict,bij_forest,g_forest = tbrf.predict(X_test,TB_test,forest)
    
    
    # filter predictions using the median filter (bij_MF), and spatial filter (bij_FMF):
    bij_MF = medianFilteredField(bij_forest,3,bounds=False)
    bij_FMF = filterField(np.reshape(bij_MF,[3,3,testParam['Nx'],testParam['Ny']]),[3,3])
    
    # get predicted tensor basis coefficients for further analysis
    g_MF = medianFilteredField(g_forest,3,bounds=False)
    g_FMF = filterField(np.reshape(g_MF,[1,10,testParam['Nx'],testParam['Ny']]),[3,3])
    g_FMF = np.reshape(g_FMF,[10,testParam['Nx']*testParam['Ny']])
    
    # write data to OpenFOAM format if necessary
    if write_g:
        for i in range(g_forest.shape[0]):
            helperFn.writeScalarField(('g%i'%i),g_FMF[i,:],home,testParam['flowCase'],testParam['Re'],'kOmega',testParam['Nx'],testParam['Ny'],testParam['time_end'],'_TBRF')

    
    RMSE_MF = np.sqrt(np.mean(np.square(bij_MF-Y_test)))
    print('RMSE b_ij, median filter: %f' % RMSE_MF)
    RMSE_FMF = np.sqrt(np.mean(np.square(bij_FMF-np.reshape(Y_test,[3,3,testParam['Nx'],testParam['Ny']]))))
    print('RMSE b_ij, median filter + gaussian filter: %f' % RMSE_FMF)
    
elif Regressor == 'RF':
    clf = RandomForestRegressor(n_estimators=100,oob_score=True)
    clf = clf.fit(x_training.T,y_training.T)
    y_predict = (clf.predict(X_test.T)).T


elif Regressor == 'TBNN':

    num_layers = 8  # Number of hidden layers in the TBNN
    num_nodes = 30  # Number of nodes per hidden layer
    max_epochs = 1000# Max number of epochs during training
    min_epochs = 100 # Min number of training epochs required
    interval = 10  # Frequency at which convergence is checked
    average_interval = 5  # Number of intervals averaged over for early stopping criteria
    split_fraction = 0.8  # Fraction of data to use for training
    learning_rate_decay = 0.99
    init_learning_rate = 1e-2
    min_learning_rate=2.5e-7
    
    # Define network structure
    structure = NetworkStructure()
    structure.set_num_layers(num_layers)
    structure.set_num_nodes(num_nodes)
    tbnn = TBNN(structure,min_learning_rate=1e-7,print_freq=interval,learning_rate_decay=learning_rate_decay)
    
    tbnn.fit(x_training.T, tb_training.T, y_training.T, max_epochs=max_epochs, 
             min_epochs=min_epochs, interval=interval, average_interval=average_interval, init_learning_rate=init_learning_rate)
        
    y_predict = (tbnn.predict(helperFn.copyAndTranspose(X_test),helperFn.copyAndTranspose(TB_test))).T

else:
    print('Unknown regressor (%s), exiting script' % Regressor)
    sys.exit()
    
    
if Realizable:
    # make prediction realizable (function is iterated 5 times)
    for i in range(5):
        y_predict = (make_realizable(y_predict.T)).T
        

#%% Analyze results

# anisotropy tensor DNS
bij_DNS = np.reshape(Y_test,[3,3,testParam['Nx'],testParam['Ny']])

bij_unfiltered = np.reshape(y_predict,[3,3,testParam['Nx'],testParam['Ny']])
if Regressor == 'TBRF':
    bij_filtered = bij_FMF
else:
    bij_filtered = filterField(bij_unfiltered,[3,3])

RMSE = np.sqrt(np.mean(np.square(bij_unfiltered-bij_DNS))) 
print('RMSE b_ij, no filter: %f' % RMSE)

RMSE_2 = np.sqrt(np.mean(np.square(bij_filtered-bij_DNS)))
print('RMSE b_ij, with filter: %f' % RMSE_2)


print('Plotting anisotropy predictions...')
# plot anisotropy tensor components
plt_indices = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]])
for i_plt in range(plt_indices.shape[0]):
    index = plt_indices[i_plt]

    maxval = np.max([np.abs(bij_DNS[index[0],index[1],:,:])])

    if maxval < 1e-12:
        maxval = 1e-12
    
    contour_levels = np.linspace(-1.05*maxval, 1.05*maxval, 50)
    cmap=plt.cm.coolwarm

    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex='col')
    subPlot1 = ax1.contourf(meshRANS[plotIndices[testParam['flowCase']]['x'],:,:], meshRANS[plotIndices[testParam['flowCase']]['y'],:,:], bij_DNS[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    subPlot2 = ax2.contourf(meshRANS[plotIndices[testParam['flowCase']]['x'],:,:], meshRANS[plotIndices[testParam['flowCase']]['y'],:,:], dataRANS_test['bij'][index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    subPlot3 = ax3.contourf(meshRANS[plotIndices[testParam['flowCase']]['x'],:,:], meshRANS[plotIndices[testParam['flowCase']]['y'],:,:], bij_unfiltered[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    subPlot4 = ax4.contourf(meshRANS[plotIndices[testParam['flowCase']]['x'],:,:], meshRANS[plotIndices[testParam['flowCase']]['y'],:,:], bij_filtered[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    
    ax1.set_title('DNS') #    
    ax2.set_title('RANS %s' % (testParam['turbModel']))
    ax3.set_title('%s prediction' % (Regressor))
    ax4.set_title('%s prediction, filtered' % (Regressor))
    plt.suptitle('Anisotropy tensor, component $b_{%i%i}$' % (index[0]+1, index[1]+1)) 
    f.subplots_adjust(right=0.8)  
    f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
    plt.show()

      
# barycentric map + plots
eigVals_ML = np.zeros([3,bij_unfiltered.shape[2],bij_unfiltered.shape[3]])
for i1 in range(bij_unfiltered.shape[2]):
    for i2 in range(bij_unfiltered.shape[3]):        
        a,b = np.linalg.eig(bij_filtered[:,:,i1,i2])
        eigVals_ML[:,i1,i2] = sorted(a, reverse=True)

analyzeBarycentricMap(eigVals_ML,Regressor,plot_stressType=plot_stressType)
analyzeBarycentricMap(dataDNS_test['eigVal'],'DNS',plot_stressType=plot_stressType)
analyzeBarycentricMap(dataRANS_test['eigVal'],'RANS',plot_stressType=plot_stressType)


if write_results:
    folder = 'DataFiles3'
    
    saveData = {}
    saveData['bijDNS'] = dataDNS_test['bij']
    saveData['bijRANS'] = dataRANS_test['bij']
    saveData['mesh'] = meshRANS
    
    if write_apriori:
        saveData['D_mah'] = D_mahalanobis
        saveData['D_kde_scott'] = D_kde_scott

    # write Data to file
    if Regressor == 'TBRF':
        saveData['bijML_MF'] = bij_MF.reshape([3,3,testParam['Nx'],testParam['Ny']])
        saveData['bijML_FMF'] = bij_FMF
        saveData['bijML_forest'] = bij_forest
        saveData['g_ML'] = g_MF
        saveData['TB'] = TB_test
        tmp_bijML = bij_FMF
        
        helperFn.writeDataFile2(testParam,trainingParam,Regressor,X_training.shape[0],saveData,folder,suffix='_Split'+str(splitting_features))
        
    else:
        saveData['bijML'] = bij_unfiltered
        saveData['bijML_F'] = bij_filtered
        tmp_bijML = bij_filtered
        helperFn.writeDataFile2(testParam,trainingParam,Regressor,X_training.shape[0],saveData,folder)


