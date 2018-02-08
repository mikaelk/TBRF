#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:48:46 2017

        
@author: Mikael Kaandorp
"""

from __future__ import division
import numpy as np
from importAndInterpolate_v6 import fn_importAndInterpolate
#from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import PyFOAM as pyfoam
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import time
#import itertools
from scipy import ndimage
from os.path import isfile
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer, Merge, Dot, Input, Reshape, Lambda
from keras.layers import dot, multiply
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

def plotFeatures(features, Nx, Ny, meshRANS,Re,trainTest,index_x=1,index_y=2):
    n_feat = features.shape[0]
    inv = features.reshape([n_feat,Nx,Ny])
    indices = np.linspace(0,n_feat-1,n_feat,dtype='int')
    
    n_rows = int(np.ceil(np.float(n_feat)/np.float(3)))
    
    f, axarr = plt.subplots(n_rows, 3, sharex='col', sharey='row')
    for i in range(indices.shape[0]):
        row = int(indices[i]/3)
        col = int(indices[i]%3)
        index = indices[i]
        
#        maxval = np.max([np.abs(inv[index,:,:])])
        maxval = 2*np.std([np.abs(inv[index,:,:])])
        if maxval < 1e-12:
            maxval = 1e-12
        contour_levels = np.linspace(-1.05*maxval, 1.05*maxval, 50)
        cmap=plt.cm.coolwarm
        cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
        cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])
    
        contPlot = axarr[row,col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], inv[index,:,:],contour_levels,cmap=cmap,extend="both")
        div = make_axes_locatable(axarr[row,col])
        cax = div.append_axes("right", size="20%", pad=0.05)
        cax.set_visible(False)
        axarr[row,col].set_title('Feature %i' % (index+1), fontsize=10)
        cbar = plt.colorbar(contPlot)
        cbar.ax.tick_params(labelsize=8) 
        plt.suptitle('%s data, Re = %i' % (trainTest, Re))
    plt.show()
    
def filterField(inputData, std):
    #Filter a field (e.g. predicted a_ij) using a gaussian filter
    if len(inputData.shape) == 4:
        outputData = np.zeros(inputData.shape)
        for i1 in range(inputData.shape[0]):
            for i2 in range(inputData.shape[1]):
                outputData[i1,i2,:,:] = ndimage.gaussian_filter(inputData[i1,i2,:,:], std, order=0, output=None, mode='nearest', cval=0.0, truncate=4.0)
    #TODO: non mesh format
    
    return outputData

def normalizeTrainingFeatures(X_training,cap):
    # normalize training features, output the normalized features and mu/std original data
    # which will be used to normalize the test features
    std_inv = np.zeros(X_training.shape[3])
    mu_inv = np.zeros(X_training.shape[3]) 
    for i1 in range(X_training.shape[3]):
        # process consists of two parts: normalizing and removing outliers,
        # and normalizing and keeping the std/mean for later processing
        std_temp = np.std(X_training[:,:,:,i1])
        mu_temp = np.mean(X_training[:,:,:,i1])
        
        #normalize
        X_training[:,:,:,i1] = (X_training[:,:,:,i1] - mu_temp) / std_temp
        
        #cap
        X_training[:,:,:,i1][X_training[:,:,:,i1]> cap] = cap
        X_training[:,:,:,i1][X_training[:,:,:,i1]< -cap] = -cap
        
        #de-normalize
        X_training[:,:,:,i1] = (X_training[:,:,:,i1]*std_temp) + mu_temp
        
        # data with outliers removed: again get the mean and std
        std_inv[i1] = np.std(X_training[:,:,:,i1])
        mu_inv[i1] = np.mean(X_training[:,:,:,i1])
        
        #normalize the data with no outliers
        X_training[:,:,:,i1] = (X_training[:,:,:,i1] - mu_inv[i1]) / std_inv[i1]
    return X_training, std_inv, mu_inv
        
def normalizeTestFeatures(X_test, std_training, mu_training, Cap_inv): # 
    # normalize test features according to the std/mu from the training features + remove outliers
    for i1 in range(X_test.shape[2]): #rescale invariants according to training data
        X_test[:,:,i1] = (X_test[:,:,i1]-mu_training[i1])/std_training[i1]
        X_test[:,:,i1][X_test[:,:,i1]>Cap_inv] = Cap_inv
        X_test[:,:,i1][X_test[:,:,i1]<-Cap_inv] = -Cap_inv
    return X_test

def randomSampling(X,Y,TB,fraction,replace):
    # take random samples with replacement from training data, 
    # N_samples = fraction*length(array)
    
    size_out = np.round(fraction*X.shape[1])
    #samples from the columns:
    idx = np.random.choice(X.shape[1],int(size_out),replace=replace)
    
    X_out = X[:,idx]
    Y_out = Y[:,idx]
    TB_out = TB[:,:,idx]
    
    return X_out,Y_out,TB_out


home            = './turbulenceData/'
# plot inline
get_ipython().magic(u'matplotlib inline')


Scale_SR            = 1 #scale the strain rate / rotation rate tensors 'none', 'ke', 'norm'
Scale_TB            = 1 #scale the tensor basis according to Ling et al.
Scale_inv           = 1 #scale the invariants to std 1 and mean 0 
Cap_inv             = 5 # filter the invariants for std>Cap, std<-Cap
reload_data         = 0
SecondaryFeatures = ['k'] #whether to use more features than just S and R: use 'k' for grad(k) features, and 'U' for velocity magnitude

n_feat = 11
DeleteFeatures = [0,3,5,12,13,14,15,17]
np.random.seed(12345)


if reload_data:
    trainingParam   = {} #dict with all training parameters (lists)
    testParam       = {} #dict with all test parameters (one parameters per dict)


trainingParam['Re']         = [3200,2900,2600,2400]
trainingParam['turbModel']  = ['kOmega','kOmega','kOmega','kOmega']
trainingParam['flowCase']   = ['SquareDuct','SquareDuct','SquareDuct','SquareDuct']
trainingParam['time_end']   = [50000,50000,50000,50000]
trainingParam['Nx']         = [50,50,50,50]
trainingParam['Ny']         = [50,50,50,50]
trainingParam['frac']       = 1                     # how much of the total available training data should be used for training
trainingParam['replace']    = False                 # choose whether the used training data is obtained with/without replacement from available data

testParam['Re']             = 3500
testParam['turbModel']      = 'kOmega'
testParam['flowCase']       = 'SquareDuct'
testParam['time_end']       = 50000
testParam['Nx']             = 50
testParam['Ny']             = 50

# indices for x and y axes when plotting
if testParam['flowCase'] == 'PeriodicHills':
    index_x = 0
    index_y = 1
elif testParam['flowCase'] == 'SquareDuct':
    index_x = 1
    index_y = 2
    
#%% loading data
if reload_data == True: #reload data; can be set to false when quickly evaluating ML fit
    
    X_training = np.zeros([len(trainingParam['Re']),trainingParam['Nx'][0],trainingParam['Ny'][0],n_feat])
    Y_training = np.zeros([len(trainingParam['Re']),trainingParam['Nx'][0],trainingParam['Ny'][0],9])
    TB_training = np.zeros([len(trainingParam['Re']),trainingParam['Nx'][0],trainingParam['Ny'][0],10,9])
    
    # -------------------------------------get training data-----------------------
    for i1 in range(len(trainingParam['Re'])):
        dataRANS_training,dataDNS_training,meshRANS = fn_importAndInterpolate(home,trainingParam['flowCase'][i1],
                                                                              trainingParam['Re'][i1],trainingParam['turbModel'][i1],
                                                                              trainingParam['time_end'][i1],trainingParam['Nx'][i1],
                                                                              trainingParam['Ny'][i1],0,0,SecondaryFeatures)
        dataRANS_training['S'],dataRANS_training['R'] = pyfoam.getSRTensors(dataRANS_training['gradU'], 
                         Scale_SR,dataRANS_training['k'],dataRANS_training['epsilon'])
        
        #check if more features are required, e.g. grad(k)
        if 'k' in SecondaryFeatures:
            dataRANS_training['A_k'] = pyfoam.getTkeFeatures(dataRANS_training['gradTke'],Scale_SR,dataRANS_training['k'],dataRANS_training['epsilon'])
            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R'],dataRANS_training['A_k']])
#            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R']])
        else:
            dataRANS_training['invariants'] = pyfoam.getInvariants([dataRANS_training['S'],dataRANS_training['R']])
        
        #get DNS eigenvectors/eigenvalues (stored in diagonal matrix):
        dataDNS_training['eigValMat'],dataDNS_training['eigVecMat'] = pyfoam.eigenDecomposition(dataDNS_training['bij'])
        #get eigenvector orientation (radians), put in training data:
        dataDNS_training['phi'] = pyfoam.eigenvectorToEuler(dataDNS_training['eigVecMat'])
        dataRANS_training['tb'] = pyfoam.getTensorBasis(dataRANS_training['S'],dataRANS_training['R'],Scale_TB)
        
        #----------------- reshape data and stack different training cases
        temp_X_training = np.reshape(dataRANS_training['invariants'],[np.shape(dataRANS_training['invariants'])[0],trainingParam['Nx'][i1]*trainingParam['Ny'][i1]])
        
        #Delete specified features if specified
        if DeleteFeatures:
            temp_X_training = np.delete(temp_X_training,DeleteFeatures,axis=0)
        # add velocity to features if necessary
        if 'U' in SecondaryFeatures:
            temp_X_training = np.vstack([temp_X_training,np.reshape(np.linalg.norm(dataRANS_training['U'],axis=0),[1,trainingParam['Nx'][i1]*trainingParam['Ny'][i1]])])
        temp_Y_training = np.reshape(dataDNS_training['bij'],[9,trainingParam['Nx'][i1]*trainingParam['Ny'][i1]])
        temp_TB_training = np.reshape(dataRANS_training['tb'],[9,dataRANS_training['tb'].shape[2],dataRANS_training['tb'].shape[3]*dataRANS_training['tb'].shape[4]])
        
        # plot input features for verification
        plotFeatures(temp_X_training, trainingParam['Nx'][i1], trainingParam['Ny'][i1],meshRANS,trainingParam['Re'][i1],'Training',index_x,index_y)
        
        #stack training data
        X_training[i1,:,:,:] = np.transpose(temp_X_training).reshape([50,50,n_feat])
        Y_training[i1,:,:,:] = np.transpose(temp_Y_training).reshape([50,50,9])
        TB_training[i1,:,:,:,:] = np.transpose(temp_TB_training).reshape([50,50,10,9])
    
    # normalize input features to std 1 and mu 0, remove outliers above |std| > Cap_inv
    X_training, trainingParam['std'], trainingParam['mu'] = normalizeTrainingFeatures(X_training,Cap_inv) 
    
    
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
            
    # Get from the test data bij, and decompose it so that the eigenvalues
    # and eigenvectors can be used for the predictions:
    dataRANS_test['eigValMat'],dataRANS_test['eigVecMat'] = pyfoam.eigenDecomposition(dataRANS_test['bij'])
    dataDNS_test['eigValMat'],dataDNS_test['eigVecMat'] = pyfoam.eigenDecomposition(dataDNS_test['bij'])
    dataRANS_test['tb'] = pyfoam.getTensorBasis(dataRANS_test['S'],dataRANS_test['R'],Scale_TB)
    
    X_test = np.reshape(dataRANS_test['invariants'],[np.shape(dataRANS_test['invariants'])[0],testParam['Nx']*testParam['Ny']])
    
    # remove specified input features
    if DeleteFeatures:
        X_test = np.delete(X_test,DeleteFeatures,axis=0)
    # add velocity to features if necessary
    if 'U' in SecondaryFeatures:
        X_test = np.vstack([X_test,np.reshape(np.linalg.norm(dataRANS_test['U'],axis=0),[1,testParam['Nx']*testParam['Ny']])])
    Y_test = np.reshape(dataDNS_test['bij'],[9,testParam['Nx']*testParam['Ny']])
    TB_test = np.reshape(dataRANS_test['tb'],[9,dataRANS_test['tb'].shape[2],dataRANS_test['tb'].shape[3]*dataRANS_test['tb'].shape[4]])
    TB_test = TB_test.T.reshape([50,50,10,9])
    
    # plot input features for verification
    plotFeatures(X_test, testParam['Nx'], testParam['Ny'],meshRANS,testParam['Re'],'Test',index_x,index_y)

    X_test = np.transpose(X_test).reshape([50,50,n_feat])
    Y_test = np.transpose(Y_test).reshape([50,50,9])
    # normalize test features according to std and mu from training data, remove outliers |std| > Cap_inv
    X_test = normalizeTestFeatures(X_test, trainingParam['std'],trainingParam['mu'], Cap_inv)
#    x_test = np.transpose(X_test)

#%% main part

def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='linear')) #activation: tanh sigmoid
    model.add(Dense(30, kernel_initializer='normal', activation='tanh',bias=True)) 
    model.add(Dense(30, kernel_initializer='normal', activation='tanh',bias=True)) 
    model.add(Dense(9, kernel_initializer='normal', activation='linear'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def convolutional_model(input_shape):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(9, 9), padding='same', activation='relu'))
#    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(11, 11), padding='same', activation='relu', input_shape=input_shape))
    
    model.add(Conv2D(9, kernel_size=(15, 15), padding='same', activation='linear'))
    optimizer = keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.000005)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
    return model
    

def convolutional_model2(input_shape):
    #convolutional neural network which maps input features to components of the anisotropy tensor, works
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
#    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(9, 9), padding='same', activation='relu'))
#    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(11, 11), padding='same', activation='relu'))
    
    model.add(Conv2D(9, kernel_size=(15, 15), padding='same', activation='linear'))
    optimizer = keras.optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.000005)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])
    return model

def tensorBasisCNN(inputShape_scalar,inputShape_tensor):
    
    scalarBranch = Sequential()
    scalarBranch.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=inputShape_scalar))
    scalarBranch.add(Conv2D(32, kernel_size=(9, 9), padding='same', activation='relu'))
    scalarBranch.add(Conv2D(64, kernel_size=(11, 11), padding='same', activation='relu'))
    scalarBranch.add(Conv2D(1, kernel_size=(15, 15), padding='same', activation='linear'))    
    
    tensorBranch = Sequential()
    tensorBranch.add(InputLayer(input_shape=inputShape_tensor))
    
    mainBranch = Sequential()
    mainBranch.add(dot(inputs=[scalarBranch, tensorBranch], axes=2))
    
    optimizer = keras.optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.000005)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])

    return mainBranch

def tensorBasisCNN2(inputShape_scalar,inputShape_tensor):
    # attempt to combine convolutional_model2 together with a tensor input layer.
    # is not working yet; a solution needs to be found to merge the output of 
    # convolutional_model2 with the tensor input layer. Possibly using the keras
    # batch_dot function

    mainInput = Input(shape=inputShape_scalar)
    scalarOutput = Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu')(mainInput)
    scalarOutput = Conv2D(1, kernel_size=(15, 15), padding='same', activation='linear')(scalarOutput)
    
    tensorInput = Input(shape=inputShape_tensor)
    
    
    merged = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1,2)))([tensorInput,scalarOutput])
#    merged = multiply(inputs=[x, tensorInput])
#    merged = Reshape(inputShape_scalar)(merged)
    
    model = Model(inputs=[mainInput,tensorInput], outputs=merged)
    
    optimizer = keras.optimizers.Adam(lr=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.000005)
    model.compile(loss='mean_squared_error', optimizer=optimizer,metrics=['accuracy'])

    return model


# TODO: scale images
n_imgs = len(trainingParam['Re'])

# normal input, i.e. features in the flow domain
input_shape = (50, 50, n_feat)

#tensor basis input
input_shape_TB = (50, 50, 9)


#model = tensorBasisCNN2(input_shape,input_shape_TB)
model = convolutional_model2(input_shape)

    
#model.fit([X_training,TB_training[:,:,:,0,:]],Y_training,epochs=400,batch_size=10)
model.fit(X_training,Y_training,epochs=500,batch_size=5)


y_predict = model.predict(X_test.reshape([1,50,50,n_feat]))




#%% Plotting
bij = np.reshape(np.transpose(Y_test,axes=(2,0,1)),[3,3,trainingParam['Nx'][0],trainingParam['Ny'][0]])

#bij_hat2 = np.reshape(bij_tree[:,:],[3,3,trainingParam['Nx'][0],trainingParam['Ny'][0]])
y_predict = np.transpose(y_predict[0],axes=(2,0,1))
bij_hat = np.reshape(y_predict,[3,3,trainingParam['Nx'][0],trainingParam['Ny'][0]])
#bij_hat = np.reshape(np.transpose(y_predict),[3,3,trainingParam['Nx'][0],trainingParam['Ny'][0]])

bij_hat2= filterField(bij_hat,[3,3])

plt_indices = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]])
index_x = 1
index_y = 2
for i_plt in range(plt_indices.shape[0]):
    index = plt_indices[i_plt]

    maxval = np.max([np.abs(bij[index[0],index[1],:,:])])
    contour_levels = np.linspace(-1.05*maxval, 1.05*maxval, 50)
    cmap=plt.cm.coolwarm
    cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
    cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])

    f, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex='col')
    subPlot1 = ax1.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], bij[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    subPlot2 = ax2.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], bij_hat[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    subPlot3 = ax3.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], bij_hat2[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
#    subPlot4 = ax4.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], bij_hat2[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
    ax1.set_title('DNS $b_{%i%i}$' % (index[0]+1, index[1]+1)) #    axarr[0].set_title('RANS $R_{%i%i}$' % (index[0]+1, index[1]+1))
    ax2.set_title('CNN $b_{%i%i}$ prediction' % (index[0]+1, index[1]+1))
    ax3.set_title('CNN $b_{%i%i}$ prediction, filtered' % (index[0]+1, index[1]+1))
    ax4.axis('off')
#    ax4.set_title('TBRF $b_{%i%i}$ prediction, filtered' % (index[0]+1, index[1]+1))
    f.subplots_adjust(right=0.8)  
    f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
    plt.show()

    
# plot weights filters first layer
weights_0 = model.get_weights()[0]
n_cols = min(weights_0.shape[3],20)
n_rows = min(weights_0.shape[2],20)

f, axarr = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')

cmap=plt.cm.coolwarm
cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])

for i in range(n_rows):
    for j in range(n_cols):
        
        maxval = np.max(weights_0[:,:,i,j])
        minval = np.min(weights_0[:,:,i,j])
        xvals = np.linspace(0,weights_0.shape[0]-1,weights_0.shape[0])
        yvals = np.linspace(0,weights_0.shape[1]-1,weights_0.shape[1])
        x,y = np.meshgrid(xvals,yvals)
        
        if maxval-minval < 1e-14:
            maxval = maxval + 1e-14
            minval = minval - 1e-14
        contour_levels = np.linspace(minval, maxval, 50)
        contPlot = axarr[i,j].contourf(x, y, weights_0[:,:,i,j],contour_levels,cmap=cmap,extend="both")

plt.show()       



