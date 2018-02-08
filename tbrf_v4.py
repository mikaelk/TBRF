#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:17:49 2017
Tensor Basis Random Forest
@author: mikael
v2: added basis tensor series scalar coefficient fields as output
v3: added RMSE per split (featImportance)
v4: include tbdt_8 in which random splitting feature is fixed
"""
import numpy as np
from tbdt_v8 import TBDT
from os.path import isfile

class TBRF:
    
    def __init__(self,n_trees=10,bootstrap_tree=True,read_from_file=True,max_levels=400,
                 min_samples_leaf=1,splitting_features='all', regularization=True,
                 regularization_lambda=0.00001,tree_filename='TREE_RF_%i',verbose=True,
                 optim_split=True,optim_threshold=1000):
    
        # properties specific for the tensor basis random forest
        self.n_trees = n_trees
        self.bootstrap_tree = bootstrap_tree
        self.read_from_file = read_from_file
           
        # properties inherited from TBDT
        self.max_levels = max_levels
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.splitting_features = splitting_features
        self.tree_filename = tree_filename
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        self.optim_split = optim_split
        self.optim_threshold = optim_threshold

    def randomSampling(self,X,Y,TB,fraction=1,replace=True):
        """
        Take random samples with or without replacement from data, 
        N_samples = fraction*length(array)
        """
        
        size_out = np.round(fraction*X.shape[1])
        #samples from the columns:
        idx = np.random.choice(X.shape[1],int(size_out),replace=replace)
        
        X_out = X[:,idx]
        Y_out = Y[:,idx]
        TB_out = TB[:,:,idx]
    
        return X_out,Y_out,TB_out

    def fit(self,X,Y,TB):
        """
        Fit a Tensor Basis Random Forest
        Given input features X, true response Y, and tensor basis TB, create
        a Random Forest structure
        Input:  X; input features, np.array([N_features, N_datapoints])
                Y; bij from DNS on which to fit tree, np.array([9, N_datapoints])
                TB; basis tensors at each gridpoint np.array([9, 10, N_datapoints])
        Output: forest, which contains dicts for all the different trees
        """
        forest = {}
        
        for i in range(self.n_trees):
            
            # print progress
            if self.verbose:
                print('-----------------------------------------------')
                print('----------------TREE NR. %i---------------------' % (i+1))
                print('-----------------------------------------------')
            
            # resample data in case bootstrapping is true
            if self.bootstrap_tree == True:
                X_sampled,Y_sampled,TB_sampled = self.randomSampling(X,Y,TB)
            else:
                X_sampled,Y_sampled,TB_sampled = X,Y,TB
            
            tree_filename = (self.tree_filename % i)
            
            # create decesion tree class 
            tbdt = TBDT(max_levels=self.max_levels ,min_samples_leaf=self.min_samples_leaf,
                        regularization=self.regularization,regularization_lambda=self.regularization_lambda,
                        splitting_features=self.splitting_features,tree_filename=tree_filename,
                        verbose=self.verbose,optim_split=self.optim_split,optim_threshold=self.optim_threshold)
            
            # read tree files if present and turned on
            if self.read_from_file and isfile(tree_filename):
                tree = tbdt.readTreeFile(tree_filename)
            else:
                tree = tbdt.fit(X_sampled,Y_sampled,TB_sampled)
            
            # add the tree structure to the forest
            forest[i] = tree
            
            # TODO: add out-of-box validation score
            
        return forest
            
    
    def predict(self,X_test,TB_test,forest):
        """
        Tensor Basis Random Forest predictions
        Given input features X_test and tensor basis TB_test, and a forest 
        structure resulting from TBRF.fit, make predictions for the anisotropy
        tensor b_ij
        Input:  X_test; input features, np.array([N_features, N_datapoints])
                TB_test; basis tensors at each gridpoint np.array([9, 10, N_datapoints])
                forest; list with individual trees, array(len(N_trees))
        Output: bij_hat: np.array([9,N_datapoints])
                bij_forest: np.array([9,N_datapoints, N_trees])
        
        """
        print('Predicting b_ij, TBRF_v4')
        # initialize predictions
        bij_forest = np.zeros([9,X_test.shape[1],len(forest)])
        g_forest = np.zeros([10,X_test.shape[1],len(forest)])
        
        # go through trees of the random forest to make predictions
        for i in range(len(forest)):
            # TODO: it should be possible to remove the following lines by using self.forest for example
            # but its working so I'm leaving it like this for now
            tree_filename = (self.tree_filename % i)
            tbdt = TBDT(max_levels=self.max_levels ,min_samples_leaf=self.min_samples_leaf,
                        regularization=self.regularization,regularization_lambda=self.regularization_lambda,
                        splitting_features=self.splitting_features,tree_filename=tree_filename,
                        verbose=self.verbose,optim_split=self.optim_split,optim_threshold=self.optim_threshold)
            bij_forest[:,:,i], g_forest[:,:,i] = tbdt.predict(X_test,TB_test,forest[i])
            print('Prediction %i' % (i+1))
        # take the mean of the tree predictions
        bij_hat = np.mean(bij_forest,axis=2)
        
        # return the TBRF prediction and TBDT predictions
        return bij_hat,bij_forest,g_forest
    