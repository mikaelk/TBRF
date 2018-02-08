#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:41:16 2017
Tensor Basis Decision Tree
Class for the tensor basis decision tree, which learns a model for a given 
Reynolds stress anisotropy tensor by creating a tree structure in which the
output is approximated by a least squares fit on a basis tensor series, see
(Pope, 1975). 

Versions:
    v2: added parallelization
    v3: added sorting and optimization 
    v4: added exceptions in self.createSplit() in case input features are the same
    v5: return base tensor series coefficients
    v6: added possibility to use integer amount of features for splitting
    v7: added saving RMSE values for analyzing feature importance
    v8: fixed a bug when selecting a lower amount of feats/split
@author: Mikael Kaandorp
"""

from __future__ import division
import numpy as np
import time
from functools import partial
from scipy.optimize import minimize_scalar

class TBDT:
    
    def __init__(self,max_levels=400,min_samples_leaf=1,splitting_features='all',
                 regularization=False,regularization_lambda=1e-5,tree_filename='TREE',
                 smoothen_tree=False,verbose=True,optim_split=True,optim_threshold=1000):
        
        self.max_levels = max_levels                        # max amount levels of nodes
        self.min_samples_leaf = min_samples_leaf            # min samples required for leaf nodes
        self.verbose = verbose                              # print more info during splitting process
        self.splitting_features = splitting_features        #
        self.tree_filename = tree_filename
        self.regularization = regularization
        self.regularization_lambda = regularization_lambda
        self.smoothen_tree = smoothen_tree
        self.optim_split = optim_split
        self.optim_threshold = optim_threshold
        
    def fitTensor(self,TT_mat,Tf_mat,T,Y):
        """
        Makes a least square fit on training data Y, by using the preconstructed
        matrices transpose(T)*T and transpose(T)*f. Used in the createSplit() routine.
        Least squares fit is done with respect to scalar coefficients g in the 
        tensor basis series
        b_{ij} = \sum_{i=1}^{10} g^{(i)} T_{i}
        """
        LHS = np.sum(TT_mat,axis=2)
        RHS = np.sum(Tf_mat,axis=1)
        g_hat = np.linalg.lstsq(LHS,RHS)[0]
        bij_hat = np.zeros([9,TT_mat.shape[2]])
        for i in range(TT_mat.shape[1]):
            bij_hat = bij_hat + g_hat[i]*T[:,i,:]
        diff = Y-bij_hat
    
        return g_hat, bij_hat, diff
    
       
    def createSplit(self,X,Y,TB,TT_mat,Tf_mat):  
        """
        Creates a split at a node for given input features X, training output Y,
        Tensor basis TB, and the preconstruced matrices for transpose(T)*T and
        transpose(T)*f. 
        
        Output: splitVar, splitVal, indices_left, indices_right, g_left, g_right
        """
        
        def findJMin_sorted(i1,X,Y,TB,TT_mat,Tf_mat):
            """
            Find optimum splitting point for feature i1. Data is pre-sorted
            to save computational costs (n log n instead of n²)
            """
            
            # flag which activates when all features are the same (e.g. due to
            # feature capping at a certain std)
            Flag_equalFeatures = False
            
            # check if all features are the same
            if np.all(X[0,:] == X[0,0]):
                X_list = np.ndarray.tolist(X.T)
                if all(X_list[0] == X_list[i] for i in range(len(X_list))):
                    Flag_equalFeatures = True
                    
            asort = np.argsort(X[i1,:])
            asort_back = np.argsort(asort)
            
            X_sorted = X[:, asort]
            Y_sorted = Y[:, asort]
            TB_sorted = TB[:,:,asort]
            TT_sorted = TT_mat[:,:,asort]
            Tf_sorted = Tf_mat[:,asort]
            
            results = {'J':1e10, 'splitVar': [], 'splitVal': [], 'indices_left': [],
                      'indices_right': [], 'g_left': [], 'g_right': [], 'MSE_left': [],
                      'MSE_right': [], 'n_left': [], 'n_right': []}
            
            # first exception: only two data-points are left.
            if X_sorted.shape[1] == 2:
                g_l, _ ,diff_l = self.fitTensor(TT_sorted[:,:,:1],Tf_sorted[:,:1],TB_sorted[:,:,:1],Y_sorted[:,:1])
                g_r, _ ,diff_r = self.fitTensor(TT_sorted[:,:,1:],Tf_sorted[:,1:],TB_sorted[:,:,1:],Y_sorted[:,1:])
                diff = np.hstack([diff_l,diff_r])
                J_tmp = (np.mean(np.square(diff)))
                i_left_sorted = np.array([1,0],dtype=bool)
                i_right_sorted = ~i_left_sorted
                
                results['J'] = J_tmp
                results['splitVar'] = i1
                results['splitVal'] = 0.5*(X_sorted[i1,0]+X_sorted[i1,1])
                results['indices_left'] = i_left_sorted[asort_back]
                results['indices_right'] = i_right_sorted[asort_back]
                results['g_left'] = g_l
                results['g_right'] = g_r  
                results['MSE_left'] = np.mean(diff_l**2)
                results['MSE_right'] = np.mean(diff_r**2)
                results['n_left'] = 1
                results['n_right'] = 1
            
            # second exception: all features all equal.
            # g_r is set equal to g_l, i_l is i_r; which will terminate further
            # splitting of the branch in self.fit()
            elif Flag_equalFeatures and X_sorted.shape[1] >2:
                g, _ ,diff = self.fitTensor(TT_sorted[:,:,:],Tf_sorted[:,:],TB_sorted[:,:,:],Y_sorted[:,:])
                J = np.sqrt(np.mean(np.square(diff)))
                
                i_right_sorted = np.ones(X_sorted.shape[1],dtype=bool)
                
                results['J'] = J
                results['splitVar'] = i1
                results['splitVal'] = 0.5*(X_sorted[i1,0]+X_sorted[i1,1])
                results['indices_left'] = i_right_sorted
                results['indices_right'] = i_right_sorted
                results['g_left'] = g
                results['g_right'] = g 
                results['MSE_left'] = 0
                results['MSE_right'] = np.mean(diff**2)
                results['n_left'] = 0
                results['n_right'] = X_sorted.shape[1]
                
            else:    
                for i2 in range(1,X_sorted.shape[1]-1):
                    
                    
                    g_l, _ ,diff_l = self.fitTensor(TT_sorted[:,:,:i2],Tf_sorted[:,:i2],TB_sorted[:,:,:i2],Y_sorted[:,:i2])
                    g_r, _ ,diff_r = self.fitTensor(TT_sorted[:,:,i2:],Tf_sorted[:,i2:],TB_sorted[:,:,i2:],Y_sorted[:,i2:])
                    diff = np.hstack([diff_l,diff_r])
                    J_tmp = (np.mean(np.square(diff)))
                    
                    if J_tmp < results['J']:
                        
                        i_left_sorted = np.zeros(X_sorted.shape[1],dtype=bool)
                        i_left_sorted[:i2] = True
                        
                        i_right_sorted = ~i_left_sorted
                        
                        results['J'] = J_tmp
                        results['splitVar'] = i1
                        results['splitVal'] = 0.5*(X_sorted[i1,i2]+X_sorted[i1,i2-1])
                        results['indices_left'] = i_left_sorted[asort_back]
                        results['indices_right'] = i_right_sorted[asort_back]
                        results['g_left'] = g_l
                        results['g_right'] = g_r  
                        results['MSE_left'] = np.mean(diff_l**2)
                        results['MSE_right'] = np.mean(diff_r**2)
                        results['n_left'] = X_sorted[:,:i2].shape[1]
                        results['n_right'] = X_sorted[:,i2:].shape[1]                    

            return results
        
        def findJMin_opt(i1,X,Y,TB,TT_mat,Tf_mat):
            """
            Find optimum splitting point by using an optimization routine.
            """
            
            # flag which activates when all features are the same (e.g. due to
            # feature capping at a certain std)
            Flag_equalFeatures = False
            
            asort = np.argsort(X[i1,:])
            asort_back = np.argsort(asort)
            
            X_sorted = X[:, asort]
            Y_sorted = Y[:, asort]
            TB_sorted = TB[:,:,asort]
            TT_sorted = TT_mat[:,:,asort]
            Tf_sorted = Tf_mat[:,asort]
            
            def objfn_J(ifloat,Y_sorted,TB_sorted,TT_sorted,Tf_sorted):
                # objective function which minimizes the RMS difference w.r.t. DNS data
                i = int(ifloat)
                g_l, _ ,diff_l = self.fitTensor(TT_sorted[:,:,:i],Tf_sorted[:,:i],TB_sorted[:,:,:i],Y_sorted[:,:i])
                g_r, _ ,diff_r = self.fitTensor(TT_sorted[:,:,i:],Tf_sorted[:,i:],TB_sorted[:,:,i:],Y_sorted[:,i:])
                diff = np.hstack([diff_l,diff_r])
                J = (np.mean(np.square(diff)))
                
                return J
            
            # minimize objective function specified above:
            res = minimize_scalar(objfn_J, args=(Y_sorted,TB_sorted,TT_sorted,Tf_sorted), method='brent',
                          tol=None, bounds=(1,X.shape[1]-1),
                          options={'xtol': 1.e-08,
                                   'maxiter': 200})
            i_split = int(res.x)
            
            #check if all input features are the same. If so, set flag such that
            #no child notes will be created further on
            if np.all(X_sorted[0,:] == X_sorted[0,0]):
                X_list = np.ndarray.tolist(X_sorted.T)
                if all(X_list[0] == X_list[i] for i in range(len(X_list))):
#                    print('all features the same')
                    i_split = 1
                    Flag_equalFeatures = True
                    
            if i_split == 0: # TODO: in case optimization algorithm does not work it returns 0. Needs further testing
                i_split = 1    
            
            # find all relevant parameters for the minimum which was found (maybe this can be improved as it is redundant)
            g_l, _ ,diff_l = self.fitTensor(TT_sorted[:,:,:i_split],Tf_sorted[:,:i_split],TB_sorted[:,:,:i_split],Y_sorted[:,:i_split])
            g_r, _ ,diff_r = self.fitTensor(TT_sorted[:,:,i_split:],Tf_sorted[:,i_split:],TB_sorted[:,:,i_split:],Y_sorted[:,i_split:])
            i_left_sorted = np.zeros(X_sorted.shape[1],dtype=bool)
            i_left_sorted[:i_split] = True
            i_right_sorted = ~i_left_sorted
            
            results = {'J':1e10, 'splitVar': [], 'splitVal': [], 'indices_left': [],
                      'indices_right': [], 'g_left': [], 'g_right': [], 'MSE_left': [],
                      'MSE_right': [], 'n_left': [], 'n_right': []}
            
            results['J'] = res.fun
            results['splitVar'] = i1
            results['splitVal'] = 0.5*(X_sorted[i1,i_split]+X_sorted[i1,i_split-1])
            results['indices_left'] = i_left_sorted[asort_back]
            results['indices_right'] = i_right_sorted[asort_back]
            results['g_left'] = g_l
            results['g_right'] = g_r  
            results['MSE_left'] = np.mean(diff_l**2)
            results['MSE_right'] = np.mean(diff_r**2)
            results['n_left'] = X_sorted[:,:i_split].shape[1]
            results['n_right'] = X_sorted[:,i_split:].shape[1] 
            
            if Flag_equalFeatures:
                # right and left splits are made equal. This leads to termination
                # of the branch later on in self.fit()
                results['g_left'] = g_r
                results['indices_left'] = i_right_sorted[asort_back]
                results['n_left'] = 0
                results['MSE_left'] = 0

            return results
        
        
        
        # select from the available features a subset of features to decide splitting from 
        if self.splitting_features == 'all': # use all available features for each split
            pass
        elif self.splitting_features == 'sqrt': #n_feat normally used for classification, see (Hastie, 2008)
            n_feat = int(np.ceil(np.sqrt(X.shape[0])))
            randomFeat = np.random.choice(np.linspace(0,X.shape[0]-1,X.shape[0],dtype=int),size=n_feat,replace=False)
            X = X[randomFeat,:]
        elif self.splitting_features == 'div3': #n_feat used for regression, see (Hastie, 2008). 
            n_feat = np.max([int(np.ceil(X.shape[0]/3)),5])
            randomFeat = np.random.choice(np.linspace(0,X.shape[0]-1,X.shape[0],dtype=int),size=n_feat,replace=False)
            X = X[randomFeat,:]
        elif isinstance(self.splitting_features, int): #if integer is given, use this value for the amount of randomly selected features
            n_feat = self.splitting_features
            randomFeat = np.random.choice(np.linspace(0,X.shape[0]-1,X.shape[0],dtype=int),size=n_feat,replace=False)
            X = X[randomFeat,:]
        else:
            print('unknown setting for amount of splitting features')
        
        # in case it is enabled, use optimization instead of brute force for large arrays
        if X.shape[1]>self.optim_threshold and self.optim_split:
            # select which splitting function to use with the partial function
            partial_findJMin = partial(findJMin_opt, X=X,Y=Y,TB=TB,TT_mat=TT_mat,Tf_mat=Tf_mat)
        else:    
            partial_findJMin = partial(findJMin_sorted, X=X,Y=Y,TB=TB,TT_mat=TT_mat,Tf_mat=Tf_mat)

        # lists which contain the output of the routine for each splitting feature
        list_J = []
        list_splitVal = []
        list_splitVar = []
        list_indices_l = []
        list_indices_r = []
        list_g_l = []
        list_g_r = []
        list_MSE_l = []
        list_MSE_r = []
        list_n_l = []
        list_n_r = []
        
        
        # go through each splitting feature to select optimum splitting point, 
        # and save the relevant data in lists
        for i in range(X.shape[0]):
            results = partial_findJMin(i)
            
            list_J.append(results['J'])
            list_splitVal.append(results['splitVal'])
            list_splitVar.append(results['splitVar'])
            list_indices_l.append(results['indices_left'])
            list_indices_r.append(results['indices_right'])
            list_g_l.append(results['g_left'])
            list_g_r.append(results['g_right'])
            list_MSE_l.append(results['MSE_left'])
            list_MSE_r.append(results['MSE_right'])
            list_n_l.append(results['n_left'])
            list_n_r.append(results['n_right'])
            
        # find best splitting fitness found for all splitting features, and return
        # relevant parameters
        best = list_J.index(min(list_J))
        
        if self.splitting_features == 'all':
            chosen_splitVar = list_splitVar[best]
        else:
            chosen_splitVar = randomFeat[best]
        
        return (chosen_splitVar, list_splitVal[best], list_indices_l[best], list_indices_r[best], 
                list_g_l[best], list_g_r[best], list_MSE_l[best], list_MSE_r[best], list_n_l[best], list_n_r[best])
        
    def getNodeIndices(self,Node,path):
        """
        The path which the training data follows in the tree is saved in the path
        variable. getNodeIndices() returns the training data indices which are binned
        in a certain node. Nodes are indicated by e.g. [1,0,0] which indicates a path
        in the tree of one right split, and 2 left splits
        
        Output: a boolean array which indicates which training samples are active at Node
        """
        if len(path.shape) == 1:
            return np.array(path == Node)
        else:
            bool_out = np.zeros(path.shape[1],dtype=bool)
            for i in range(path.shape[1]):
                # check if the path for a given training data point corresponds to the
                # given node
                if (Node == path[:,i]).all():
                    bool_out[i] = True
            return bool_out
        
    def predict(self,X,TB,tree):
        """
        Predict the anisotropy given input features X and basis tensors TB. Third 
        input to this function is the tree structure, which contains lists with 
        the splitting variables, values, basis tensor coefficients and nodes
        """
        bij_hat = np.zeros([9,X.shape[1]])
        g_field = np.zeros([10,X.shape[1]])
        
        for i1 in range(X.shape[1]): # go through every datapoint
            
            path = []
            # start at root, add 0 or 1 to input feature path depending on left/right split
            if X[tree['splitVar'][0],i1] <= tree['splitVal'][0]:
                path.append(0)
            else:
                path.append(1)
            
            # while the given path is present in the tree structure, keep binning the data
            while path in tree['path']:
                # return the index of the node in the tree structure:
                index_currNode = int(tree['path'].index(path))

                if X[tree['splitVar'][index_currNode+1],i1] <= tree['splitVal'][index_currNode+1]:
                    path.append(0)
                else:
                    path.append(1)
               
            # while loop ends when path is not present in the tree. Remove last element
            # of the split, and save the last split for prediction
            lastSplit = path[-1:]
            path = path[:-1]

            # get index prediction, which gives the corresponding values for g    
            index_prediction = tree['path'].index(path)
            
            if lastSplit[0] == 0:
                g = tree['g'][(index_prediction+1)*2]
            else:
                g = tree['g'][(index_prediction+1)*2+1]
            
            temp_bij = np.zeros(9)
            for i2 in range(g.shape[0]):
                temp_bij += g[i2]*TB[:,i2,i1]
                g_field[i2,i1] = g[i2]


            # if smoothing is on, propagate the prediction towards the root of the tree
            # still to be properly implemented and tested
            # TODO:
            if self.smoothen_tree:
                print('Warning: smoothing TBDT still to be implemented')
                
                # counter for while loop
                c = 0
                
                while path in tree['path']:
                    
                    index_parent = tree['path'].index(path)
                    
                    if c == 0:
                        n_data_child = self.min_samples_leaf
                    else:
                        n_data_child = tree['n_data'][index_parent+1]
                        
                    g_parent = tree['g'][index_parent]

                    # update the prediction for bij
                    g = (g*n_data_child + 15.0*g_parent)/(n_data_child + 15.0)

                    c += 1
                    lastSplit = path[-1:]
                    path = path[:-1]
#                print(g)    
                temp_bij = np.zeros(9)
                for i2 in range(g.shape[0]):
                    temp_bij += g[i2]*TB[:,i2,i1]
                    
            bij_hat[:,i1] = temp_bij 
        return bij_hat, g_field
            
    
    def readTreeFile(self,file_path):
        """
        Read tree structure from specified file. The tree files are written during
        training of the trees. Output is a tree structure with lists of splitting 
        variables, values, basis tensor coefficients and nodes
        """
        print('Reading tree file: %s' % file_path)
        tree_out = {}
        tree_out['splitVar'] = []
        tree_out['splitVal'] = []
        tree_out['g'] = []
        tree_out['path'] = []
        tree_out['n_data'] = []
        tree_out['MSE'] = [] # mean squared error per split
        tree_out['n'] = [] # amount of data points per split
        
        with open(file_path) as file:
            line_node = []
            line_varVal = []
            line_gl = []
            line_gr = []
            line_data = []
            line_MSEl = []
            line_MSEr = []
            line_nl = []
            line_nr = []
            
            for i,line in enumerate(file):  
                
                if 'Node' in line:
                    line_node = i+1
                    line_varVal = i+5
                    line_gl = i+7
                    line_gr = i+9
                    line_data = i+3
                    line_MSEl = i + 11
                    line_MSEr = i + 13
                    line_nl = i + 15
                    line_nr = i + 17
                
                if i == 2: # TB LLS MSE
                    MSE = float(line)
                    tree_out['MSE'].append(MSE)
                    
                if i == 6:
                    dataList = line.split('(')[1].split(')')[0]
                    dataList = dataList.split(',')
                    n_data = int(dataList[1])
                    tree_out['n'].append(n_data)
                    
                if i == line_node:
                    # skip root node, as this is indicated by '[]'
                    if 'Root' in line:
                        pass
                    else:
                        # get string with node indices
                        intList = line.split('[')[1].split(']')[0]
                        intList = intList.split(' ')[:-1]
                        Node = []
                        for i1 in range(len(intList)):
                            Node.append(int(intList[i1]))
                        tree_out['path'].append(Node)
                
                if i == line_data:
                    dataList = line.split('(')[1].split(')')[0]
                    dataList = dataList.split(',')
                    n_var = int(dataList[0])
                    n_data = int(dataList[1])
                    tree_out['n_data'].append(n_data)
                
                if i == line_varVal:
                    varValList = line.split('[')[1].split(']')[0]
                    varValList = varValList.split(',')
                    splitVar = int(varValList[0])
                    splitVal = float(varValList[1])
                    tree_out['splitVar'].append(splitVar)
                    tree_out['splitVal'].append(splitVal)
                    
                if i == line_gl:
                    glList = line.split('[')[1].split(']')[0]
                    glList = glList.split(' ')[:-1]
                    gl = []
                    for i1 in range(len(glList)):
                        gl.append(np.float(glList[i1]))
                    gl = np.array(gl)
                    tree_out['g'].append(gl)
                    
                        
                if i == line_gr:
                    grList = line.split('[')[1].split(']')[0]
                    grList = grList.split(' ')[:-1]
                    gr = []
                    for i1 in range(len(grList)):
                        gr.append(np.float(grList[i1]))
                    gr = np.array(gr)   
                    tree_out['g'].append(gr)
                
                if i == line_MSEl:
                    tree_out['MSE'].append(float(line))
                if i == line_MSEr:
                    tree_out['MSE'].append(float(line))
                if i == line_nl:
                    tree_out['n'].append(int(line))
                if i == line_nr:
                    tree_out['n'].append(int(line))
                    
        return tree_out
          
    def fit(self,X,Y,TB):
        """
        Fit tensor basis decision tree
        Input:  X; input features, np.array([N_features, N_gridPoints])
                Y; bij from DNS on which to fit tree, np.array([9, N_gridPoints])
                TB; basis tensors at each gridpoint np.array([9, 10, N_gridPoints])
        Output: tree; dict which contains lists of nodes (path), basis tensor coefficients (g),
                splitting variables and values
                bij_hat; fitted values for bij
        """
   
        TreeName = self.tree_filename
        
        tStart_fitting = time.time()
        
        f = open(TreeName, 'w')
        
        levels_max = self.max_levels
          
        # preconstruct the N_obs matrices for the LHS and RHS terms in the least squares problem
        TT_mat = np.zeros([TB.shape[1],TB.shape[1],TB.shape[2]])
        Tf_mat = np.zeros([TB.shape[1],TB.shape[2]])
        for i1 in range(X.shape[1]):
            TT_mat[:,:,i1] = np.dot(np.transpose(TB[:,:,i1]),TB[:,:,i1])
            if self.regularization:
                TT_mat[:,:,i1] = TT_mat[:,:,i1]+self.regularization_lambda*np.eye(TB.shape[1])
            Tf_mat[:,i1] = np.dot(np.transpose(TB[:,:,i1]),Y[:,i1])
        
        # create tree dict, which contains the nodes and corresponding values which are 
        # necessary for predictions later
        # path:         all nodes which are created. 0 indicates a left path,
        #               1 a right path. 2 indicates the feature ended in a terminal node
        # g:            all least squares coefficients at each node
        # splitVar:     the variable used for splitting
        # splitVal:     the value ""
        # N_data: total amount of datapoints used for training
        # n_data: amount of data points in each node
        
        tree = {}
        tree['path'] = []
        tree['g'] = []
        tree['splitVar'] = []
        tree['splitVal'] = []
        tree['N_data'] = TT_mat.shape[2]
        tree['n_data'] = []
        tree['MSE'] = []
        tree['n'] = []
        
        # queue which contains the child nodes which need to be resolved in the next i-iteration
        tmpQueue = [] #temp variable which stores child nodes
        Queue = []  #Queue is set to tmpQueue at next iteration (i.e. child nodes are now current nodes)
        
        for i in range(levels_max):
            
            if i == 0: #TODO: merge creation of root node (i=0) and further nodes 
                f.write('----------------Start building tree: Root node (level 0)--------------------\n')
                
                if self.verbose:
                    print('------------Building root node-----------------')
                
                start = time.time()
                
                g, _ ,diff = self.fitTensor(TT_mat[:,:,:],Tf_mat[:,:],TB[:,:,:],Y[:,:])
                tree['MSE'].append(np.mean(diff**2))
                tree['n'].append(X.shape[1])
                
                f.write('MSE:\n%f\n'% np.mean(diff**2))
                
                #root node: initialization and first split
                splitVar,splitVal,i_left,i_right,g_l,g_r,MSE_l,MSE_r,n_l,n_r = self.createSplit(X,Y,TB,TT_mat,Tf_mat)
                # path: variable which contains the path of the nodes which the training features follow
                path = np.array(i_right*1)
                
                # add all necessary information to the tree dict
                tree['g'].append(g_l)
                tree['g'].append(g_r)
                tree['splitVar'].append(splitVar)
                tree['splitVal'].append(splitVal)
                tree['n_data'].append(X.shape[1])
                tree['MSE'].append(MSE_l)
                tree['MSE'].append(MSE_r)
                tree['n'].append(n_l)
                tree['n'].append(n_r)
                
                # check if child node is a terminal node, otherwise add to queue
                # minimum samples leaf reached (standard one) -> no more splitting possible:
                if X[:,i_left].shape[1] == self.min_samples_leaf:
                    pass
                # empty node should not happen, print error and abort
                elif X[:,i_left].shape[1] == 0:
                    print('error: indices left empty')
                    break
                # left/right bin indices are the same, can happen when input features 
                # are equal with only two features to choose from
                elif all(i_right == i_left):
                    pass
                # otherwise: child node is not a terminal node, add child node
                # to queue for further splitting
                else:
                    tmpQueue.append([0])
                    tree['path'].append([0])
                    
                if X[:,i_right].shape[1] == self.min_samples_leaf:
                    pass
                elif X[:,i_right].shape[1] == 0:
                    print('error: indices right empty')
                    break
                elif all(i_right == i_left):
                    pass
                else:
                    tmpQueue.append([1])
                    tree['path'].append([1])
                    
                end = time.time()
                print('Root node constructed in %f seconds' % (end-start))
                
                # write information necessary to reconstruct tree
                f.write('Node: \n[Root]\n')
                f.write('Data shape to be split: \n' + str(X.shape) + '\n')
                f.write('Chosen splitting var and val: \n' + str([splitVar,splitVal]) + '\n')
                f.write('g_left: \n[')
                np.savetxt(f,g_l,fmt='%f',newline=' ', delimiter=',')
                f.write('] \ng_right: \n[')
                np.savetxt(f,g_r,fmt='%f',newline=' ', delimiter=',')
                f.write('] \n')
                
                f.write('MSE_left: \n%.9f\n' % MSE_l)
                f.write('MSE_right: \n%.9f\n'% MSE_r)
                f.write('n_left: \n%i\n'% n_l)
                f.write('n_right: \n%i\n'% n_r)
               
                f.write('\n')        
                f.write('-----------------Root node constructed-----------------------------\n')
                
            else:
                
                if Queue:
                    f.write('-----------------building tree: level %i------------------------\n' % i)
                
                if self.verbose and Queue:
                    print('-------------Building tree level %i-------------------' % i)
                    print('Amount of nodes: %i' % (len(Queue)))
                
                start = time.time()
                
                # new path variables for each node will be added to tmp_path.
                # after going though each node, path will be set to tmp_path.
                # a '2' in 'path' indicates the data is fully split
                tmp_path = np.vstack([path,2*np.ones([1,tree['N_data']])])
                
                # go through nodes
                for i1 in range(len(Queue)):
                    
                    # get current node
                    Node = np.array(Queue[i1])
                       
                    # get a boolean array with training data points corresponding to 
                    # the current node. Maybe a nicer solution can be found for this
                    indices = self.getNodeIndices(Node,path)
         
                    # write important information to tree file
                    f.write('Node: \n[')
                    np.savetxt(f,Node,fmt='%i',newline=' ', delimiter=',')
                    f.write('] \n')
                    f.write('Data shape to be split: \n' + str(X[:,indices].shape) + '\n')
                    
                    # split data into left and right bin
                    splitVar,splitVal,i_left,i_right,g_l,g_r,MSE_l,MSE_r,n_l,n_r = self.createSplit(X[:,indices],Y[:,indices],
                                                                                   TB[:,:,indices],TT_mat[:,:,indices],Tf_mat[:,indices])
                    
                    # write to tree file
                    f.write('Chosen splitting var and val: \n' + str([splitVar,splitVal]) + '\n')
                    f.write('g_left: \n[')
                    np.savetxt(f,g_l,fmt='%f',newline=' ', delimiter=',')
                    f.write('] \ng_right: \n[')
                    np.savetxt(f,g_r,fmt='%f',newline=' ', delimiter=',')
                    f.write('] \n')

                    f.write('MSE_left: \n%.9f\n' % MSE_l)
                    f.write('MSE_right: \n%.9f\n'% MSE_r)
                    f.write('n_left: \n%i\n'% n_l)
                    f.write('n_right: \n%i\n'% n_r)      
                    f.write('\n')    
                    
                    # add left and right split to tree structure
                    tree['g'].append(g_l)
                    tree['g'].append(g_r)
                    tree['splitVar'].append(splitVar)
                    tree['splitVal'].append(splitVal)
                    tree['n_data'].append(X[:,indices].shape[1])
                    tree['MSE'].append(MSE_l)
                    tree['MSE'].append(MSE_r)
                    tree['n'].append(n_l)
                    tree['n'].append(n_r)               
                    
                    # check whether the left and right splits are terminal nodes,
                    # and add child nodes to queue
                    # one datapoint -> no more splitting possible:
                    if X[:,indices][:,i_left].shape[1] <= self.min_samples_leaf:
                        pass
                    # left/right bin indices are the same, can happen when input features 
                    # are equal
                    elif all(i_right == i_left):
                        pass
                    # empty node should not happen, just in case for debugging:
                    elif X[:,indices][:,i_left].shape[1] == 0:
                        print('error: indices left empty')
                        print(X[:,indices])
                        break
                    # otherwise, create child node and add to queue and tree structure
                    else:
                        tmpQueue.append(Queue[i1]+[0])
                        tree['path'].append(Queue[i1]+[0])
                    

                    if X[:,indices][:,i_right].shape[1] <= self.min_samples_leaf:
                        pass
                    elif all(i_right == i_left):
                        pass              
                    elif X[:,indices][:,i_right].shape[1] == 0:
                        print('error: indices right empty')
                        break
                    else:
                        tmpQueue.append(Queue[i1]+[1])
                        tree['path'].append(Queue[i1]+[1])
                                   
                    tmp_path[i,indices] = i_right
                            
                #update the paths of all the training variables        
                path=tmp_path
                end = time.time()
                
                if self.verbose and Queue:
                    print('Tree level %i constructed in %f seconds' % (i,end-start))
                
            # add child nodes to current queue        
            Queue = tmpQueue
            tmpQueue = []
        
        tEnd_fitting = time.time()
        tTotal_fitting = tEnd_fitting - tStart_fitting
        f.write('\n\n Total time training tree: ' + str(tTotal_fitting) + ' seconds \n')
        f.close()
        
        return tree