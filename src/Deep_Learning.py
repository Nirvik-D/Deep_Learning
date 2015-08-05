'''
Created on Jan 5, 2015

@author: Lightning
'''
import numpy as np
from theano import *
import theano.tensor as T

import cPickle, gzip

# Load the dataset
#filename = 'X:\Deep_Learning\mnist.pkl.gz'
#f = gzip.open(filename, 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()

#----------function definition for shared variable---------    
def shared_dataset(data_xy):
    data_x, data_y = data_xy
    configX = theano.config.floatX
    shared_x = theano.shared(np.asarray(data_x, dtype=configX))
    shared_y = theano.shared(np.asarray(data_y , dtype=configX))
    return shared_x, T.cast(shared_y, dtype='int32')
#---------function--------definition-----end---------------

#test_set_x, test_set_y = shared_dataset(test_set)
#valid_set_x, valid_set_y = shared_dataset(valid_set)
#train_set_x, train_set_y = shared_dataset(train_set)

#----setting the batch size--------------------------------
#batch_size = 500

#-----Accessing the 3rd minibatch of the training set------
#data = train_set_x [ 2*500 : 3*500 ]
#Label = train_set_y [ 2*500 : 3*500 ]

#----------------0-1 Loss----------------------------------
def zero_one_loss(p_y_given_x):
    L= T.sum(T.neq(T.argmax(p_y_given_x)))
    return L
#----------------End of 0-1 loss function-------------------

#---Begin the definition of log likelihood loss function----
def log_likelihood_loss(y, p_y_given_x):
    LL = T.sum(T.log(p_y_given_x[T.arange(y.shape[0]),y])) 
    return LL
#----End of log likelihood function definition--------------

#--beginning of definition of negative log likelihood loss function-------
def negative_log_likelihood(y, p_y_given_x):
    NLL = -(T.sum(T.log(p_y_given_x[T.arange(y.shape[0]), y])))
    return NLL
#----End of negative log likelihood loss function----------

#------Begin definition of Minibatch SGD-------------------
def MiniSGD(loss, params, learning_rate, training_batches, x_batch, y_batch, stopping_conditions):
    stopping_condition_is_met = stopping_conditions
    d_loss_wrt_params = T.grad(loss, params)
    updates = [(params, params - learning_rate * d_loss_wrt_params)]
    MSGD = theano.function([x_batch, y_batch], loss, updates = updates)
    
    for(x_batch, y_batch) in training_batches:
        print ('Current Loss is', MSGD(x_batch, y_batch))
        if stopping_condition_is_met :
            return params
#--------End of Minibatch SGD------------------------------

#--------Begin definition of the L1/L2 regularization------
def L1_L2_regularization(params, y, p_y_given_x, lambda_1, lambda_2):
    L1 = T.sum(abs(params))
    L2 = T.sum(params ** 2)
    loss = negative_log_likelihood(y, p_y_given_x) + lambda_1 * L1 + lambda_2 * L2
    return loss
#----End of the L1-L2 regularization definition------------

    

                
                    
