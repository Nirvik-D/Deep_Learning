'''
Created on Jan 10, 2015

@author: Lightning
'''
#from Multi_Layer_Perceptron import multiLayerPerceptron
from Logistic_Regression import sgd_optimization_mnist
#from LeNet import leNetModel
#from Denoising_AutoEncoders import denoisingAutoEncoders
#from Stacked_Denoising_AutoEncoders import stackedDenoisingAutoEncoders
#from Restricted_Boltzmann_Machines import restrictedBoltzmannMachines
#from Deep_Belief_Network import deepBeliefNetwork

if __name__ == '__main__':
    
    #----Logistic Regression Optimization-------------------------------
    
    sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, 
                           dataset='mnist.pkl.gz', batch_size=600)
                           
    #-----End of the Logistic Regression--------------------------------
    
    #---Begin implementation of Multi-Layer-Perceptron------------------
    
    #multiLayerPerceptron(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             #dataset='X:\Deep_Learning\mnist.pkl.gz', batch_size=20, n_hidden=500)
             
    #-----End implementation of the MLP----------------------------------

    #------Begin implementing LeNet from here----------------------------
    
    #leNetModel(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             #dataset='X:\Deep_Learning\mnist.pkl.gz', batch_size=20, n_hidden=500, nkerns=[20,50])
    
    #---------End of LeNet implementation--------------------------------
    
    #-----------Begin implementing the Denoising Auto-Encoders-----------
    
    #denoisingAutoEncoders(learning_rate=0.1, training_epochs=15,
            #dataset='X:\Deep_Learning\mnist.pkl.gz',
            #batch_size=20, output_folder='X:\Deep_Learning\dA_plots')
    
    #--------------------End Implementation of dA------------------------
    
    #---------------Begin implementation of Stacked Auto Encoders--------
    
    #stackedDenoisingAutoEncoders(finetune_lr=0.1, pretraining_epochs=15,
             #pretrain_lr=0.001, training_epochs=1000,
             #dataset='X:\Deep_Learning\mnist.pkl.gz', batch_size=1)
    
    #----------------End of the Stacked Auto Encoders ------------------
    
    #------------Begin implementation of Restricted_Boltzmann_Machines--
    
    #restrictedBoltzmannMachines(learning_rate=0.1, training_epochs=15,
             #dataset='X:\Deep_Learning\mnist.pkl.gz', batch_size=20,
             #n_chains=20, n_samples=10, output_folder='X://Deep_Learning//rbm_plots',
             #n_hidden=500, destination_file='X:\Deep_Learning\samples.png')
     
     #-------End of Restricted_Boltzmann_Machines----------------------
     
     #-------Begin implementation of Deep_Belief_Network---------------
     
     #deepBeliefNetwork(finetune_lr=0.1, pretraining_epochs=100,
             #pretrain_lr=0.01, k=1, training_epochs=1000,
             #dataset='X:\Deep_Learning\mnist.pkl.gz', batch_size=10)
             
    #------End of implementation of Deep_Belief_Network---------------