from cProfile import label
from re import L
from unicodedata import name
import os
import matplotlib.pyplot as plt
import numpy as np
from mlp.optimisers import Optimiser
from mlp.penalties import L1Penalty, L2Penalty
plt.style.use('ggplot')
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider, EMNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.layers import DropoutLayer

def training(
   model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
   
   # As well as monitoring the error over training also monitor classification
   # accuracy i.e. proportion of most-probable predicted classes being equal to targets
   data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

   # Use the created objects to initialise a new Optimiser instance.
   optimiser = Optimiser(
      model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

   # Run the optimiser for num_epochs epochs (full passes through the training set)
   # printing statistics every epoch.
   stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
   return stats, keys, run_time

def set_state(seed):
   # Seed a random number generator
   rng = np.random.RandomState(seed)
   return rng
   
def data_load(batch_size, rng):
   # Create data provider objects for the MNIST data set
   train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
   valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
   return train_data, valid_data
   
def train(params):
   # set random state
   rng = set_state(params['seed'])
   # Set up a logger object to print info about the training run to stdout
   logger = logging.getLogger()
   logger.setLevel(logging.INFO)
   logger.handlers = [logging.StreamHandler()]
   # load data
   train_data, valid_data = data_load(params['batch_size'], rng)

   # data_stats and keys
   stats_list = []
   keys_list = []
   
   
   # initialize weights
   weights_init = GlorotUniformInit(rng=rng)
   biases_init = ConstantInit(0.)

   # Create dropout model
   
   # model = MultipleLayerModel([
   #    AffineLayer(params['input_dim'], params['hidden_dim'], weights_init, biases_init), # first hidden layer
   #    DropoutLayer(rng=rng, incl_prob=params['incl_prob']),
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init), # second hidden layer
   #    DropoutLayer(rng=rng, incl_prob=params['incl_prob']),
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init), # third hidden layer
   #    DropoutLayer(rng=rng, incl_prob=params['incl_prob']),
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['output_dim'], weights_init, biases_init) # output layer
   # ])

   # # Create L1 penalty model
   # model = MultipleLayerModel([
   #    AffineLayer(params['input_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L1Penalty(coefficient=params['L1_coff'])), # first hidden layer
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L1Penalty(coefficient=params['L1_coff'])), # second hidden layer
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L1Penalty(coefficient=params['L1_coff'])), # third hidden layer
   #    ReluLayer(),
   #    AffineLayer(params['hidden_dim'], params['output_dim'], weights_init, biases_init) # output layer
   # ])

   # Create L2 penalty model
   model = MultipleLayerModel([
      AffineLayer(params['input_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L2Penalty(coefficient=params['L2_coff'])), # first hidden layer
      ReluLayer(),
      AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L2Penalty(coefficient=params['L2_coff'])), # second hidden layer
      ReluLayer(),
      AffineLayer(params['hidden_dim'], params['hidden_dim'], weights_init, biases_init, weights_penalty=L2Penalty(coefficient=params['L2_coff'])), # third hidden layer
      ReluLayer(),
      AffineLayer(params['hidden_dim'], params['output_dim'], weights_init, biases_init) # output layer
   ])
      
   error = CrossEntropySoftmaxError()
   # Use a Adam learning rule
   learning_rule = AdamLearningRule(learning_rate=params['learning_rate'])
   stats, keys, run_time = training(model, error, learning_rule, train_data, valid_data, params['num_epochs'], params['stats_interval'], notebook=True)
   stats_list.append(stats)
   keys_list.append(keys)

def plot_results(data_label, data_result):
   # plot dropout
   x1 = data_label[0]
   y11 = data_result[0][:,0]
   y12 = data_result[0][:,2] - data_result[0][:,1]
   fig1, ax11 = plt.subplots()

   ax12 = ax11.twinx()
   ax11.plot(x1, y11, 'r-', label='Val acc')
   ax12.plot(x1, y12, 'b-', label='Gap')

   ax11.set_xlabel('Dropout value')
   ax11.set_ylabel('Accuracy')
   ax12.set_ylabel('Generalization Gap')
   fig1.legend()
   fig1.tight_layout()
   fig1.savefig("Dropout_value.pdf")

   # plot weight decay
   x2 = data_label[1]
   y21_L1 = data_result[1][:,0]
   y21_L2 = data_result[2][:,0]
   y22_L1 = data_result[1][:,2] - data_result[1][:,1]
   y22_L2 = data_result[2][:,2] - data_result[2][:,1]
   fig2, ax21 = plt.subplots()

   ax22 = ax21.twinx()
   ax21.plot(x2, y21_L1, 'r-', label = 'L1 Val. Acc')
   ax21.plot(x2, y21_L2, 'b-', label = 'L2 Val. Acc')
   ax22.plot(x2, y22_L1, 'r:', label = 'L1 Gap', linewidth=5)
   ax22.plot(x2, y22_L2, 'b:', label = 'L2 Gap', linewidth=5)
   ax21.set_xlabel('Weight decay value')
   ax21.set_ylabel('Accuracy')
   ax22.set_ylabel('Generalization Gap')
   fig2.legend()
   fig2.tight_layout()
   fig2.savefig("weight_decay.pdf")
   plt.show()
   
   

if __name__ == "__main__":
   os.environ['MLP_DATA_DIR'] = "C:\\Users\\mrj\\Desktop\\courses\\mlp\\mlpractical\\data\\"
   # Setup hyperparameters
   params = dict(
   batch_size = 100,
   learning_rate = 1e-4,
   num_epochs = 100,
   stats_interval = 1,
   input_dim = 784, 
   output_dim = 47,
   hidden_dim = 128,
   seed = 23624,
   num_layers = 3,
   incl_prob = 0.95,
   L1_coff = 1e-3,
   L2_coff = 1e-4,
   )
   # _ = train(params)

   dropout = np.array([0.6, 0.7, 0.85, 0.97])
   L1 = np.array([5e-4, 1e-3, 5e-3, 5e-2])
   L2 = np.array([5e-4, 1e-3, 5e-3, 5e-2])
   data_label = [dropout, L1, L2]
   data_dropout = np.array([
      [0.807, 0.549, 0.593],
      [0.830, 0.442, 0.502],
      [0.851, 0.329, 0.434],
      [0.854, 0.244, 0.457]
   ])
   data_L1 = np.array([
      [0.795, 0.642, 0.658],
      [0.771, 0.744, 0.764],
      [0.241, 3.850, 3.850],
      [0.220, 3.850, 3.850]
   ])
   data_L2 = np.array([
      [0.851, 0.306, 0.460],
      [0.849, 0.334, 0.450],
      [0.813, 0.586, 0.607],
      [0.392, 2.258, 2.256]
   ])
   data_result = [data_dropout, data_L1, data_L2]
   plot_results(data_label, data_result)
   