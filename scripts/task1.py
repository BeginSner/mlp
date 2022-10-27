from unicodedata import name
import os
import matplotlib.pyplot as plt
import numpy as np
from mlp.optimisers import Optimiser
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

def ploting(stats_list, keys_list, stats_interval, hidden_num_list):
   # Plot the change in the validation and training set error over training.
   fig_1 = plt.figure(figsize=(8, 4))
   ax_1 = fig_1.add_subplot(111)
   for index, hidden_num in enumerate(hidden_num_list):
      for k in ['error(train)', 'error(valid)']:
         ax_1.plot(np.arange(1, stats_list[index].shape[0]) * stats_interval, 
                  stats_list[index][1:, keys_list[index][k]], label=k+'width {}'.format(hidden_num))
   ax_1.legend(loc=0)
   ax_1.set_xlabel('Epoch number')
   ax_1.set_ylabel('Error')

   # Plot the change in the validation and training set accuracy over training.
   fig_2 = plt.figure(figsize=(8, 4))
   ax_2 = fig_2.add_subplot(111)
   for index, hidden_num in enumerate(hidden_num_list):
      for k in ['acc(train)', 'acc(valid)']:
         ax_2.plot(np.arange(1, stats_list[index].shape[0]) * stats_interval, 
                  stats_list[index][1:, keys_list[index][k]], label=k+'width {}'.format(hidden_num))
   ax_2.legend(loc=0)
   ax_2.set_xlabel('Epoch number')
   ax_2.set_ylabel('Accuracy')
   fig_1.tight_layout()
   fig_1.savefig("one_hidden_layer_error.pdf")
   fig_2.tight_layout()
   fig_2.savefig("one_hidden_layer_acc.pdf")
   plt.show()
   return fig_1, ax_1, fig_2, ax_2

def set_state(seed):
   # Seed a random number generator
   seed = 11102019 
   rng = np.random.RandomState(seed)
   return rng
   
def data_load(batch_size, rng):
   # Create data provider objects for the MNIST data set
   train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
   valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
   return train_data, valid_data
   
def train_and_plot(params):
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
   for hidden_dim in params['hidden_dim_list']:
      for layer_num in params['num_layers']:
         print("----------------------------hidden_dim = {}---------------------------------".format(hidden_dim))
         print("----------------------------num_layers = {}---------------------------------".format(layer_num))
         # initialize weights
         weights_init = GlorotUniformInit(rng=rng)
         biases_init = ConstantInit(0.)

         # Create model with ONE hidden layer
         if layer_num == 1:
            model = MultipleLayerModel([
               AffineLayer(params['input_dim'], hidden_dim, weights_init, biases_init), # hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, params['output_dim'], weights_init, biases_init) # output layer
            ])
         elif layer_num == 2:
            model = MultipleLayerModel([
               AffineLayer(params['input_dim'], hidden_dim, weights_init, biases_init), # first hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), # second hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, params['output_dim'], weights_init, biases_init) # output layer
            ])
         elif layer_num == 3:
            model = MultipleLayerModel([
               AffineLayer(params['input_dim'], hidden_dim, weights_init, biases_init), # first hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), # second hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), # third hidden layer
               ReluLayer(),
               AffineLayer(hidden_dim, params['output_dim'], weights_init, biases_init) # output layer
            ])
            
         error = CrossEntropySoftmaxError()
         # Use a Adam learning rule
         learning_rule = AdamLearningRule(learning_rate=params['learning_rate'])
         stats, keys, run_time = training(model, error, learning_rule, train_data, valid_data, params['num_epochs'], params['stats_interval'], notebook=True)
         stats_list.append(stats)
         keys_list.append(keys)

         # dont forget to reset dataset!!!
         train_data.reset()
         valid_data.reset()
   # plot
   fig_1, ax_1, fig_2, ax_2 = ploting(stats_list, keys_list, params['stats_interval'], hidden_num_list = params['hidden_dim_list'])




if __name__ == "__main__":
   os.environ['MLP_DATA_DIR'] = "C:\\Users\\mrj\\Desktop\\courses\\mlp\\mlpractical\\data\\"
   # Setup hyperparameters
   params = dict(
   batch_size = 100,
   learning_rate = 9e-4,
   num_epochs = 100,
   stats_interval = 1,
   input_dim = 784, 
   output_dim = 47,
   hidden_dim_list = [32,64,128],
   seed = 11102019,
   num_layers = [1])

   _ = train_and_plot(params)