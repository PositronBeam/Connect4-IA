# %matplotlib inline

import logging
import config
import numpy as np

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model

from loss import softmax_cross_entropy_with_logits

import loggers as lg

import tensorflow.keras.backend as K

VALUE_HEAD = 'value_head'
POLICY_HEAD = 'policy_head'

class Gen_Model():
	def __init__(self, reg_const, learning_rate, input_dim, output_dim):
		self.reg_const = reg_const
		self.learning_rate = learning_rate
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.model = None

	def predict(self, x):
		return self.model.predict(x)

	def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
		return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split = validation_split, batch_size = batch_size)

	def write(self, version):
		self.model.save(config.run_folder + 'models/version' + "{0:0>4}".format(version) + '.h5')

	def read(self, game, run_number, version):
		return load_model( config.run_archive_folder + game + '/run' + str(run_number).zfill(4) + "/models/version" + "{0:0>4}".format(version) + '.h5', custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})

	def printWeightAverages(self):
		layers = self.model.layers
		for i, l in enumerate(layers):
			try:
				x = l.get_weights()[0]
				lg.logger_model.info('WEIGHT LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('------------------')
		for i, l in enumerate(layers):
			try:
				x = l.get_weights()[1]
				lg.logger_model.info('BIAS LAYER %d: ABSAV = %f, SD =%f, ABSMAX =%f, ABSMIN =%f', i, np.mean(np.abs(x)), np.std(x), np.max(np.abs(x)), np.min(np.abs(x)))
			except:
				pass
		lg.logger_model.info('******************')


class RandomModel():

    def predict(self, input):

        batch_size = len(input)

        preds = [None] * 2

        preds[0] = np.array([[config.VALUE_DEFAULT_ACTION] for i in range(batch_size)])
        preds[1] = [np.random.randint(-100, 100, size=config.GRID_SHAPE[1]) for i in range(batch_size)] 

        return preds


class GenRandomModel(Gen_Model):

	def __init__(self):
		Gen_Model.__init__(self, 1, 1, 1, 1)
		self.model = RandomModel()


class Residual_CNN(Gen_Model):
	def __init__(self, reg_const, learning_rate, input_dim,  output_dim, hidden_layers):
		Gen_Model.__init__(self, reg_const, learning_rate, input_dim, output_dim)
		self.hidden_layers = hidden_layers
		self.num_layers = len(hidden_layers)
		self.model = self._build_model()

	def residual_layer(self, input_block, filters, kernel_size):

		x = self.conv_layer(input_block, filters, kernel_size)

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)

		x = add([input_block, x])

		x = LeakyReLU()(x)

		return (x)

	def conv_layer(self, x, filters, kernel_size):

		x = Conv2D(
		filters = filters
		, kernel_size = kernel_size
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		return (x)

	def value_head(self, x):

		x = Conv2D(
		filters = 1
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)


		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		x = Dense(
			20
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			)(x)

		x = LeakyReLU()(x)

		x = Dense(
			1
			, use_bias=False
			, activation='tanh'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = VALUE_HEAD
			)(x)



		return (x)

	def policy_head(self, x):

		x = Conv2D(
		filters = 2
		, kernel_size = (1,1)
		, data_format="channels_first"
		, padding = 'same'
		, use_bias=False
		, activation='linear'
		, kernel_regularizer = regularizers.l2(self.reg_const)
		)(x)

		x = BatchNormalization(axis=1)(x)
		x = LeakyReLU()(x)

		x = Flatten()(x)

		# The output is a linear value, not a softmax, because we force the probabilities of the forbidden actions
		# to 0 in agent.get_preds(self, state). We do this by setting the output of the policy_head to -100 before
		# computing the softmax.
		x = Dense(
			self.output_dim
			, use_bias=False
			, activation='linear'
			, kernel_regularizer=regularizers.l2(self.reg_const)
			, name = POLICY_HEAD
			)(x)

		return (x)

	def _build_model(self):

		main_input = Input(shape = self.input_dim, name = 'main_input')

		x = self.conv_layer(main_input, self.hidden_layers[0]['filters'], self.hidden_layers[0]['kernel_size'])

		if len(self.hidden_layers) > 1:
			for h in self.hidden_layers[1:]:
				x = self.residual_layer(x, h['filters'], h['kernel_size'])

		vh = self.value_head(x)
		ph = self.policy_head(x)

		model = Model(inputs=[main_input], outputs=[vh, ph])

		return model

	def compile_with_loss_weights(self, loss_weights={VALUE_HEAD: 0.5, POLICY_HEAD: 0.5}):

		self.model.compile(loss={VALUE_HEAD: 'mean_squared_error', POLICY_HEAD: softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),	
			loss_weights=loss_weights	
			)
		return self.model

	def plot_cnn(self, to_file=config.run_folder + 'models/model.png'):
		plot_model(self.model, to_file=to_file, show_shapes = True)
