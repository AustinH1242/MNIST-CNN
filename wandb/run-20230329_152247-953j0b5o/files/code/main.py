import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import os
import warnings
warnings.filterwarnings(action="ignore")

model_count = len(os.listdir('savestates/'))
model_name = f'model_{model_count+1}'
input_shape = (28,28,1)

wandb.init(
	    # set the wandb project where this run will be logged
	    project="MNIST",

		# set the display name for the run
		name = model_name,
	
	    # track hyperparameters and run metadata with wandb.config
		config = {
	        "layer_1": ("Conv2D", (5,5), "input_shape=(28,28,1)"),
	        "activation_1": "sigmoid",
			"layer_2": ('Flatten', "input_shape=(24,24,5)"),
	        "layer_3": ("Dropout", 0.2),
			"layer_4": ("Dense", 10),
	        "activation_4": "softmax",
	        "optimizer": "adam",
	        "loss": "sparse_categorical_crossentropy",
	        "metric": "accuracy",
	        "epoch": 4,
	        "batch_size": 300
	    }
		
	)
	# [optional] use wandb.config as your config
config = wandb.config

def create_model():
	model = keras.Sequential([
		Conv2D(5, 5, input_shape=input_shape, activation=config.activation_1),
		Flatten(input_shape=(24,24,5)),
    	Dropout(0.2),
    	Dense(10, activation=config.activation_4)
  ])

	model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])
	return model

def main():
	data = keras.datasets.mnist.load_data()
	(trainImages, trainLabels), (testImages, testLabels) = data
	
	checkpoint_path = f"savestates/{model_name}/cp.ckpt"
	cwd = os.getcwd()
	file_path = cwd + f"/savestates/{model_name}/metrics.txt"

	def print_to_file(s):
		with open(file_path,'a') as f:
			print(s, file=f)
	
	# Create a basic model instance
	model = create_model()
	# Display the model's architecture
	os.system('clear')
	model.summary()
	
	# Create a callback that saves the model's weights
	cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

	# The number of images trained before model weights are updated.
	# Model will update the weights ([# of samples] / [batch size])
	# times per epoch 
	batch_size = config.batch_size
	# Number of times to loop through the training data
	epochs = config.epoch

	# Do the backpropagation and update model weights
	model.fit(trainImages, trainLabels, batch_size=batch_size,  verbose=1, callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models"),cp_callback], epochs = epochs)
	# Evaluate the model's accuracy with the testing data
	score = model.evaluate(testImages, testLabels, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	with open(file_path, 'w') as f:
		model.summary(print_fn = print_to_file, expand_nested = True, show_trainable = True)
		f.write(f'Test accuracy:\n{score[1]}\n\n')
		f.write(f'Test loss:\n{score[0]}\n\n')
	
	wandb.finish()	

	# This is for comparing new vs saved models
	"""
 	# Savestate to be used
	save = 'savestates/model_1/cp.ckpt'
	# Create a basic model instance
	basicModel = create_model()
	# Create a trained model
	trainedModel = create_model()
	trainedModel.load_weights(save)

	# Evaluate the untrained model
	loss, acc = basicModel.evaluate(testImages, testLabels, verbose=2)
	print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

	# Evaluate the trained model
	loss, acc = trainedModel.evaluate(testImages, testLabels, verbose=2)
	print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
	"""
	
if __name__ == '__main__':
	main()