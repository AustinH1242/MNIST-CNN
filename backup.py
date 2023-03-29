import math
from os import system as sys
from time import sleep as slp
import numpy as np
import random
from mnist import MNIST

def main():
	# Defines text color codes for use in printing outputs
	class text_color:
		red = '\033[38;5;1m'
		green = '\033[38;5;2m'
		yellow = '\033[38;5;226m'
		white = '\033[38;5;255m'
		orange = '\033[38;5;214m'

	
	# Creates Node class to house the network's neurons
	class Node:
		def __init__(self, pos, weights, bias, inputs, state):
			self.pos = pos
			self.layer, self.row = pos
			self.weights = weights
			self.bias = bias
			self.inputs = inputs
			self.prevLayerLen = len(inputs)
			self.state = state
			if self.state == 'input':
				self.isInput = True
			else:
				self.isInput = False
			if self.state == 'output':
				self.isOutput = True
			else:
				self.isOutput = False
			self.output = self.calcOutput()
			self.all = f'pos: {self.pos}\nstate: {self.state}\nweights: {self.weights}\ninputs: {self.inputs}\noutput: {self.output}\n'
			
			
		def sigmoidActivation(self, x):
			if x < -700:
				x = -700
			return 1 / (1 + math.exp(-1 * x))

			
		def calcOutput(self):
			result = 0
			if self.state != 'input':
				for (input, val) in self.inputs.items():
						result += val * self.weights[input] + self.bias * self.weights['bias']
				if self.state != 'output':
					return self.sigmoidActivation(result)
				else:
					return result
			else:
				return inputs[self.row]


				
	# Creates neural network class to hold Node objects
	class Network:
		def __init__(self, layers, rows, inputs, outputs):
			self.layers = layers
			self.rows = rows
			self.inputs = inputs
			self.numOutputs = outputs
			self.outputs = [0 for i in range(self.numOutputs)]
			self.numInputs = len(self.inputs)
			self.choice = None
			self.nodes = {}
			# Loops over each node to create and store them
			for x in range(self.layers+2):
				num = 0
				if x == self.layers+1:
					num = self.numOutputs
					state = 'output'
				elif x != 0:
					num = self.rows
					state = 'hidden'
				else:
					num = self.numInputs
					state = 'input'
				for y in range(num):
					bias = 1
					pos = (x,y)
					weights = {'bias': random.uniform(-1,1)}
					inputs = {}
					if x == 0:
						prevLayer = 0
						bias = 0
						inputs[pos] = self.inputs[y]
					elif x == 1:
						prevLayer = self.numInputs
					else:
						prevLayer = self.rows
					for i in range(prevLayer):
						prevKey = (x-1,i)
						weights[prevKey] = random.uniform(-1,1)
						inputs[prevKey] = self.nodes[prevKey].calcOutput()
					node = Node(pos, weights, bias, inputs, state)
					self.nodes[pos] = node

					
		def printNodes(self):
			for pos, node in self.nodes.items():
				print(node.all)

		
		def printOutputs(self):
			sys('clear')
			for pos, node in self.nodes.items():
				if node.state == 'output':
					output = node.output
					if output <= 0.1:
						color = text_color.red
					elif output <= 0.3:
						color = text_color.orange
					elif output <= 0.7:
						color = text_color.yellow
					else:
						color = text_color.green
					print(f'{color}{pos[1]}: {output:.7f}')
					print(f'{text_color.white}')
			print(f'Choice: {self.choice}')

		
		def softMax(self, pos, v):
			y = v[pos[1]]
			sum = 0
			for num in v:
				sum += math.e ** num
			result = (math.e ** y) / sum
			self.nodes[pos].output = result
			return result

		
		def finalize(self):
			v = []
			max = [0,0]
			for pos, node in self.nodes.items():
				if node.state == 'output':
					v.append(node.output)
			for pos, node in self.nodes.items():
				if node.state == 'output':					
					node.output = self.softMax(pos, v)
					self.outputs[pos[1]] = node.output
					if node.output > max[0]:
						max[0] = node.output
						max[1] = pos[1]
			self.choice = max[1]


		def squaredError(self, truth):
			vals = [0 for i in range(10)]
			vals[truth] = 1
			sum = 0
			for i in range(len(self.outputs)):
				sum += (vals[i]-self.outputs[i])**2
			return sum
				
		
			
		def backProp(self):
			pass

			
		def train(self, generations, imageFiles):
			for i in range(generations):
				for imageFile in imageFiles:
					pass 


			
	sys('clear')
	#mndata = MNIST('./python-mnist/data')
	#mndata.gz = True
	#trainImages, trainLabels = mndata.load_training()
	#testImages, testLabels = mndata.load_testing()

	learningRate = 1
	hiddenLayers = 2
	rows = 10
	inputs = [0 for i in range(784)]
	OUTPUTS = 10 # Do NOT change
	network = Network(hiddenLayers, rows, inputs, OUTPUTS)
	network.finalize()
	network.printOutputs()
	print(network.squaredError(9))



if __name__ == '__main__':
	main()