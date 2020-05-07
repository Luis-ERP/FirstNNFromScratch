import numpy as np

class Neural_Network:
	def __init__(self, inputs, hiddens, outputs, learning_rate=0.1):
		self.n_inputs = inputs
		self.n_hiddens = hiddens
		self.n_outputs = outputs
		self.w_ih = np.random.randn(self.n_inputs, self.n_hiddens) # np.array([[0.1,0.25,0.3],[0.5,0.4,0.1]])
		self.w_ho = np.random.randn(self.n_hiddens, self.n_outputs) # np.array([[0.2,0.1],[0.3,0.6],[0.4,0.7]])

		self.bias_h = np.random.randn(self.n_hiddens) # np.array([0.1,0.1,0.1])
		self.bias_o = np.random.randn(self.n_outputs) # np.array([0.1,0.1])

		self.learning_rate = learning_rate

	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))

	def prime_sigmoid(self, sigmoid):
		return sigmoid * (1 - sigmoid)

	def feed_forward(self, inputs):
		hidden_o = np.dot(inputs, self.w_ih)
		hidden_o = np.add(hidden_o, self.bias_h)
		hidden_o = self.sigmoid(hidden_o)
		self.hidden_o = hidden_o

		output = np.dot(hidden_o, self.w_ho)
		output = np.add(output, self.bias_o)
		output = self.sigmoid(output)

		return output

	def back_propagation(self, targets, outputs):
		# calculate error for output layer
		err = targets - outputs
		w_ho_deltas = err * np.dot(self.learning_rate, self.prime_sigmoid(outputs))
		w_ho_deltas = np.dot(w_ho_deltas, np.transpose(w_ho_deltas)) # dimension (1 x 3)

		#calculate error for the hidden layer
		err_h = np.dot(w_ho_deltas, np.transpose(self.hidden_o))
		w_ih_deltas = err_h * np.dot(self.learning_rate, self.prime_sigmoid(self.hidden_o))
		w_ih_deltas = np.dot(w_ih_deltas, np.transpose(w_ih_deltas))

		#calculate the deltas for the bias
		b_ho_deltas = err * np.dot(self.learning_rate, self.prime_sigmoid(outputs))
		b_ih_deltas = err_h * np.dot(self.learning_rate, self.prime_sigmoid(self.hidden_o))

		#tunning the weights
		self.w_ih += w_ih_deltas
		self.w_ho += w_ho_deltas
		#tunning the bias
		self.bias_h += b_ih_deltas
		self.bias_o += b_ho_deltas

		# prints
		print(err)

	def train(self, inputs, targets, epochs=1000):
		import random
		for n in range(epochs):
			value = random.randrange(0,len(inputs))
			# feed formward
			outputs = self.feed_forward(inputs[value])
			# backward
			self.back_propagation(targets[value], outputs)

	def predict(self, inputs):
		prediction = self.feed_forward(inputs)
		
		print('prediction is',prediction)

# setup
import csv
nn = Neural_Network(2, 2, 1)

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([0,1,1,0])
nn.train(inputs, targets)
nn.predict([0,0])

"""
with open('db.csv', 'r') as f:
	data = list(csv.reader(f))

	for index, row in enumerate(data):
		for index2, value in enumerate(row):
			del data[index][7:8]
			data[index][index2] = float(value)

	data = np.array(data)
	inputs = data[:, [0,1,2,3]]
	targets = data[:, [4,5,6]]

	nn.train(inputs, targets)
	nn.predict([5.90,3,5.1,1.8])
"""