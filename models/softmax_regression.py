

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
	def __init__(self, input_size=28 * 28, num_classes=10):
		"""
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
		super().__init__(input_size, num_classes)
		self._weight_init()

	def _weight_init(self):
		'''
		initialize weights of the single layer regression network. No bias term included.
		:return: None; self.weights is filled based on method
		- W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
		'''
		np.random.seed(1024)
		self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
		self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

	def cross_entropy_dev(self, x_pred, y):
		"""
		Compute derivative of Cross-Entropy Loss based on prediction of the network and labels
		:param x_pred: scores from the model (N, num_classes)
		:param y: Labels of instances in the batch
		:return: The computed derivative Cross-Entropy Loss
		"""
		#############################################################################
		# TODO:                                                                     #
		#    1) Implement the derivative of cross entropy function                        #
		#############################################################################
		one_hot_labels = np.zeros((len(y), x_pred.shape[1]))
		for i in range(len(y)):
			one_hot_labels[i, y[i]] = 1
		return (self.softmax(x_pred) - one_hot_labels) / x_pred.shape[0]

	#############################################################################
	#                              END OF YOUR CODE                             #
	#############################################################################

	def forward(self, X, y, mode='train'):
		"""
		Compute loss and gradients using softmax with vectorization.

		:param X: a batch of image (N, 28x28)
		:param y: labels of images in the batch (N,)
		:return:
			loss: the loss associated with the batch
			accuracy: the accuracy of the batch
		"""
		loss = None
		gradient = None
		accuracy = None
		#############################################################################
		# TODO:                                                                     #
		#    1) Implement the forward process and compute the Cross-Entropy loss    #
		#    2) Compute the gradient of the loss with respect to the weights        #
		# Hint:                                                                     #
		#   Store your intermediate outputs before ReLU for backwards               #
		#############################################################################
		Z = X @ self.weights['W1']
		A = self.ReLU(Z)
		S = self.softmax(A)

		loss = self.cross_entropy_loss(S, y)
		accuracy = self.compute_accuracy(S, y)
		#############################################################################
		#                              END OF YOUR CODE                             #
		#############################################################################
		if mode != 'train':
			return loss, accuracy

		#############################################################################
		# TODO:                                                                     #
		#    1) Implement the backward process:                                     #
		#        1) Compute gradients of each weight by chain rule                  #
		#        2) Store the gradients in self.gradients                           #
		#############################################################################
		dL_dA = self.cross_entropy_dev(A, y)
		dA_dZ = self.ReLU_dev(Z)
		dL_dZ = np.multiply(dL_dA, dA_dZ)

		dL_dW = X.T@dL_dZ

		self.gradients['W1'] = dL_dW
		#############################################################################
		#                              END OF YOUR CODE                             #
		#############################################################################
		return loss, accuracy
