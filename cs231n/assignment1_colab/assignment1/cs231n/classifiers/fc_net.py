from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        
        self.params = {
          'W1': np.random.randn(input_dim, hidden_dim) * weight_scale,
          'b1': np.zeros(hidden_dim),
          'W2': np.random.randn(hidden_dim, num_classes) * weight_scale,
          'b2': np.zeros(num_classes)
        }
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1, W2, b2 = self.params.values()

        out1, cache1 = affine_forward(X, W1, b1)
        out2, cache2 = relu_forward(out1)
        scores, cache3 = affine_forward(out2, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dout3, dW2, db2 = affine_backward(dloss, cache3)
        dout2 = relu_backward(dout3, cache2)
        dout1, dW1, db1 = affine_backward(dout2, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True

class FullyConnectedNet(object):
    """Fully connected net with arbitrary number of hidden layers,
    optional dropout, and optional batch/layer normalization."""

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # Initialize weights and biases.                                           #
        ############################################################################
        layer_input_dim = input_dim
        for i, hd in enumerate(hidden_dims):
            self.params[f"W{i+1}"] = weight_scale * np.random.randn(layer_input_dim, hd)
            self.params[f"b{i+1}"] = np.zeros(hd)
            if self.normalization in ["batchnorm", "layernorm"]:
                self.params[f"gamma{i+1}"] = np.ones(hd)
                self.params[f"beta{i+1}"] = np.zeros(hd)
            layer_input_dim = hd

        # Last layer
        self.params[f"W{self.num_layers}"] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        self.params[f"b{self.num_layers}"] = np.zeros(num_classes)

        ############################################################################
        # Dropout + batchnorm params                                               #
        ############################################################################
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for _ in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        caches = {}
        dropout_caches = {}
        out = X.reshape(X.shape[0], -1)  # flatten input

        ############################################################################
        # Forward pass                                                             #
        ############################################################################
        for i in range(1, self.num_layers):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]

            affine_out, affine_cache = affine_forward(out, W, b)

            norm_cache = None
            if self.normalization == "batchnorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                affine_out, norm_cache = batchnorm_forward(affine_out, gamma, beta, self.bn_params[i-1])
            elif self.normalization == "layernorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                affine_out, norm_cache = layernorm_forward(affine_out, gamma, beta, self.bn_params[i-1])

            relu_out, relu_cache = relu_forward(affine_out)

            if self.use_dropout:
                relu_out, do_cache = dropout_forward(relu_out, self.dropout_param)
                dropout_caches[i] = do_cache

            caches[i] = (affine_cache, norm_cache, relu_cache)
            out = relu_out

        # Last affine layer (no ReLU)
        scores, final_cache = affine_forward(out, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"])
        caches[self.num_layers] = final_cache

        if mode == "test":
            return scores

        ############################################################################
        # Backward pass                                                            #
        ############################################################################
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to loss
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f"W{i}"] ** 2)

        # Last layer backward
        dx, dW, db = affine_backward(dscores, caches[self.num_layers])
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        # Hidden layers backward
        for i in reversed(range(1, self.num_layers)):
            affine_cache, norm_cache, relu_cache = caches[i]

            if self.use_dropout:
                dx = dropout_backward(dx, dropout_caches[i])

            dx = relu_backward(dx, relu_cache)

            if self.normalization == "batchnorm":
                dx, dgamma, dbeta = batchnorm_backward(dx, norm_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            elif self.normalization == "layernorm":
                dx, dgamma, dbeta = layernorm_backward(dx, norm_cache)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta

            dx, dW, db = affine_backward(dx, affine_cache)
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db

        return loss, grads
