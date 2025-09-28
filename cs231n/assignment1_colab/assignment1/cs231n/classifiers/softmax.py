import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Naive implementation of softmax loss with loops.

    Inputs:
    - W: (D, C) weights
    - X: (N, D) data
    - y: (N,) labels
    - reg: regularization strength

    Returns:
    - loss: scalar
    - dW: (D, C) gradient
    """
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= scores.max()              # numeric stability
        p = np.exp(scores)
        p /= p.sum()                        # softmax probabilities

        loss -= np.log(p[y[i]])             # cross-entropy loss

        p[y[i]] -= 1                        # gradient adjustment
        dW += np.outer(X[i], p)             # accumulate gradient

    # Average and regularize
    loss = loss / num_train + reg * np.sum(W*W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Vectorized implementation of softmax loss.
    """
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability

    exp_scores = np.exp(scores)
    prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    correct_prob = prob[np.arange(num_train), y]
    loss = np.sum(-np.log(correct_prob)) / num_train
    loss += reg * np.sum(W*W)

    # Gradient
    dscores = prob
    dscores[np.arange(num_train), y] -= 1
    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized_slim(W, X, y, reg):
    """
    Slim, concise vectorized implementation of softmax loss.
    """
    num_train = X.shape[0]

    scores = X @ W
    scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability

    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    correct_softmax = softmax[np.arange(num_train), y]

    loss = np.sum(-np.log(correct_softmax)) / num_train
    loss += reg * np.sum(W*W)

    # Gradient
    softmax[np.arange(num_train), y] -= 1
    dW = X.T @ softmax / num_train
    dW += 2 * reg * W

    return loss, dW
