#!/usr/bin/env python3
import numpy as np

from metrics import Metrics

class GradientDescent:
  def __init__(self, X, y, lr):
    self.X = X
    self.y = y
    self.lr = lr

    self.N, self.D = self.X.shape

  def batch(self, n_iter, beta=None):
    if beta is None:
      beta = np.random.rand(self.D)
    else:
      assert isinstance(beta, np.ndarray)

    # init metrics
    losses = [Metrics.mse(self.X, self.y, beta)]
    betas = [beta.copy()]

    for _ in range(n_iter):
      grad = self.X.T.dot(self.X.dot(beta) - self.y) / self.N
      beta -= self.lr * grad
      # save metrics
      losses.append(Metrics.mse(self.X, self.y, beta))
      betas.append(beta.copy())

    return np.array(betas), np.array(losses)
