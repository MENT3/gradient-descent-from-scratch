#!/usr/bin/env python3
import numpy as np

from metrics import Metrics

class GradientDescent:
  def __init__(self, X, y, lr):
    self.X = X
    self.y = y
    self.lr = lr

    self.N, self.D = self.X.shape

  def batch(self, n_iter, beta=None, n_batch=None):
    n_batch = 10 if n_batch is None else n_batch
    _X = np.split(self.X, n_batch)[np.random.randint(0, 9)]
    _y = np.split(self.y, n_batch)[np.random.randint(0, 9)]

    if beta is None:
      _beta = np.random.rand(self.D)
    else:
      assert isinstance(beta, np.ndarray)
      _beta = beta.copy()

    # init metrics
    losses = [Metrics.mse(_X, _y, beta)]
    betas = [_beta]

    for _ in range(n_iter):
      grad = _X.T.dot(_X.dot(_beta) - _y) / self.N
      _beta -= self.lr * grad
      # save metrics
      losses.append(Metrics.mse(_X, _y, _beta))
      betas.append(beta.copy())

    return np.array(betas), np.array(losses)
