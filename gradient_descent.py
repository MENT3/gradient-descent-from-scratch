#!/usr/bin/env python3
import numpy as np

from metrics import Metrics

def batch_gradient_descent(X, y, lr, max_iters, beta=None):
  if beta is None:
    beta = np.random.rand(X.shape[1])
  else:
    assert isinstance(beta, np.ndarray)

  # init metrics
  losses = [Metrics.mse(X, y, beta)]
  betas = [beta.copy()]

  for _ in range(max_iters):
    grad = X.T.dot(X.dot(beta) - y) / X.shape[0]
    beta -= lr * grad
    # save metrics
    losses.append(Metrics.mse(X, y, beta))
    betas.append(beta.copy())

  return np.array(betas), np.array(losses)
