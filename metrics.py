class Metrics():
  @staticmethod
  def mse(X, y, beta):
    y_pred = X.dot(beta)
    return (y_pred - y).dot(y_pred - y) / X.shape[0]
