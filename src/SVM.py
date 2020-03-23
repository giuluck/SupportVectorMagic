import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.random import random

class SVC:
  #
  # learning_rate: the learning rate of the descent
  #         alpha: the adaptive gradient hyperparameter (if 'None', no adaptive method is used)
  #             C: the regulation factor (if 'None', no regularization is applied)
  #    iterations: the maximum number of iterations 
  #     tolerance: the tolerance to reach
  #
  def __init__(
      self,
      learning_rate = 1e-2, 
      alpha = 10.0, 
      C = 1.0, 
      iterations = 1e6, 
      tolerance = 1e-10,
      info = False
  ):
    #
    # adaptive function
    #
    if alpha == None:
      def af(ratio): return 1
    else:
      sqalpha = np.sqrt(alpha)
      def af(ratio):
        if ratio > 1:
          return alpha**-ratio + 1
        elif ratio > 0:
          return ratio / sqalpha
        else:
          return 1
    #
    # gradient function
    #
    if C == None:
      def hg(X, y, t):
        t = t.reshape(-1, 1)
        return np.sum([-v if np.dot(v, t) < 1 else np.zeros(X.shape[1]) for v in X * y], axis = 0)
    else:
      def hg(X, y, t):
        t = t.reshape(-1, 1)
        g = np.sum([-v if np.dot(v, t) < 1 else np.zeros(X.shape[1]) for v in X * y], axis = 0)
        return C * g + t.reshape(-1,)
    #
    # parameters
    #
    self.adaptive_factor = lambda r: af(r)
    self.gradient = lambda X, y, t: hg(X, y, t)
    self.learning_rate = learning_rate
    self.iterations = int(iterations)
    self.tolerance = tolerance,
    self.info = False

  #
  #     X: the features dataset
  #     y: the classes vector
  # guess: the initial guess (if 'None', a vector of zeros is used)
  #
  def fit(self, X, y, guess = None):
    X_unmodified = X
    y_unmodified = y
    #
    # data preprocessing
    #
    self.classes_ = np.unique(y)
    y = np.array([-1 if c == self.classes_[0] else 1 for c in y]).reshape(-1, 1)
    if X.ndim == 1:
      X = X.reshape(-1, 1)
    ones = np.ones(X.shape[0]).reshape(-1, 1)
    X = np.concatenate((ones, X), axis = 1)
    #
    # actual gradient descent
    #
    variance = 0
    current_rate = self.learning_rate
    theta = np.zeros(X.shape[1]) if guess == None else guess
    for iteration in range(self.iterations):
      previous_variance = variance
      variance = -current_rate * self.gradient(X, y, theta)
      theta = theta + variance
      if norm(variance) < self.tolerance: break
      current_rate = current_rate * self.adaptive_factor(norm(previous_variance / variance))
    #
    # set parameters
    #
    self.coef_ = theta[1:]
    self.intercept_ = theta[0]
    self.labels_ = self.label(X_unmodified, y_unmodified)
    self.evaluations_ = self.evaluate(X_unmodified, y_unmodified)
    return self

  #
  # X: the features
  #
  # > returns an array of predictions
  #
  def predict(self, X):
    v = np.dot(X, self.coef_) + self.intercept_
    return np.array([self.classes_[0] if p < 0 else self.classes_[1] for p in v])

  #
  # X: the features
  # y: the classes
  #
  # > returns an array of labels of kind True/False - Positive/Negative
  #
  def label(self, X, y):
    p = self.predict(X)
    d = { (True, True): 'TP', (False, False): 'TN', (True, False): 'FN', (False, True): 'FP' }
    return np.array([d[(yy, pp)] for (yy, pp) in zip(y, p)])
    
  #
  # X: the features
  # y: the classes
  #
  # > returns a dictionary of the main measures for a binary classifier
  #
  def evaluate(self, X, y):
    results = self.label(X, y)
    TP = len(results[results == 'TP'])
    TN = len(results[results == 'TN'])
    FN = len(results[results == 'FN'])
    FP = len(results[results == 'FP'])
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return { 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1 }

# 
#                n: the number of samples
#       hyperplane: the separation hyperplane
#       dimensions: the number of features
# threshold_factor: factor to increment the probability to reject a sample 
#   outlier_factor: factor to increment the probability to generate an outlier
#
# > returns a tuple containing the dataset of independent features and the
#   array of classes (perturbed with salt-and-pepper noise)
#
def generate_data(
    n,
    hyperplane,
    dimensions = 2,
    threshold_factor = 1.0,
    outlier_factor = 0.5
):
  X = []
  y = []
  while len(X) < n:
    x = 2 * random(dimensions) - 1
    c = hyperplane(x)
    if threshold_factor * random() < abs(c):
      X.append(list(x))
      y.append((outlier_factor * random() > abs(c)) ^ (c < 0))
  return (np.array(X), np.array(y))

# setting random seed
np.random.seed(0)

# generating train/set data and applying regression
hyperplane = lambda x: x[0] / 2 - x[1]
X_train, y_train = generate_data(1000, hyperplane)
X_test, y_test = generate_data(300, hyperplane)
svc = SVC().fit(X_train, y_train)

# evaluating the model on test and train data
training_evaluation = svc.evaluations_
test_evaluation = svc.evaluate(X_test, y_test)
print('EVALUATION OF THE TRAINING SET:')
print('>   Accuracy: {}%'.format(round(100 * training_evaluation['accuracy'])))
print('>  Precision: {}%'.format(round(100 * training_evaluation['precision'])))
print('>     Recall: {}%'.format(round(100 * training_evaluation['recall'])))
print('> F1 Measure: {}%'.format(round((100 * training_evaluation['f1']))))
print()
print('EVALUATION OF THE TEST SET:')
print('>   Accuracy: {}%'.format(round(100 * test_evaluation['accuracy'])))
print('>  Precision: {}%'.format(round(100 * test_evaluation['precision'])))
print('>     Recall: {}%'.format(round(100 * test_evaluation['recall'])))
print('> F1 Measure: {}%'.format(round((100 * test_evaluation['f1']))))

# plotting results
r = svc.label(X_test, y_test)
c = svc.intercept_
a, b = svc.coef_
x = np.linspace(-1, 1, 10)
x0 = X_test[:, 0]
x1 = X_test[:, 1]
plt.plot(x0[r == 'TP'], x1[r == 'TP'], 'bo', label = 'TP')
plt.plot(x0[r == 'TN'], x1[r == 'TN'], 'ro', label = 'TN')
plt.plot(x0[r == 'FN'], x1[r == 'FN'], 'bx', label = 'FN')
plt.plot(x0[r == 'FP'], x1[r == 'FP'], 'rx', label = 'FP')
plt.plot(x, -(a * x + c) / b, 'k')
plt.title('Results on Test Set')
plt.legend(loc = 'lower left')
plt.grid()
plt.show()