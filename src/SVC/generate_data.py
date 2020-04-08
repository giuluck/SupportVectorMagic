import numpy as np
from numpy.random import random


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
        dimensions=2,
        threshold_factor=1.0,
        outlier_factor=0.5
):
    X = []
    y = []
    while len(X) < n:
        x = 2 * random(dimensions) - 1
        c = hyperplane(x)
        if threshold_factor * random() < abs(c):
            X.append(list(x))
            y.append((outlier_factor * random() > abs(c)) ^ (c < 0))
    return np.array(X), np.array(y)
