import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_data
from SVC import SVC


# the hyperplane to use for generating data
def hp(data): return data[0] / 2 - data[1]


# setting random seed
np.random.seed(0)

# generating train/set data and applying regression
X_train, y_train = generate_data(1000, hp)
X_test, y_test = generate_data(300, hp)
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
plt.plot(x0[r == 'TP'], x1[r == 'TP'], 'bo', label='TP')
plt.plot(x0[r == 'TN'], x1[r == 'TN'], 'ro', label='TN')
plt.plot(x0[r == 'FN'], x1[r == 'FN'], 'bx', label='FN')
plt.plot(x0[r == 'FP'], x1[r == 'FP'], 'rx', label='FP')
plt.plot(x, -(a * x + c) / b, 'k')
plt.title('Results on Test Set')
plt.legend(loc='lower left')
plt.grid()
plt.show()
