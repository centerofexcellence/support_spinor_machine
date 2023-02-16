import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from support_spinor_machine import SupportSpinorMachine

class TestSupportSpinorMachine(unittest.TestCase):

    def test_classification(self):
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = SupportSpinorMachine(C=1, gamma=1e-3, mu=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.assertGreaterEqual(acc, 0.9)

    def test_regression(self):
        X, y = make_regression(n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model = SupportSpinorMachine(C=10, gamma=1e-3, mu=1, regression=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.assertLess(mse, 30)

if __name__ == '__main__':
    unittest.main()
