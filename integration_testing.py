import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from support_spinor_machine import SupportSpinorMachine

def test_classification():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    model = SupportSpinorMachine(C=1, gamma=1e-3, mu=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    assert acc >= 0.9, f"Expected accuracy >= 0.9, but got {acc}"

def test_regression():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    model = SupportSpinorMachine(C=10, gamma=1e-3, mu=1, regression=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 30, f"Expected MSE < 30, but got {mse}"

if __name__ == '__main__':
    test_classification()
    test_regression()
    print("All tests passed!")
