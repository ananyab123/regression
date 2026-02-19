from typing import Optional, Any
from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(BaseEstimator):
    """
    A class that implements linear regression (fit and predict)
    """

    def __init__(self, coefficients: Optional[np.ndarray] = None) -> None:
        self.coefficients = coefficients

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit a linear regression model to the given data.
        Add a column of ones to X for the intercept term.
        Use the closed-form equation to find the coefficients of the linear regression model.
        Store the coefficients in the coefficients attribute.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """
        # TODO
        # Add a column of ones to X for the intercept term
        X = np.asarray(X)
        if X.ndim ==1:
            X = X.reshape(-1, 1)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        self.coefficients = np.linalg.pinv(Xb) @ y # come back to this later idk if i did it right
        return self
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data using the trained model.
        
        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
        - Output:
            y_pred: a numpy array of shape (n,) containing the predicted labels
        """
        # TODO
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_bias =np.c_[np.ones((X.shape[0], 1)), X]
        y_pred =X_bias @ self.coefficients
        return y_pred

    def mse(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the mean squared error of the model on the given data.
        Use predict to compute y_pred and compute the MSE.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y_true: a numpy array of shape (n,) containing the true labels
        - Output:
            The mean squared error of the model on the given data.
        """
        # TODO
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y_true) ** 2)
        return mse

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Return the coefficients of the linear regression model.
        """
        return self.coefficients
    
    def plot_model(self, X: np.ndarray, y: np.ndarray, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        """
        Plot the data points and the model.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
            title: a string containing the title of the plot
            xlabel: a string containing the label for the x-axis
            ylabel: a string containing the label for the y-axis
        """
        plt.plot(X, y, 'o')
        xs = np.linspace(X.min(), X.max(), 100)
        ys = self.predict(xs)
        plt.plot(xs, ys, label="Model")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

class PolynomialRegression(BaseEstimator):
    """
    A class that implements polynomial regression (fit and predict)
    """
    
    def __init__(self, degree: int = 2, coefficients: Optional[np.ndarray] = None, ridge_regression: bool = False, lam: float = 0.1) -> None:
        self.degree = degree
        if ridge_regression:
            self.linear_regression = RidgeRegression(lam=lam, coefficients=coefficients)
        else:
            self.linear_regression = LinearRegression(coefficients=coefficients)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fit the model to the data (find the coefficients of underlying model)
        Transform X to polynomial features and then fit a linear regression model to the transformed data.

        X needs to be transformed into X_poly where each column of X_poly is X raised to the power of i, where i is the column index.
        fit the parameters of self.linear_regression to X_poly and y.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """
        # TODO
        #Poly features using numpy
        X = np.asarray(X)
        if X.ndim== 1:
            X = X.reshape(-1, 1)
        X_poly = np.concatenate([X ** i for i in range(1, self.degree + 1)], axis=1)
        self.linear_regression.fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from underlying model.

        Again, X must be transformed into polynomial features before being passed to the underlying model.
        Once X has been transformed, you can call predict on X_poly.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
        - Output:
            y_pred: a numpy array of shape (n,) containing the predicted labels
        """
        # TODO
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_poly = np.concatenate([X** i for i in range(1, self.degree + 1)], axis=1)
        y_pred =self.linear_regression.predict(X_poly)
        return y_pred

    def mse(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the mean squared error of the model on the given data.
        Use the underlying model to compute the MSE.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y_true: a numpy array of shape (n,) containing the true labels
        - Output:
            The mean squared error of the model on the given data.
        """
        # TODO
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y_true)** 2)
        return mse
    
    def plot_model(self, X: np.ndarray, y: np.ndarray, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        self.linear_regression.plot_model(X, y, title, xlabel, ylabel)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        return self.linear_regression.get_coefficients()

class RidgeRegression(LinearRegression):
    def __init__(self, lam: float = 0.1, coefficients: Optional[np.ndarray] = None) -> None:
        self.lam = lam
        self.coefficients = coefficients
  
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit a ridge regression model to the given data.
        Add a column of ones to X for the intercept term.
        Use the closed-form equation to find the coefficients of the ridge regression model.
        Store the coefficients in the coefficients attribute.

        - Input:
            X: a numpy array of shape (n, d) where n is the number of data points and d is the number of features
            y: a numpy array of shape (n,) containing the true labels
        """
        # TODO
        X = np.asarray(X)
        if X.ndim == 1:
            X=X.reshape(-1, 1)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]

        # identity matrixc
        I = np.eye(Xb.shape[1])
        I[0, 0] =0
        A = Xb.T @ Xb + self.lam * I
        b = Xb.T @ y

        self.coefficients = np.linalg.pinv(A) @ b
        return self