from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        self.pi_ = np.zeros((self.classes_.shape[0]))
        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for k in self.classes_:
            X_k = X[y == k]
            self.pi_[k] = X_k.shape[0] / X.shape[0]
            self.mu_[k] = np.mean(X_k, axis=0)
            self.vars_[k] = (np.sum(((X_k - self.mu_[k]) ** 2), axis=0)) / \
                            X_k.shape[0]
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        x_likelihoods = np.array([])
        for x_index, x in enumerate(X):
            temp_likelihoods = np.array([])
            for k_index, k in enumerate(self.classes_):
                likelihood = 1
                for j in range(X.shape[1]):
                    sigma = self.vars_[k_index][j]
                    normal_distr = (np.exp(
                        - (X[x_index][j] - self.mu_[k_index][j]) ** 2 / (
                                2 * sigma)) / np.sqrt(
                        2 * np.pi * sigma))
                    likelihood = likelihood * normal_distr * self.pi_[k]
                temp_likelihoods = np.append(temp_likelihoods, likelihood)
            if x_index == 0:
                x_likelihoods = temp_likelihoods
            else:
                x_likelihoods = np.vstack((x_likelihoods, temp_likelihoods))
        return x_likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
