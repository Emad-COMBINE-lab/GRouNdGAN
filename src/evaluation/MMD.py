import typing

from sklearn.neighbors import NearestNeighbors
from keras import backend as K
import numpy as np
import tensorflow as tf


class MMD:
    """
    Maximum Mean Discrepancy (MMD) class for computing distribution similarity
    between real and generated samples using Gaussian kernels.
    """

    def __init__(self, real_cells: np.ndarray):
        """
        Initialize the MMD class with scale and weight parameters based on the median
        nearest neighbor distance among real cells.

        Parameters
        ----------
        real_cells : np.ndarray
            A NumPy array representing real cell data (cells x features).
        """
        n_neighbors = 25
        med = np.ones(20)
        for ii in range(1, 20):
            sample = real_cells[
                np.random.randint(real_cells.shape[0] - 1, size=real_cells.shape[0]), :
            ]
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
            distances, _ = nbrs.kneighbors(sample)
            # nearest neighbor is the point so we need to exclude it
            med[ii] = np.median(distances[:, 1:n_neighbors])

        med = np.median(med)
        scales = [med / 2, med, med * 2]

        weights = K.eval(K.shape(scales)[0])
        weights = K.variable(value=np.asarray(weights))

        self.scales = np.expand_dims(np.expand_dims(scales, -1), -1)
        self.weights = np.expand_dims(np.expand_dims(weights, -1), -1)

    def squaredDistance(
        self,
        X: typing.Union[np.ndarray, "tf.Tensor"],
        Y: typing.Union[np.ndarray, "tf.Tensor"],
    ) -> "tf.Tensor":
        """
        Compute pairwise squared Euclidean distances between rows of X and Y.

        Parameters
        ----------
        X : np.ndarray or tf.Tensor
            Input array of shape (n, d).

        Y : np.ndarray or tf.Tensor
            Input array of shape (m, d).

        Returns
        -------
        tf.Tensor
            A tensor of shape (n, m) representing squared distances.
        """
        # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
        # broadcasted subtraction, a square, and a sum.
        r = K.expand_dims(X, axis=1)
        return K.sum(K.square(r - Y), axis=-1)

    def gaussian_kernel(
        self,
        a: typing.Union[np.ndarray, "tf.Tensor"],
        b: typing.Union[np.ndarray, "tf.Tensor"],
    ) -> "tf.Tensor":
        """
        Compute the multi-scale Gaussian kernel between two datasets.

        Parameters
        ----------
        a : np.ndarray or tf.Tensor
            Input array of shape (n, d).

        b : np.ndarray or tf.Tensor
            Input array of shape (m, d).

        Returns
        -------
        tf.Tensor
            A tensor of shape (n, m) representing the Gaussian kernel matrix.
        """
        numerator = np.expand_dims(self.squaredDistance(a, b), 0)
        return np.sum(self.weights * np.exp(-numerator / (np.power(self.scales, 2))), 0)

    def compute(
        self,
        a: typing.Union[np.ndarray, "tf.Tensor"],
        b: typing.Union[np.ndarray, "tf.Tensor"],
    ) -> "tf.Tensor":
        """
        Compute the Maximum Mean Discrepancy (MMD) between two samples.

        Parameters
        ----------
        a : np.ndarray or tf.Tensor
            First sample of shape (n, d).

        b : np.ndarray or tf.Tensor
            Second sample of shape (m, d).

        Returns
        -------
        tf.Tensor
            The MMD score between the two distributions.
        """
        return (
            self.gaussian_kernel(a, a).mean()
            + self.gaussian_kernel(b, b).mean()
            - 2 * self.gaussian_kernel(a, b).mean()
        )
