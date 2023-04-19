import numpy as np
class LinearRegression:
    """
    Class for performing Linear Regression using Gradient Descent Algorithm.
    """
    
    def __init__(self, n):
        """
        Parameters
        -----------
        n: int
            Number of training instances.
        """
        self.w = np.zeros(n)
    
    def predict(self, X):
        """
        Function returns the value wX + b

        Parameters
        -----------
        X: Numpy ndarray
            Feature Matrix.


        Returns
        -------
        y: Numpy ndarray
            Predicted label vector

        """
        return self.w @ X
    
    def loss(self, X, y):
        """
        Function to calculate the loss of the model.

        Parameters
        -----------
        X: Numpy ndarray
            Feature Matrix
        y: Numpy ndarray
            Label Vector

        Returns
        -------
        loss: Numpy ndarray
            loss of the model
        

        
        """

        e = y - self.predict(y)
        loss = (1/2) * (np.transpose(e) @ e)
        return loss


    def update_weights(self, grad, lr):
        """
        Gradient Descent weight updation function.

        Parameters
        ----------
        grad: Numpy ndarray
            gradient of loss with respect to w
        lr: float
            Learning Rate


        Returns
        -------
        w: Numpy ndarray
            Updated Weights
        """
        return self.w - (lr * grad)

    def fit(self, X, y, epochs, lr):

        """
        Function to estimate parameters of the linear regression model using Gradient Descent Algorithm.

        Parameters
        ----------
        X: Numpy ndarray
            Feature Matrix
        y: Numpy ndarray
            Label Vector
        epochs: int
            Number of training steps
        lr: float
            Learning Rate

        Returns
        -------
        w: Numpy ndarray
            Weight Vector
        """


        self.w_all = []
        self.errors = []
        for i in np.arange(0, epochs):
            loss_grad = np.transpose(X) @ (self.predict(X) - y)
            self.w_all.append(self.w)
            self.errors.append(self.loss(X, y))
            self.w = self.update_weights(loss_grad, lr)
        return self.w