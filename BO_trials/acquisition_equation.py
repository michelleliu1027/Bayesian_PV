# acquisition equations
import numpy as np

def ucb(X , gp, dim, delta):
        """
        Calculates the GP-UCB acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
        x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        mean, var = gp.predict(X[:, np.newaxis], return_cov=True)
        mean = np.atleast_2d(mean).T
        var = np.atleast_2d(var).T  
        beta = 2*np.log(np.power(5000,2.1)*np.square(np.pi)/(3*delta))
        # return mean - np.sqrt(beta)* np.sqrt(np.diag(var))
        return mean - np.sqrt(beta) * np.sqrt(np.diag(var))
