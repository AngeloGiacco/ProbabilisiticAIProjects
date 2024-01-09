"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import math


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
ROOT2 = math.sqrt(2)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        f_kernels = [Matern("fixed",nu=2.5),  0.5 * RBF(length_scale=10.0), 0.5 * RBF(length_scale=1.0), 0.5 * RBF(length_scale=0.5)]
        f_kernel_hyperparam = 0 #tune this to a value [0,3]
        f_kernel =  ROOT2 * RBF(length_scale=10)

        self.f_gp = GaussianProcessRegressor(kernel = f_kernel, n_restarts_optimizer=0, alpha = 0.15 ** 2)

        v_kernels = [Matern("fixed", nu=2.5), ROOT2 * RBF(length_scale=10.0), ROOT2 * RBF(length_scale=1.0), ROOT2 * RBF(length_scale=0.5)]
        v_kernel_hyperparam = 0 #tune this to a value[0,3]
        v_kernel = ConstantKernel(4, "fixed") +  ROOT2 * RBF(length_scale=0.5) #select kernel and set prior mean to 4

        self.v_gp = GaussianProcessRegressor(kernel = v_kernel, n_restarts_optimizer=0, alpha = 0.0001 ** 2)

        self.best_x = None
        self.best_f = 0

        self.first_x = None


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def upper_confidence_bound(self, x, f_beta = 2):
        f_pred, f_std = self.f_gp.predict(x.reshape(-1, 1), return_std=True)
        ucb = f_pred + f_beta * f_std
        return ucb

    def safety_bound(self, x, v_beta = 3):
        v_pred, v_std = self.v_gp.predict(x.reshape(-1, 1), return_std=True)
        lsb = v_pred + v_beta * v_std
        return lsb

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        ucb_values = self.upper_confidence_bound(x)
        lsb_values = self.safety_bound(x)
        l = 10 # cost function hyperparameter
        return ucb_values - l * np.maximum(lsb_values - 4, 0)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        if self.first_x is None:
            self.first_x = x + 0.1
        # TODO: Add the observed data {x, f, v} to your model.
        x = np.array(x).reshape(-1,1)
        f = np.array(f).reshape(-1,1)
        v = np.array(v).reshape(-1,1)
        if f > self.best_f and v < SAFETY_THRESHOLD:
            self.best_x = x
            self.best_f = f
        self.f_gp.fit(x, f)
        self.v_gp.fit(x, v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        return self.first_x

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
