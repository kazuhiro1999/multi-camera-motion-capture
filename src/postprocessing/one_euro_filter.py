import math
import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev
    
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.initialized = False

    def apply(self, t, x):
        """Compute the filtered signal."""
        t /= 1000  # unixtime millisecond to second
        
        if not self.initialized:
            self.x_prev = np.asarray(x, dtype=float)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = float(t)
            self.initialized = True
            return self.x_prev

        x = np.asarray(x, dtype=float)
        t_e = t - self.t_prev

        dx = (x - self.x_prev) / t_e
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat