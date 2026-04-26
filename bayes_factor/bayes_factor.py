import math
import scipy.integrate


class BayesFactor:
    def __init__(self, n, k, spike_a=0.4999, spike_b=0.5001):
        
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        # range
        if n < 0:
            raise ValueError("n must be non-negative")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > n:
            raise ValueError("k cannot be greater than n")

        # interval checks
        if spike_a < 0 or spike_b > 1:
            raise ValueError("spike prior bounds must be between 0 and 1")
        if spike_a >= spike_b:
            raise ValueError("spike_a must be less than spike_b")

        self.n = n
        self.k = k
        self.spike_a = spike_a
        self.spike_b = spike_b

    def likelihood(self, theta):
        if not isinstance(theta, (int, float)):
            raise TypeError("theta must be a number")
        if theta < 0 or theta > 1:
            raise ValueError("theta must be between 0 and 1")

        comb = math.comb(self.n, self.k)
        return comb * (theta ** self.k) * ((1 - theta) ** (self.n - self.k))

    def evidence_slab(self):
        result, error = scipy.integrate.quad(lambda x: self.likelihood(x), 0, 1)
        if result < 0:
            raise ArithmeticError("slab evidence cannot be negative")
        return result

    def evidence_spike(self):
        width = self.spike_b - self.spike_a
        if width <= 0:
            raise ValueError("spike prior width must be positive")

        result, error = scipy.integrate.quad(
            lambda x: self.likelihood(x) * (1 / width),
            self.spike_a,
            self.spike_b
        )
        if result < 0:
            raise ArithmeticError("spike evidence cannot be negative")
        return result

    def bayes_factor(self):
        slab = self.evidence_slab()
        spike = self.evidence_spike()

        if slab == 0:
            raise ZeroDivisionError("slab evidence is zero")

        return spike / slab
