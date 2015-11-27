#!/usr/bin/env python3

from simplebd import Simplebd
from statistics import mean, variance
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_POINTS = 200

class SimplebdStats:

    def __init__(self, kb = 0.2, kd = 0.003, x0 = 50, t0 = 0, t_end = 1000, N = 1000):
        self._kb = kb
        self._kd = kd
        self._t0 = t0
        self._t_end = t_end
        self._x0 = x0
        self._N = N


        self._collect = []
        for i in range(N):
            simplebd = Simplebd(kb, kd, t0, t_end, x0)
            simplebd.generate()
            self._collect.append(simplebd.get_data())

    def find_x_at_t(self, i, t):
        if t < self._t0 or t > self._t_end:
            raise Exception('t is out of range')

        num = len(self._collect[i])

        for j in range(num):
            if t < self._collect[i][0]['t']:
                return self._x0
            if t >= self._collect[i][j]['t'] and t < self._collect[i][j+1]['t']:
                return self._collect[i][j]['x']

    def get_distribution_at_t(self, t):
        x = []
        for i in range(self._N):
            x.append(self.find_x_at_t(i, t))
        return x

    def get_lambda(self, t):
        return self._kb / self._kd * (1 - np.exp(- self._kd * t))

    def poisson_distribution(self, lambda_, x, t):
        return np.exp(-lambda_) * (lambda_) ** x / math.factorial(x)

    def get_analytical_mean_at_t(self, t):
        return get_lambda(t)

    def get_analytical_variance_at_t(self, t):
        return get_lambda(t)

    def get_analytical_statistics(self, analytical_function):

        x = np.vectorize(analytical_function)(self.t)

        return x

    def get_stochastic_statistics(self, stochastic_statistics_function):

        x = np.vectorize(lambda v: stochastic_statistics_function(self.get_distribution_at_t(v)))(self.t)

        return x

    def get_statistics(self):

        self.t = np.linspace(self._t0, self._t_end, NUMBER_OF_POINTS)

        self.x_max_stochastic       = self.get_stochastic_statistics(max)
        self.x_min_stochastic       = self.get_stochastic_statistics(min)
        self.x_mean_analytical      = self.get_analytical_statistics(self.get_analytical_mean_at_t)
        self.x_mean_stochastic      = self.get_stochastic_statistics(mean)
        self.x_variance_stochastic  = self.get_stochastic_statistics(variance)
        self.x_variance_analytical  = self.get_analytical_statistics(self.get_analytical_variance_at_t)

    def compare_distribution(self):
        number_of_snapshots = 4

        t = np.linspace(self._t0, self._t_end, number_of_snapshots)

        for i in t:



if __name__ == '__main__':

    simplebdStats = SimplebdStats()
    simplebdStats.get_statistics()





