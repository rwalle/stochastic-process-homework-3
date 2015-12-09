#!/usr/bin/env python3

from simplebd import Simplebd
from statistics import mean, variance
from collections import Counter
from scipy.stats import poisson, skew
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NUMBER_OF_POINTS = 200

class SimplebdStats:

    def __init__(self, kb = 0.2, kd = 0.004, x0 = 30, t0 = 0, t_end = 2000, N = 1000):
        self._kb = kb
        self._kd = kd
        self._t0 = t0
        self._t_end = t_end
        self._x0 = x0
        self._N = N
        self._distribution_at_t = {}


        self._collect = []
        for i in range(N):
            simplebd = Simplebd(kb, kd, t0, t_end, x0)
            simplebd.generate()
            self._collect.append(simplebd.get_data())

    def find_x_at_t(self, i, t):
        if t < self._t0 or t > self._t_end:
            raise Exception('t is out of range')

        num = len(self._collect[i])

        if t < self._collect[i][0]['t']:
            return self._x0

        for j in range(num - 1):
            if t >= self._collect[i][j]['t'] and t < self._collect[i][j + 1]['t']:
                return self._collect[i][j]['x']

        return self._collect[i][num - 1]['x']

    def get_first_passage(self, theta):

        c = []

        for i in range(self._N):
            num = len(self._collect[i])

            for j in range(num):
                if self._collect[i][j]['x'] >= theta:
                    c.append(self._collect[i][j]['t'])
                    break

        return c





    def get_distribution_at_t(self, t):
        if t in self._distribution_at_t:
            return self._distribution_at_t[t]
        else:
            x = []
            for i in range(self._N):
                x.append(self.find_x_at_t(i, t))
            self._distribution_at_t[t] = x
            return x

    def get_lambda(self, t):
        return self._kb / self._kd * (1 - np.exp(- self._kd * t))

    def get_probability(self, x, t):
        lambda_ = self.get_lambda(t)
        return lambda_ ** x / np.math.factorial(x) * np.exp(- lambda_)

    def poisson_distribution(self, lambda_, x):
        return poisson.pmf(x, lambda_)

    def get_analytical_corr_at_t(self, t, tau):
        return self.get_analytical_variance_at_t(t) * np.exp(-self._kd * tau)

    def get_analytical_mean_at_t(self, t):
        return self._kb / self._kd + (self._x0 - self._kb / self._kd) * np.exp(- self._kd * t)

    def get_analytical_variance_at_t(self, t):
        return -self._x0 * np.exp(-2*self._kd*t) + self._kb / self._kd * (1 - np.exp(-self._kd*t)) + np.exp(-self._kd*t)*self._x0

    def get_analytical_x2_at_t(self, t):
        return np.exp(- 2 * self._kd * t) * (self._kb ** 2 / self._kd ** 2 - 2 * self._kb / self._kd * self._x0 - self._x0 + self._x0 ** 2) + \
                np.exp(- self._kd * t) * (- 2 * self._kb ** 2 / self._kd ** 2 - self._kb / self._kd + 2 * self._kb * self._x0 / self._kd + self._x0) + \
                1 / self._kd ** 2 * (self._kb ** 2 + self._kb * self._kd)

    def get_analytical_x3_at_t(self, t):
        kb = self._kb
        kd = self._kd
        x0 = self._x0
        return np.exp(-3 * kd * t) / kd ** 3 * (-kb ** 3 + 3 * kb ** 2 * kd * x0 + 3 * kd ** 2 * kb * x0 + 2 * kd ** 3 * x0 - 3 * kb * kd ** 2 * x0 ** 2 - 3 * kd ** 3 * x0 ** 2 + kd ** 3 * x0 ** 3) + \
                np.exp(-2 * kd * t) / kd ** 3 * (3 * kb ** 3 + 3 * kb **2 * kd - 6 * kb ** 2 * kd * x0 - 9 * kb * kd ** 2 * x0 - 3 * kd ** 3 * x0 + 3 * kb * kd ** 2 * x0 ** 2 + 3 * kd ** 3 * x0 ** 2) + \
                np.exp(-kd * t) / kd ** 3 * (-3 * kb ** 3 - 6 * kb ** 2 * kd - kb * kd ** 2 + 3 * kb ** 2 * kd * x0 + 6 * kb * kd ** 2 * x0 + kd ** 3 * x0) + \
                1 / kd ** 3 * (3 * kb ** 2 * kd + kb ** 3 + kb * kd ** 2)

    def get_analytical_skewness_at_t(self, t):
        return (self.get_analytical_x3_at_t(t) - 3 * self.get_analytical_mean_at_t(t) * self.get_analytical_variance_at_t(t) - self.get_analytical_mean_at_t(t) ** 3) / self.get_analytical_variance_at_t(t) ** 1.5

    def get_simulative_x2_at_t(self, a):
        return np.mean(np.array(a) ** 2)

    def get_simulative_x3_at_t(self, a):
        return np.mean(np.array(a) ** 3)

    def get_analytical_statistics(self, analytical_function):

        x = np.vectorize(analytical_function)(self.t)

        return x

    def get_simulative_statistics(self, stochastic_statistics_function):

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

    def compare(self, analytical_function, simulative_function, ylabel_text):
        self.t = np.linspace(self._t0, self._t_end, NUMBER_OF_POINTS)
        x_mean_analytical = self.get_analytical_statistics(analytical_function)
        x_mean_simulative = self.get_simulative_statistics(simulative_function)
        plt.plot(self.t, x_mean_analytical, color = 'blue')
        plt.plot(self.t, x_mean_simulative, color = 'red')
        red = mpatches.Patch(color = 'red', label = 'Simulation')
        blue = mpatches.Patch(color = 'blue', label = 'Analytical Prediction')
        plt.legend(handles = [blue, red])
        plt.xlabel('t')
        plt.ylabel(ylabel_text)
        plt.show(block = False)

    def compare_mean(self):
        return self.compare(self.get_analytical_mean_at_t, mean, 'mean')

    def compare_variance(self):
        return self.compare(self.get_analytical_variance_at_t, variance, 'variance')

    def compare_skewness(self):
        return self.compare(self.get_analytical_skewness_at_t, skew, 'skewness')

    def compare_x2(self):
        return self.compare(self.get_analytical_x2_at_t, self.get_simulative_x2_at_t, 'x2')

    def compare_x3(self):
        return self.compare(self.get_analytical_x3_at_t, self.get_simulative_x3_at_t, 'x3')

    def compare_distribution(self):
        number_of_snapshots = 4

        snapshots = np.linspace(self._t0 + (self._t_end - self._t0) / number_of_snapshots,
                self._t_end, number_of_snapshots)

        for t in snapshots:
            stochastic_distribution = np.sort(self.get_distribution_at_t(t))
            count = Counter(stochastic_distribution)
            x_stochastic = np.asarray(list(count.keys()))
            p_stochastic = np.vectorize(lambda x: x / self._N)(list(count.values()))
            x_analytical = np.arange(min(x_stochastic), max(x_stochastic), 1)
            p_analytical = np.vectorize(self.poisson_distribution)(self.get_lambda(t),
                    x_analytical)
            plt.plot(x_stochastic, p_stochastic)
            plt.plot(x_analytical, p_analytical)
            plt.xlabel("t")
            plt.ylabel("x")
            plt.show()

    def correlation(self):
        t_start = self._t0 + (self._t_end - self._t0) / 4
        t_end = self._t0 + (self._t_end - self._t0) * 3 / 4
        dis_0 = self.get_distribution_at_t(t_start)
        t = np.linspace(t_start, t_end, NUMBER_OF_POINTS)
        corr_stochastic = np.vectorize(lambda x: np.cov(dis_0, self.get_distribution_at_t(x))[0][1])(t)
        corr_analytical = np.vectorize(self.get_analytical_corr_at_t)(t_start, t - t_start)
        #print(t)
        #print(corr_stochastic)
        #print(corr_analytical)
        plt.plot(t, corr_stochastic)
        plt.plot(t, corr_analytical)
        plt.show()

    def get_analytical_first_passage_dist(self, theta = 45):
        t = np.linspace(self._t0, self._t_end, self._N)
        p0 = []
        for t0 in t:
            s = 0
            for x in range(theta + 1):
                s += self.get_probability(x, t0)
            p0.append(s)
        t_axis = []
        p_axis = []

        for idx, t1 in enumerate(t[1:-1]):
            t_axis.append(t1)
            dp = - (p0[idx + 1] - p0[idx - 1]) / (t[idx + 1] - t[idx - 1])
            p_axis.append(dp)

        return t_axis, p_axis






    def compare_first_passage_dist(self, theta = 45):
        dist = self.get_first_passage(theta)

        #self.simulative_distribution = np.sort(dist)
        plt.hist(dist, 20, normed = True)
#        self.count = Counter(self.simulative_distribution)
#        self.x_simulative = np.asarray(list(self.count.keys()))
#        self.p_simulative = np.vectorize(lambda x: x / self._N)(list(self.count.values()))
        x_analytical, p_analytical = self.get_analytical_first_passage_dist()
#        plt.plot(self.x_simulative, self.p_simulative)
        plt.plot(x_analytical, p_analytical)
        plt.show()






if __name__ == '__main__':

    simplebdStats = SimplebdStats()
    simplebdStats.get_statistics()





