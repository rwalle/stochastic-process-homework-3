#!/usr/bin/env python3

from simplebd import Simplebd
from statistics import mean, variance
from scipy.stats import poisson, skew
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SimplebdStats:

    def __init__(self, kb = 0.2, kd = 0.004, x0 = 1, t0 = 0, t_end = 2000, NUMBER_OF_PATHS = 1000):

        """Assigning values
        kb: birth rate
        kd: death rate
        x0: initial population
        t0: start time
        t_end: stop time
        """

        self._kb = kb
        self._kd = kd
        self._t0 = t0
        self._t_end = t_end
        self._x0 = x0

        self._NUMBER_OF_PATHS = NUMBER_OF_PATHS
        self._NUMBER_OF_MOMENTS = 200

        self._HIST_LAGS = 20

        self._distribution_at_t = {} # used for caching

        self._data = []

        # generate paths
        for i in range(self._NUMBER_OF_PATHS):
            simplebd = Simplebd(kb, kd, t0, t_end, x0)
            simplebd.generate()
            self._data.append(simplebd.get_data())

    def find_x_at_t(self, i, t):

        """
        Given a time t, find out value of x for path i
        """

        if t < self._t0 or t > self._t_end:
            raise ValueError('t is out of range')

        if i < 0 or i > self._NUMBER_OF_PATHS:
            raise ValueError('i is out of range')

        num = len(self._data[i])

        if t < self._data[i][0]['t']:
            return self._x0

        for j in range(num - 1):
            if t >= self._data[i][j]['t'] and t < self._data[i][j + 1]['t']:
                return self._data[i][j]['x']

        return self._data[i][num - 1]['x']

    def get_distribution_at_t(self, t):

        """
        Return a collection of values of x at time t
        """

        if t < self._t0 or t > self._t_end:
            raise ValueError('t is out of range')

        #_distribution_at_t seves as cache
        if t in self._distribution_at_t:
            return self._distribution_at_t[t]
        else:
            x = []
            for i in range(self._NUMBER_OF_PATHS):
                x.append(self.find_x_at_t(i, t))
            self._distribution_at_t[t] = x
            return x

    def get_lambda(self, t):

        """
        Return the value of lambda at given time t
        """

        return self._kb / self._kd * (1 - np.exp(- self._kd * t))

    def get_probability(self, x, t):

        """
        Return the probability P(x, t)
        """

        lambda_ = self.get_lambda(t)
        return lambda_ ** x / np.math.factorial(x) * np.exp(- lambda_)

    def poisson_distribution(self, lambda_, x):

        """
        Return value of poisson distribution at x with lambda = lambda_
        """

        return poisson.pmf(x, lambda_)

    def get_analytical_correlation_at_t(self, t, tau):

        """
        Return analytical correlation of time t and t + tau
        """

        return self.get_analytical_variance_at_t(t) * np.exp(-self._kd * tau)

    def get_analytical_mean_at_t(self, t):

        """
        Return analytical ensemble mean at time t
        """

        return self._kb / self._kd + (self._x0 - self._kb / self._kd) * np.exp(- self._kd * t)

    def get_analytical_variance_at_t(self, t):

        """
         Return analytical ensemble variance at time t
        """

        return -self._x0 * np.exp(-2 * self._kd * t) + self._kb / self._kd * (1 - np.exp(-self._kd * t)) + np.exp(-self._kd * t) * self._x0

    def get_analytical_x2_at_t(self, t):

        """
        Return analytical <x^2> at time t
        """

        return np.exp(- 2 * self._kd * t) * (self._kb ** 2 / self._kd ** 2 - 2 * self._kb / self._kd * self._x0 - self._x0 + self._x0 ** 2) + \
                np.exp(- self._kd * t) * (- 2 * self._kb ** 2 / self._kd ** 2 - self._kb / self._kd + 2 * self._kb * self._x0 / self._kd + self._x0) + \
                1 / self._kd ** 2 * (self._kb ** 2 + self._kb * self._kd)

    def get_analytical_x3_at_t(self, t):

        """
        Return analytical <x^3> at time t
        """

        kb = self._kb
        kd = self._kd
        x0 = self._x0
        return np.exp(-3 * kd * t) / kd ** 3 * (-kb ** 3 + 3 * kb ** 2 * kd * x0 + 3 * kd ** 2 * kb * x0 + 2 * kd ** 3 * x0 - 3 * kb * kd ** 2 * x0 ** 2 - 3 * kd ** 3 * x0 ** 2 + kd ** 3 * x0 ** 3) + \
                np.exp(-2 * kd * t) / kd ** 3 * (3 * kb ** 3 + 3 * kb **2 * kd - 6 * kb ** 2 * kd * x0 - 9 * kb * kd ** 2 * x0 - 3 * kd ** 3 * x0 + 3 * kb * kd ** 2 * x0 ** 2 + 3 * kd ** 3 * x0 ** 2) + \
                np.exp(-kd * t) / kd ** 3 * (-3 * kb ** 3 - 6 * kb ** 2 * kd - kb * kd ** 2 + 3 * kb ** 2 * kd * x0 + 6 * kb * kd ** 2 * x0 + kd ** 3 * x0) + \
                1 / kd ** 3 * (3 * kb ** 2 * kd + kb ** 3 + kb * kd ** 2)

    def get_analytical_skewness_at_t(self, t):

        """
        Return analytical skewness at time t
        """

        return (self.get_analytical_x3_at_t(t) - 3 * self.get_analytical_mean_at_t(t) * self.get_analytical_variance_at_t(t) - self.get_analytical_mean_at_t(t) ** 3) / self.get_analytical_variance_at_t(t) ** 1.5

    def get_simulative_x2_at_t(self, a):

        """
        Return simulative <x^2> at time t
        """

        return np.mean(np.array(a) ** 2)

    def get_simulative_x3_at_t(self, a):

        """
        Return simulative <x^3> at time t
        """

        return np.mean(np.array(a) ** 3)

    def get_analytical_statistics(self, analytical_function):

        x = np.vectorize(analytical_function)(self.t)

        return x

    def get_simulative_statistics(self, simulative_statistics_function):

        x = np.vectorize(lambda v: simulative_statistics_function(self.get_distribution_at_t(v)))(self.t)

        return x

    def single_path(self):
        data = self._data[0]

        t = []
        x = []

        for i in range(len(data)):
            t.append(data[i]['t'])
            x.append(data[i]['x'])

        plt.plot(t, x)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.show(block = False)



    def compare(self, analytical_function, simulative_function, ylabel_text):

        """
        Compare analytical and simulative result defined by function analytical_function and simulative_function at time t
        and plot figure
        """

        self.t = np.linspace(self._t0, self._t_end, self._NUMBER_OF_PATHS)

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
        number_of_snapshots = 5

        snapshots = np.linspace(self._t0 + (self._t_end - self._t0) / number_of_snapshots,
                self._t_end, number_of_snapshots)

        for t in snapshots:

            simulative_distribution = np.sort(self.get_distribution_at_t(t))
            x_analytical = np.arange(min(simulative_distribution), max(simulative_distribution), 1)
            p_analytical = np.vectorize(self.poisson_distribution)(self.get_lambda(t),
                    x_analytical)

            # plot
            plt.hist(simulative_distribution, self._HIST_LAGS, normed = True, color = 'red')
            plt.plot(x_analytical, p_analytical, color = 'blue')

            red = mpatches.Patch(color = 'red', label = 'Simulation')
            blue = mpatches.Patch(color = 'blue', label = 'Analytical Prediction')
            plt.legend(handles = [blue, red])

            plt.xlabel("x")
            plt.ylabel("p")
            plt.title("t = " + str(t))
            plt.show()

    def compare_time_average(self):

        tau = 1 / self._kd
        t1 = tau * 6
        t2 = tau * 205
        single_path = SimplebdStats(kb = self._kb, kd = self._kd, x0 = self._x0, t0 = self._t0, t_end = t2, NUMBER_OF_PATHS = 1)
        t = np.linspace(t1, t2, 200)
        x_single_path = np.vectorize(single_path.find_x_at_t)(0, t)
        x_time_average = np.mean(x_single_path)
        x_ensemble = self.get_distribution_at_t(t1)
        x_ensemble_average = np.mean(x_ensemble)
        print("Time Average is: " + str(x_time_average))
        print("Ensemble Average is: " + str(x_ensemble_average))
        print("\n")


    def compare_autocorrelation(self):

        t_start = self._t0 + (self._t_end - self._t0) / 4
        t_end = self._t0 + (self._t_end - self._t0) * 3 / 4
        dis_0 = self.get_distribution_at_t(t_start)
        t = np.linspace(t_start, t_end, self._NUMBER_OF_MOMENTS)

        corr_simulative = np.vectorize(lambda x: np.cov(dis_0, self.get_distribution_at_t(x))[0][1])(t)
        corr_analytical = np.vectorize(self.get_analytical_correlation_at_t)(t_start, t - t_start)

        # plot
        plt.plot(t - t_start, corr_simulative, color = 'red')
        plt.plot(t - t_start, corr_analytical, color = 'blue')
        red = mpatches.Patch(color = 'red', label = 'Simulation')
        blue = mpatches.Patch(color = 'blue', label = 'Analytical Prediction')
        plt.legend(handles = [blue, red])
        plt.xlabel(r'$\tau$')
        plt.ylabel("autocorrelation")

        plt.show()

    def get_analytical_first_passage_distribution(self, theta = 45):

        t = np.linspace(self._t0, self._t_end, self._NUMBER_OF_MOMENTS)
        p0 = []
        for t0 in t:
            s = 0
            for x in range(theta + 1):
                s += self.get_probability(x, t0)
            p0.append(s)
        t_axis = []
        p_axis = []

        # use difference as differential. f'(x) = (f(x+dx)-f(x-dx))/(2dx)
        for idx in range(1, len(t) - 2):
            t_axis.append(t[idx])
            dp = - (p0[idx + 1] - p0[idx - 1]) / (t[idx + 1] - t[idx - 1])
            p_axis.append(dp)

        return t_axis, p_axis

    def get_simulative_first_passage_distribution(self, theta):

        c = []

        for i in range(self._NUMBER_OF_PATHS):
            num = len(self._data[i])

            for j in range(num):
                if self._data[i][j]['x'] >= theta:
                    c.append(self._data[i][j]['t'])
                    break

        return c

    def compare_first_passage_dist(self, theta = 45):

        """
        theta is the threshold
        """

        dist = self.get_simulative_first_passage_distribution(theta)

        x_analytical, p_analytical = self.get_analytical_first_passage_distribution()

        plt.hist(dist, self._HIST_LAGS, normed = True, color = 'red')
        plt.plot(x_analytical, p_analytical, color = 'blue')
        red = mpatches.Patch(color = 'red', label = 'Simulation')
        blue = mpatches.Patch(color = 'blue', label = 'Analytical Prediction')
        plt.legend(handles = [blue, red])
        plt.show()






if __name__ == '__main__':

    simplebdStats = SimplebdStats(kb = 0.2, kd = 0.004, x0 = 2, t0 = 0, t_end = 2000)

    input("0. A Single Path. Press enter to continue\n")
    simplebdStats.single_path()

    input("3. Comparison between analytical and simulative result for mean. Press enter to continue\n")
    simplebdStats.compare_mean()

    input("3. Comparison between analytical and simulative result for variance. Press enter to continue\n")
    simplebdStats.compare_variance()

    input("3. Comparison between analytical and simulative result for skewness. Press enter to continue\n")
    simplebdStats.compare_skewness()

    input("4. Comparison between analytical and simulative result for distribution at 5 different times. Press enter to continue\n")
    simplebdStats.compare_distribution()

    input("5. Time average equals ensemble average. Press enter to continue.")
    simplebdStats.compare_time_average()

    t_start = simplebdStats._t0 + (simplebdStats._t_end - simplebdStats._t0) / 4
    input("6. Comparison between analytical and simulative result for correlation function C(t, t + tau) where t = " + str(t_start) + ". Press enter to continue\n")
    simplebdStats.compare_autocorrelation()

    input("7. Comparison between analytical and simulative result for first time passage at threshold 45. Press enter to continue")
    simplebdStats.compare_first_passage_dist()

