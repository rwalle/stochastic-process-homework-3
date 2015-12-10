#!/usr/bin/env python3

from random import random
from math import log

class Simplebd:

    def __init__(self, kb, kd, t0, t_end, x0):

        """Assigning values

        kb    : birth rate
        kd    : death rate
        t0    : start time
        t_end : stop time
        x0    : initial population
        """

        self._kb    = kb
        self._kd    = kd
        self._t0    = t0
        self._t_end = t_end
        self._x0    = x0

    def _get_a(self, x):

        """Returning 'a' values

        x: subscript
        """

        a1 = self._kb
        a2 = self._kd * x
        return [a1 + a2, a1, a2]

    def _delta_x(self, mu):

        """Decide whether x plus one or minus one based on value of mu

        mu : number of reaction happened
        """

        if mu != 1 and mu != 2:
            raise ValueError('Wrong mu value')

        if mu == 1:
            return 1
        else:
            return -1

    def generate(self):

        """generate a path
        """

        t = self._t0
        x = self._x0
        self._data = []

        while t < self._t_end:
            n1 = random()
            n2 = random()

            a   = self._get_a(x)
            tau = 1 / a[0] * log(1 / n1)
            mu  = 0
            sa  = 0

            while sa < n2 * a[0]:
                mu += 1
                sa = sa + a[mu]

            t += tau
            x += self._delta_x(mu)

            self._data.append({'t': t, 'x': x})

    def get_data(self):
        return self._data
