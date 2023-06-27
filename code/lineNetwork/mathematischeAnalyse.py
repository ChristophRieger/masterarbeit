# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:53:10 2023

@author: chris
"""

import numpy as np

PvonXvorausgesetztY = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]],
                      "float64")
PvonYvorausgesetztZ = np.array([[0.9, 0.0333, 0.0333, 0.0333],
                         [0.0333, 0.9, 0.0333, 0.0333],
                         [0.0333, 0.0333, 0.9, 0.0333],
                         [0.0333, 0.0333, 0.0333, 0.9]],
                      "float64")



x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
erg1 = np.dot(PvonXvorausgesetztY, x)

z = np.array([1, 0, 0, 0])
erg2 = np.dot(PvonYvorausgesetztZ, z)

erg3 = erg1 * erg2;
# % erg3 hat die wahrscheinlichkeiten für jedes yi drinnen, aber die
# % normalisierung fehlt noch.

# % zum normalisieren muss ich jede Zeile von erg3 durch die summe aller
# % Zeilen von erg3 dividieren.
PvonYvorausgesetztXundZ = erg3 / sum(erg3);

# % zum Vergleich ohne Prior:
PvonYvorausgesetztX = erg1 / sum(erg1);

