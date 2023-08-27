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

x = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0])
erg11 = np.dot(PvonXvorausgesetztY, x)

x2 = 1 - x
erg12 = np.dot(1 - PvonXvorausgesetztY, x2)
erg1 = erg11 * erg12

z = np.array([0, 0, 0, 1])
erg2 = np.dot(PvonYvorausgesetztZ, z)

erg3 = erg1 * erg2;
# % zum normalisieren muss ich jede Zeile von erg3 durch die summe aller
# % Zeilen von erg3 dividieren.
PvonYvorausgesetztXundZ = erg3 / sum(erg3);

"""
Takes the active input pixels x and the prior in a one-hot encoded manner and 
calculates the probability of Y given X and Z.
x: active input pixels (eg.: np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]))
z: prior (eg.: np.array([1, 0, 0, 0]))
returns PvonYvorausgesetztXundZ: Probabilities for Y given X and Z
"""
def calcPvonYvorausgesetztXundZ(x, z):

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
  erg11 = np.dot(PvonXvorausgesetztY, x)
  erg12 = np.dot(1 - PvonXvorausgesetztY, 1 - x)

  erg1 = erg11 * erg12
  erg2 = np.dot(PvonYvorausgesetztZ, z)
  erg3 = erg1 * erg2;
  # % erg3 hat die wahrscheinlichkeiten für jedes yi drinnen, aber die
  # % normalisierung fehlt noch.
  
  # % zum Vergleich ohne Prior:
  # PvonYvorausgesetztX = erg1 / sum(erg1);
  
  # % zum normalisieren muss ich jede Zeile von erg3 durch die summe aller
  # % Zeilen von erg3 dividieren.
  PvonYvorausgesetztXundZ = erg3 / sum(erg3);
  return PvonYvorausgesetztXundZ

"""
Takes the active input pixels x and the prior in a one-hot encoded manner and 
calculates the probability of Y given X and Z = null.
x: active input pixels (eg.: np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]))
z: prior (eg.: np.array([0, 0, 0, 0]))
returns PvonYvorausgesetztXundZ: Probabilities for Y given X and ZNull
"""
def calcPvonYvorausgesetztXundZNull(x, z):

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
  erg11 = np.dot(PvonXvorausgesetztY, x)
  erg12 = np.dot(1 - PvonXvorausgesetztY, 1 - x)

  erg1 = erg11 * erg12
  # erg2 = np.dot(PvonYvorausgesetztZ, z)
  erg3 = erg1;
  # % erg3 hat die wahrscheinlichkeiten für jedes yi drinnen, aber die
  # % normalisierung fehlt noch.
  
  # % zum Vergleich ohne Prior:
  # PvonYvorausgesetztX = erg1 / sum(erg1);
  
  # % zum normalisieren muss ich jede Zeile von erg3 durch die summe aller
  # % Zeilen von erg3 dividieren.
  PvonYvorausgesetztXundZ = erg3 / sum(erg3);
  return PvonYvorausgesetztXundZ
  


