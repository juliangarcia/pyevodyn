pyevodyn
========

Introduction
+++++++++++++

Evolutionary Dynamics with Python . This project is focused on implementing numerical routines to teach and study evolutionary dynamics. The main focus so far is on the Moran process, but I intend to work more on standard deterministic dynamics (e.g., replicator) as well as other stochastic processes.


Modules
++++++++

There is an experimental module for stochastic simulations, but as of now the performance is rather prohibitive for real studies.

There is an analytical module, that uses sympy to facilitate calculations with symbolic variables.

The most mature module relies on NumPy to numerically compute fixation probabilities and stationary distributions in a Moran process. Pairwise invasion analysis can also be done via numerical methods in symmetric 2-player games.

Future 
++++++++

* A C extension needs to be developed in order to support simulation with decent performance.
* Multiprocessing support for machines with several cores
* Analysis of deterministic dynamics



