pyevodyn
========

Evolutionary Dynamics with Python . This project is focused on implementing numerical routines to teach and study evolutionary dynamics.

The main focus so far is on the Moran process, but I intend to work more on standard deterministic dynamics (e.g., replicator) as well as other stochastic processes.

There is an experimental module for stochastic simulations, but as of now the performance is rather prohibitive for real studies.

There is an analytical module, that uses sympy to facilitate calculations with symbolic variables.

The most mature modules uses NumPy to numerically compute fixation probabilities and stationary distributions in a Moran process. Pairwise invasion analysis can also be done via numerical methods.

