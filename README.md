# Feedback Loop Simulations in Recommender Systems

This repository contains the simulation code developed for the experimental part of my Master's Thesis, *Between Feedback and Feedback Loops: Coupled Dynamics in Recommender Systems*.

The code implements a synthetic recommender-system environment to study how different feedback loop mechanisms behave in isolation and how their effects change when several mechanisms operate in a coupled manner.

## Repository Structure

```text
.
├── funciones_utiles.py
├── compilacion.ipynb
└── compilacion_3g.ipynb
```

- `funciones_utiles.py`: auxiliary module with shared functions for user creation, homophily-based replacement, logistic-regression training, error computation, outcome simulation, and plotting.

- `compilacion.ipynb`: main notebook containing the core simulation experiments, including the isolated feedback loops, the coupled feedback-loop configurations, and robustness checks for the sampling–ML model coupling.

- `compilacion_3g.ipynb`: additional robustness notebook for the sampling–ML model coupling under a modified population structure with three groups, including the corresponding isolated cases used for comparison.

## Feedback Loops

The simulations focus on coupled feedback-loop configurations, using the corresponding isolated cases as baselines for comparison.

The isolated baselines include:

- Sampling feedback loop
- ML-model feedback loop
- Individual feedback loop
- Outcome feedback loop

The coupled configurations include:

- Sampling–ML model coupling
- Sampling–individual coupling
- Individual–ML model coupling
- Outcome–individual coupling

## Requirements

The code is written in Python and uses the following main libraries:

```text
numpy
scipy
scikit-learn
matplotlib
jupyter
```

The notebooks should be run from the repository root so that `funciones_utiles.py` can be imported correctly.

## Notes

The simulations are stochastic, so repeated executions may produce slightly different results.

The file `compilacion_3g.ipynb` is kept separate because it uses a modified population structure for the three-group robustness analysis.

## Purpose

The purpose of this repository is to make the simulation code used in the thesis available for consultation and reproducibility.
