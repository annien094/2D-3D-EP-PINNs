# Physics-Informed Neural Networks can accurately model cardiac electrophysiology in 3D geometries and fibrillatory conditions (STACOM 2024)
Accepted to be published in the Statistical Atlases and Computational Modeling of the Heart Workshop 2024, a MICCAI satellite workshop.
This is the repository holding all code used in the article.

### Abstract
Physics-Informed Neural Networks (PINNs) are fast becoming an important tool to solve differential equations rapidly and accurately, and to identify the systems parameters that best agree with a given set of measurements. PINNs have been used for cardiac electrophysiology (EP), but only in simple 1D and 2D geometries and for sinus rhythm or single rotor dynamics. Here, we demonstrate how PINNs can be used to accurately reconstruct the propagation of cardiac action potential in more complex geometries and dynamical regimes, including 3D spherical geometries and spiral break-up conditions that model cardiac fibrillation.

We also demonstrate that PINNs can be used to estimate EP parameters in cardiac EP models with some biological detail. We estimate the diffusion coefficient and parameters related to ion channel conductances in the Fenton-Karma model in a 2D setup. Our results are an important step towards the deployment of PINNs to realistic cardiac geometries and arrhythmic conditions.   


## Running the code
The repository is structured by the different geometry/electrophysiological models. In each folder, run the `main*` files. There are also auxillirary files `utils.py` and data files that are needed for the training.

The numerical simulation code (FEM or FD) to generate sythetic data is in Data_generation.
