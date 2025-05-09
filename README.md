# PINN Solver for 1D Unsteady Advection Equation

This repository contains the implementation of a Physics-Informed Neural Network (PINN) to solve the 1D unsteady advection-diffusion equation. It was developed as part of the seminar paper:
"Machine Learning and Derivative-Based Nonlinear Optimization: A Dual Perspective"
RWTH Aachen University, Summer Semester 2025 — Zeyad Ibrahim

## Objective

To train a PINN using PyTorch that satisfies:
- Initial condition
- Dirichlet boundary conditions
- The PDE residual of the advection-diffusion equation

All of this is done without supervised data, relying purely on physics constraints embedded into the loss function.

## Model Summary

- Fully connected neural network with Tanh activations
- Input: (t, x) pairs
- Output: scalar field q(t, x)
- Trained using:
  - Adam optimizer (5000 epochs)
  - L-BFGS optimizer for fine convergence
- Loss function:
  Loss = 10 * IC_Loss + 10 * BC_Loss + PDE_Residual_Loss

## Requirements

To install dependencies:

pip install torch numpy matplotlib scipy

Or using the provided requirements file:

pip install -r requirements.txt

## How to Run

Option 1: Run Python script directly

python 1d_unsteady_advection_pinn.py

This will:
- Train the model
- Evaluate it against finite-difference (FD) and analytical solutions
- Print L2 and L∞ error metrics
- Generate visualizations (error heatmaps, 3D surfaces, slices)

## Output Summary

You will get:
- Training logs (loss values)
- Learned solution plot q(t, x)
- Absolute error heatmaps compared to:
  - Analytical solution
  - Finite difference method (FDM)
- Time slices and spatial traces at various fixed points
- Comparative PDE residual maps (PINN vs. FD vs. analytic)

## Files

- 1d_unsteady_advection_pinn.py — Main script (converted from notebook)
- 1D_Unsteady_Advection_PINN.ipynb — Original Colab notebook (optional)
- README.md — This guide
- requirements.txt — Dependency list (optional)

## Reproducibility

All results in the seminar report are directly reproducible using this codebase. No external datasets are required. You can change v and D in the script to explore other advection-diffusion regimes.

## Contact

If you have any questions, please contact:
zeyad.ibrahim@rwth-aachen.de
Simulation Sciences — RWTH Aachen University
