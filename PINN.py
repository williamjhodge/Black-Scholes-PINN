import torch
import torch.nn as nn
import numpy as np
from Model import PINN



# Physics

def pde_residual(model, S, t, r, sigma):
    V = model(S, t)
    dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]
    dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    residual = dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V
    return torch.mean(residual**2)

def boundary_loss_expiry(model, K, T, S_max, n_points=200): # boundary condition at t = T (expiry)
    S_boundary = torch.rand(n_points, 1) * S_max
    t_boundary = torch.full_like(S_boundary, T)
    V_pred = model(S_boundary, t_boundary)
    V_true = torch.clamp(S_boundary - K, min=0)
    return torch.mean((V_pred - V_true)**2)

def boundary_loss_far_field(model, K, T, S_max, n_points=200): # boundary condition at S = S_max (far-field)
    S_boundary = torch.full((n_points, 1), S_max)
    t_boundary = torch.rand(n_points, 1) * T
    V_pred = model(S_boundary, t_boundary)
    V_true = S_max - K * torch.exp(-0.05 * (T - t_boundary))  # Black-Scholes far-field condition
    return torch.mean((V_pred - V_true)**2)


# Training 
r, sigma, K, T, S_max = 0.05, 0.2, 100, 1.0, 200
n_epochs = 10000

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iteratively based on training data.

for epoch in range(n_epochs):
    S = (torch.rand(1000, 1) * S_max).requires_grad_(True)
    t = (torch.rand(1000, 1) * T).requires_grad_(True)

    loss_pde = pde_residual(model, S, t, r, sigma)
    loss_boundary = boundary_loss_expiry(model, K, T, S_max) + boundary_loss_far_field(model, K, T, S_max)
    loss = loss_pde + loss_boundary

    optimizer.zero_grad() #clears old gradients from the last step (otherwise they will accumulate)
    loss.backward() #computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation
    optimizer.step() #updates the value of the parameters based on the current gradient (stored in .grad attribute of a parameter) and the learning rate

    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.6f}")

torch.save(model.state_dict(), "pinn_model.pt")