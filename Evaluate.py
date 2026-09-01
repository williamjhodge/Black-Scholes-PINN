import torch
import scipy.stats 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from Model import PINN

def black_scholes_call(S, t, K, T, r, sigma):
    tau = T - t   # time remaining until expiry
    tau = np.maximum(tau, 1e-6)   # avoid divide-by-zero exactly at expiry
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    price = S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    return price


model = PINN()
model.load_state_dict(torch.load("pinn_model.pt"))
model.eval()   # switches model to evaluation mode

K, T, r, sigma, S_max = 100, 1.0, 0.05, 0.2, 200


S_vals = np.linspace(1, S_max, 100)
t_vals = np.linspace(0, T, 100)
S_grid, t_grid = np.meshgrid(S_vals, t_vals)

bs_prices = black_scholes_call(S_grid, t_grid, K, T, r, sigma)

S_tensor = torch.tensor(S_grid.flatten(), dtype=torch.float32).unsqueeze(1)
t_tensor = torch.tensor(t_grid.flatten(), dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    pinn_prices = model(S_tensor, t_tensor).numpy().reshape(S_grid.shape)

error = np.abs(pinn_prices - bs_prices)
print(f"Mean absolute error: {error.mean():.4f}")
print(f"Max absolute error: {error.max():.4f}")
max_idx = np.unravel_index(np.argmax(error), error.shape)
print(f"Max error at S={S_grid[max_idx]:.2f}, t={t_grid[max_idx]:.2f}") #maximum error occurs at S=200, t=0.00 consistent with PINNs' known difficulty propagating boundary information to distant regions of the domain without additional anchoring conditions — suggesting that adding a second boundary condition at S_max would likely improve accuracy. This has been implemented in the updated PINN.py file with the boundary_loss_far_field function. It reduced the maximum error from 18.5 to 2.4, however this is still located at the far-field boundary.

plt.figure(figsize=(8, 6))
plt.contourf(S_grid, t_grid, error, levels=50, cmap="viridis")
plt.colorbar(label="Absolute error")
plt.xlabel("Stock price S")
plt.ylabel("Time t")
plt.title("PINN vs Black-Scholes: absolute error")
plt.show()