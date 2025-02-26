# Physics-Informed Neural Network (PINN) vs. Traditional Neural Network (ANN)

## 📌 Introduction

In this experiment, we compare **Physics-Informed Neural Networks (PINNs)** and **Artificial Neural Networks (ANNs)** for solving the **1D Heat Equation**:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

where $u(x, t)$ represents the heat distribution, and $\alpha$ is the diffusion coefficient.

### 🔹 What is PINN?
PINNs use **both data and physics constraints** (e.g., differential equations) to improve learning.

### 🔹 What is ANN?
A traditional ANN learns purely from **data without any physics knowledge**.

### 📌 Goal
- Train both **PINN and ANN** on noisy data.
- Compare their ability to **recover the underlying solution**.

---
## 📌 Implementation

### 🔹 **Step 1: Define PINN and ANN Models**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the PINN network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        input_tensor = torch.cat((x, t), dim=1)
        return self.net(input_tensor)

# Define the traditional ANN network
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        input_tensor = torch.cat((x, t), dim=1)
        return self.net(input_tensor)
```

---
### 🔹 **Step 2: Define the Physics Loss for PINN**
```python
def physics_loss(model, x, t, alpha=0.01):
    x.requires_grad = True
    t.requires_grad = True
    
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha * u_xx  # Heat equation residual
    return torch.mean(residual**2)
```

---
### 🔹 **Step 3: Generate Noisy Training Data**
```python
N = 1000  # Number of training points
x_train = torch.rand((N, 1)) * 2 - 1  # x in [-1, 1]
t_train = torch.rand((N, 1)) * 2 - 1  # t in [-1, 1]

noise_level = 0.1
u_exact = torch.sin(torch.pi * x_train)  # True function
u_noisy = u_exact + noise_level * torch.randn_like(u_exact)  # Noisy data
```

---
### 🔹 **Step 4: Train Both Models**
```python
pinn_model = PINN()
ann_model = ANN()
optimizer_pinn = optim.Adam(pinn_model.parameters(), lr=0.01)
optimizer_ann = optim.Adam(ann_model.parameters(), lr=0.01)

epochs = 5000
for epoch in range(epochs):
    optimizer_pinn.zero_grad()
    u_pred = pinn_model(x_train, torch.zeros_like(t_train))
    loss_data = torch.mean((u_pred - u_noisy) ** 2)
    loss_physics = physics_loss(pinn_model, x_train, t_train)
    loss = loss_data + loss_physics
    loss.backward()
    optimizer_pinn.step()
    
    optimizer_ann.zero_grad()
    u_ann_pred = ann_model(x_train, torch.zeros_like(t_train))
    loss_ann = torch.mean((u_ann_pred - u_noisy) ** 2)
    loss_ann.backward()
    optimizer_ann.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, PINN Loss: {loss.item():.6f}, ANN Loss: {loss_ann.item():.6f}")
```

---
### 🔹 **Step 5: Visualizing Results**
```python
x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.zeros_like(x_test)

u_true = torch.sin(torch.pi * x_test).detach().numpy()
u_noisy_sample = u_noisy[:100].detach().numpy()
u_pinn_pred = pinn_model(x_test, t_test).detach().numpy()
u_ann_pred = ann_model(x_test, t_test).detach().numpy()

plt.figure(figsize=(10, 5))
plt.plot(x_test, u_true, label="True Solution", linestyle="dashed", color="blue")
plt.scatter(x_train[:100], u_noisy_sample, label="Noisy Data", color="gray", alpha=0.5)
plt.plot(x_test, u_pinn_pred, label="PINN Prediction", color="red")
plt.plot(x_test, u_ann_pred, label="ANN Prediction", color="green")
plt.xlabel("x")
plt.ylabel("u(x, 0)")
plt.title("Comparison of PINN, ANN, and Noisy Data")
plt.legend()
plt.grid()
plt.show()
```

---
## 📌 Results and Observations
✅ **PINN learns a smooth solution**, despite noisy data.
✅ **ANN overfits noise**, failing to recover the true function.
✅ **PINN generalizes better**, as it incorporates physical laws.

| Model | Uses Physics? | Handles Noisy Data? | Generalization |
|--------|--------------|----------------|--------------|
| **PINN** | ✅ Yes | ✅ Robust | ✅ Excellent |
| **ANN** | ❌ No | ❌ Overfits | ❌ Poor |

---
## 📌 Future Work
✅ Extend PINN to **2D Heat Equation**
✅ Apply PINNs to **Navier-Stokes (fluid dynamics)**
✅ Experiment with **real-world physics problems**


