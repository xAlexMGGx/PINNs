import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# ---------------- NN Architecture ----------------
class Model(nn.Module):
    def __init__(self, layers: list):
        """
        Initialize the PINN model.

        Args:
            layers: List of integers representing the number of neurons in each layer.
        """
        super(Model, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"activation_{i}", nn.SiLU())

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the PINN.
        Combines spatial and temporal inputs and passes them through the network.

        Args:
            x: Spatial and temporal input tensor combined.

        Returns:
            Output tensor after passing through the network.
        """
        inputs = x
        return self.net(inputs)


class PINN:
    def __init__(self, layers: list, lr: float = 1e-2):
        """
        Initialize the HeatEquationPINN.
        Args:
            layers: List of integers representing the number of neurons in each layer.
            lr: Learning rate for the optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_values = {
            "heat_eq_loss": [],
            "initial_condition_loss": [],
            "boundary_condition_loss": [],
            "total_loss": [],
        }

    def heat_equation_loss(
        self, X_flat: torch.Tensor, f: callable
    ) -> torch.Tensor:
        """
        Compute the loss for the heat equation PDE.
        Args:
            X_flat: Flattened spatial input tensor.
            f: Function representing the source term in the PDE.
        Returns:
            Mean squared error loss for the heat equation.
        """
        u = self.model(X_flat)
        u_x = torch.autograd.grad(
            u, X_flat, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, X_flat, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        return torch.mean((-u_xx - f(X_flat)) ** 2)

    def boundary_condition_loss(
        self,
        boundary_conditions: list,
        interval: tuple,
    ) -> torch.Tensor:
        """
        Compute the loss for the boundary conditions.
        Args:
            boundary_conditions: List of boundary condition values at the left and right boundaries.
            interval: Tuple representing the spatial interval [x0, xL].
        Returns:
            Mean squared error loss for the boundary conditions.
        """
        x1, x2 = interval

        u_bc1 = self.model(x1)
        u_bc2 = self.model(x2)
        return torch.mean((u_bc1 - boundary_conditions[0]) ** 2) + torch.mean(
            (u_bc2 - boundary_conditions[1]) ** 2
        )

    def loss(
        self,
        X_flat: torch.Tensor,
        weight_factors: list,
        boundary_cond: list,
        f: callable,
        interval: tuple,
    ) -> torch.Tensor:
        """
        Compute the total loss for the PINN.
        Args:
            X_flat: Flattened spatial input tensor.
            T_flat: Flattened temporal input tensor.
            x_ic: Spatial input tensor for the initial condition.
            t_bc: Temporal input tensor for the boundary conditions.
            weight_factors: List of weight factors for the losses.
            boundary_cond: List of boundary condition values at the left and right boundaries.
            u0: Function representing the initial condition.
            f: Function representing the source term in the PDE.
            interval: Tuple representing the spatial interval [x0, xL].
        Returns:
            Total loss for the PINN, which is a weighted sum of the PDE loss, initial
            condition loss, and boundary condition loss.
        """
        w1, w2 = weight_factors
        Lpde = self.heat_equation_loss(X_flat, f)
        Lb = self.boundary_condition_loss(boundary_cond, interval)
        return w1 * Lpde + w2 * Lb

    def train(
        self,
        epochs: int,
        print_every: int,
        data: list,
        weight_factors: list,
        boundary_cond: list,
        f: callable,
        interval: tuple,
    ):
        """
        Train the PINN model.

        Args:
            epochs: Number of training epochs.
            print_every: Frequency of printing the loss during training.
            data: List containing the training data [X_flat, T_flat, x_ic, t_bc].
            weight_factors: List of weight factors for the losses.
            boundary_cond: List of boundary condition values at the left and right boundaries.
            u0: Function representing the initial condition.
            f: Function representing the source term in the PDE.
            interval: Tuple representing the spatial interval [x0, xL].
            mode: String representing the mode of the boundary conditions.
        """
        X_flat, x_ic, = data
        X_flat = X_flat.to(self.device)
        x_ic = x_ic.to(self.device)
        X_flat.requires_grad = True
        # No need to activate gradients for x_ic and t_bc

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            loss_val = self.loss(
                X_flat,
                weight_factors,
                boundary_cond,
                f,
                interval,
            )
            loss_val.backward()
            self.optimizer.step()
            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch {epoch}, Loss: {loss_val.item():.6f}")
            self.loss_values["heat_eq_loss"].append(
                self.heat_equation_loss(X_flat, f).item() * weight_factors[0]
            )
            self.loss_values["boundary_condition_loss"].append(
                self.boundary_condition_loss(boundary_cond, interval).item()
                * weight_factors[1]
            )
            self.loss_values["total_loss"].append(loss_val.item())

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict the solution of the heat equation at given spatial and temporal points.
        Args:
            x: Spatial input tensor.
        Returns:
            Numpy array of predicted values at the given points.
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()

    def draw_loss(self):
        """
        Draw the loss values during training.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_values["heat_eq_loss"], label="PDE Loss")
        plt.plot(self.loss_values["initial_condition_loss"], label="IC Loss")
        plt.plot(self.loss_values["boundary_condition_loss"], label="BC Loss")
        plt.yscale("log")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

