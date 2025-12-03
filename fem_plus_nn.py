import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# IMPORTANTE: Esto asume que tienes 'fem.py' en el mismo directorio
from fem import solve_poisson_1d

# ==========================================
# 1. DEFINICIÓN DEL PROBLEMA FÍSICO
# ==========================================

# Configuración del dominio y el pico
L_domain = 1.0
A_peak = 2.5       # Altura del pico
x0_peak = 0.8      # Posición del pico
alpha = 5e-3       # Ancho del pico (muy estrecho para FEM)

def u_exact_func(x):
    """Solución exacta: Parte suave (parábola) + Pico Gaussiano"""
    suave = x * (L_domain - x)
    pico = A_peak * np.exp(-(x - x0_peak)**2 / alpha)
    return suave + pico

def f_source_func(x):
    """
    Fuente f(x) tal que -u'' = f.
    Calculamos f como la suma de -u'' de cada componente.
    """
    # 1. Parte suave: u = x(1-x) -> u'' = -2 -> f_suave = 2
    f_suave = 2.0 * np.ones_like(x)
    
    # 2. Parte pico: u = A * exp(-k(x-x0)^2)
    k = 1.0 / alpha
    exp_term = np.exp(-k * (x - x0_peak)**2)
    # Segunda derivada del pico
    u_xx_pico = A_peak * exp_term * (4 * k**2 * (x - x0_peak)**2 - 2 * k)
    # f_pico = -u''
    f_pico = -u_xx_pico
    
    return f_suave + f_pico

# ==========================================
# 2. RESOLUCIÓN FEM (Usando tu fem.py)
# ==========================================

# Usamos N=10 para simular una malla gruesa donde FEM falla
N_elements = 11
x_fem, u_fem_nodes = solve_poisson_1d(f_source_func, L=L_domain, N=N_elements)

# Interpolador para evaluar u_FEM en cualquier punto x (necesario para sumar a la NN)
fem_spline = CubicSpline(x_fem, u_fem_nodes)
def get_u_fem(x_query):
    return fem_spline(x_query)


# ==========================================
# 3. RED NEURONAL HÍBRIDA (PyTorch)
# ==========================================

# Configuración
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Definición de la Red (MLP simple)
class ResidualNN(nn.Module):
    def __init__(self, hidden_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Función de ventana para localizar la corrección (solo cerca del pico)
def window_func(x_tensor):
    # Centrada en x0_peak=0.8, ancho controlado
    return torch.exp(-(x_tensor - x0_peak)**2 / 0.15**2)

model = ResidualNN(hidden_dim=300).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Datos de entrenamiento para la NN
# Usamos puntos densos para aprender la física (f) correctamente
len_x_train = 500
x_train_np = np.linspace(0, L_domain, len_x_train)
x_train = torch.tensor(x_train_np, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
f_target = torch.tensor(f_source_func(x_train_np), dtype=torch.float32).unsqueeze(1).to(device)
u_fem = get_u_fem(x_train_np)
u_fem_tensor = torch.tensor(u_fem, dtype=torch.float32).unsqueeze(1).to(device)
# get second derivative of u_fem 
fem_spline_2nd_deriv = fem_spline.derivative(nu=2)
u_fem_2nd_deriv = fem_spline_2nd_deriv(x_train_np)
u_fem_2nd_deriv_tensor = torch.tensor(u_fem_2nd_deriv, dtype=torch.float32).unsqueeze(1).to(device)

# Determinamos máscara en la que aplicar el entrenamiento en función de los residuos de FEM
fem_residual = - fem_spline_2nd_deriv(x_train_np) - f_source_func(x_train_np)
mask_window = np.abs(fem_residual) > 1

kernel_size = len_x_train // 15
kernel = np.ones(kernel_size) / kernel_size
mask_window = np.convolve(mask_window.astype(float), kernel, mode='same') > 0.5

x_train = x_train[mask_window].detach().clone()
x_train.requires_grad = True
f_target = f_target[mask_window]
u_fem_2nd_deriv_tensor = u_fem_2nd_deriv_tensor[mask_window]
u_fem_nodes_tensor = torch.tensor(u_fem_nodes, dtype=torch.float32).unsqueeze(1).to(device)


# ==========================================
# 4. BUCLE DE ENTRENAMIENTO
# ==========================================
print(f"Entrenando corrección híbrida para malla FEM de N={N_elements}...")

epochs = 20000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # ... (Dentro del bucle de entrenamiento) ...
    
    # 1. Forward Pass
    nn_output = model(x_train)
    b_val = window_func(x_train)
    correction = b_val * nn_output
    
    # 2. Calcular derivadas de la CORRECCIÓN (NN)
    # Calculamos la segunda derivada solo de la parte neuronal
    grad_u_nn = torch.autograd.grad(correction, x_train, grad_outputs=torch.ones_like(correction), create_graph=True)[0]
    grad_uu_nn = torch.autograd.grad(grad_u_nn, x_train, grad_outputs=torch.ones_like(grad_u_nn), create_graph=True)[0]
    
    # 3. Residuo de la EDP COMPLETA: -(u_FEM'' + u_NN'') = f
    # Aquí sumamos la curvatura que ya aporta el FEM (u_fem_xx_tensor)
    # para que la NN solo tenga que aportar lo que falta.
    
    u_total_xx = u_fem_2nd_deriv_tensor + grad_uu_nn  # Suma de curvaturas
    residual = - u_total_xx - f_target          # El residuo real
    
    # 4. Loss
    loss_res = torch.mean(residual**2)

    loss_cor = torch.mean((model(torch.tensor(x_fem, dtype=torch.float32).unsqueeze(1).to(device)) - u_fem_nodes_tensor)**2)  # Regularización para evitar grandes correcciones fuera de la ventana

    
    loss = loss_res + loss_cor
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}: Res={loss_res.item():.6f}")#, Cor={loss_cor.item():.6f}")

# ==========================================
# 5. VISUALIZACIÓN
# ==========================================

# Malla fina para gráficos
x_plot = np.linspace(0, L_domain, 500)
x_tensor = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1).to(device)

# 1. Obtener predicciones finales
with torch.no_grad():
    nn_pred = model(x_tensor)
    b_plot = window_func(x_tensor)
    u_correction = (b_plot * nn_pred).cpu().numpy().flatten()

# 2. Construir soluciones
u_exact = u_exact_func(x_plot)
u_fem_interp = get_u_fem(x_plot)  # FEM interpolado
u_hybrid = u_correction + u_fem_interp # Híbrido

# 3. Graficar
plt.figure(figsize=(10, 6))
plt.plot(x_plot, u_exact, 'k--', linewidth=2, label='Exacta')
plt.plot(x_plot, u_fem_interp, 'r-', alpha=0.6, linewidth=2, label='FEM (Malla basta)')
plt.plot(x_fem, u_fem_nodes, 'ro', label='Nodos FEM')
plt.plot(x_plot, u_hybrid, 'b-', linewidth=2, label='Híbrido (FEM + NN)')
plt.plot(x_plot, u_correction, 'g-', linewidth=1, label='Corrección NN')

# Zona sombreada donde actúa la NN
plt.axvspan(0.7, 0.9, color='gray', alpha=0.1, label='Zona Corrección')

plt.title('Mejora de FEM (N=10) mediante Red Neuronal (Physics-Informed)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fem_plus_nn_solution.png')
plt.show()


# Gráfico de Error
plt.figure(figsize=(10, 4))
plt.plot(x_plot, np.abs(u_exact - u_fem_interp), 'r', label='Error FEM')
plt.plot(x_plot, np.abs(u_exact - u_hybrid), 'b', label='Error Híbrido')
plt.title('Comparación de Error Absoluto')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.legend()
plt.grid(True)
plt.show()
