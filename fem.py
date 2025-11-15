import numpy as np

# ------------------------------------------------------------
# FEM 1D para -u'' = f en (0,L) con Dirichlet en 0 y L
# ------------------------------------------------------------
def solve_poisson_1d(f, L=1.0, N=50, g0=0.0, gL=0.0):
    """
    -u''(x) = f(x) en (0, L),  u(0)=g0, u(L)=gL
    FEM con elementos lineales en malla uniforme de N elementos.
    f: función callable que acepte np.array y devuelva np.array
    """
    # Mallado
    x = np.linspace(0.0, L, N+1)
    h = np.diff(x)

    # Matriz global y vector
    A = np.zeros((N+1, N+1), dtype=float)
    b = np.zeros(N+1, dtype=float)

    # Cuadratura de Gauss 2 puntos por elemento
    xi_g = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w_g  = np.array([1.0, 1.0])

    # Ensamblaje
    for e in range(N):
        h_e = h[e]
        i, j = e, e+1

        # Rigidez local
        Ke = (1.0/h_e) * np.array([[1.0, -1.0],
                                   [-1.0,  1.0]])

        # Cargas locales
        Fe = np.zeros(2)
        for xi, w in zip(xi_g, w_g):
            xq = x[i] + 0.5*(xi+1.0)*h_e
            N1 = 0.5*(1.0 - xi)
            N2 = 0.5*(1.0 + xi)
            Fe += w * f(np.array([xq])) * np.array([N1, N2]) * (h_e/2.0)

        # Ensamblaje global
        A[i, i] += Ke[0,0]; A[i, j] += Ke[0,1]
        A[j, i] += Ke[1,0]; A[j, j] += Ke[1,1]
        b[i]    += Fe[0]
        b[j]    += Fe[1]

    # Dirichlet fuerte
    A[0, :] = 0.0; A[0, 0] = 1.0; b[0] = g0
    A[-1,:] = 0.0; A[-1,-1]= 1.0; b[-1]= gL

    # Resolver
    u = np.linalg.solve(A, b)
    return x, u

