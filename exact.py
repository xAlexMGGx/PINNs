import numpy as np

def exact_solution_dirichlet(f, x_eval, L=1.0, g0=0.0, gL=0.0, M=400):
    """
    Devuelve u_exact(x) que satisface -u''=f en (0,L), u(0)=g0, u(L)=gL.
    Usa la representación con función de Green y cuadratura Gauss–Legendre con M puntos.
    x_eval: array de puntos donde evaluar (p.ej., la malla FEM)
    """
    # Nodos/ pesos Gauss–Legendre mapeados a [0,L]
    xi, wi = np.polynomial.legendre.leggauss(M)
    s = 0.5*(xi + 1.0)*L
    w = 0.5*L*wi

    fs = f(s)  # f en nodos de integración

    u_exact = np.empty_like(x_eval, dtype=float)
    for k, x in enumerate(x_eval):
        # Green homogéneo-Dirichlet (para -u'')
        # G(x,s) = s*(L-x)/L si s <= x; = x*(L-s)/L si s > x
        G = np.where(s <= x, s*(L - x)/L, x*(L - s)/L)
        integral = np.sum(G * fs * w)

        # Parte afín que impone las CC no homogéneas
        linear_bc = g0*(1.0 - x/L) + gL*(x/L)

        u_exact[k] = linear_bc + integral

    return u_exact