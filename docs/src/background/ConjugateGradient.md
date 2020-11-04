# Conjugate Gradient Method
The conjugate gradient method can be used to solve symmetric positive
definite (SPD) linear systems.

# Algorithm
To solve ``\mat{A} \vec{Q} = \vec{Q_{rhs}}`` with initial guess ``Q_0``
and preconditioner ``\mat{M}``:
```math
\begin{aligned*}
    \vec{r_0} &= \vec{Q_{rhs}} - \mat{A} \vec{Q_0} \\
    \vec{z_0} &= \mat{M^{-1}} \vec{r_0} \\
    \vec{p_0} &= \vec{z_0} \\
\end{aligned*}
\text{for } i = 1, 2, ...
\begin{aligned*}
    \alpha_i &= \frac{\vec{r_{i-1}^T} \vec{z_{i-1}}}{\vec{p_{i-1}^T} \mat{A} \vec{p_{i-1}}} \text{step size} \\
    \vec{Q_i} &= \vec{Q_{i-1}} + \alpha_i \vec{p_{i-1}} \\
    \vec{r_i} &= \vec{r_{i-1}} - \alpha_i \mat{A} \vec{p_{i-1}} \\
    \vec{z_i} &= \mat{M^{-1}} \vec{r_i} \\
    \beta_i &= \frac{\vec{r_i^T} \vec{z_i}}{\vec{r_{i-1}^T} \vec{z_{i-1}}} \\
    \vec{p_i} &= \vec{z_i} + \beta_i \vec{p_{i-1}}
\end{aligned*}
```

# Tutorial - 2D Laplacian
``
    using Solvent
    using LinearAlgebra
    using SparseArrays
    using Random

    # Creates a Laplacian matrix based on the code from: http://math.mit.edu/~stevenj/18.303/lecture-10.html
    # construct the (M+1)xM matrix D, not including the 1/dx factor

    sdiff1(M) = [ [1.0 spzeros(1, M-1)]; spdiagm(1=>ones(M-1)) - I ]

    function Laplacian(Nx, Ny, Lx, Ly)
        dx = Lx / (Nx+1)
        dy = Ly / (Ny+1)
        Dx = sdiff1(Nx) / dx
        Dy = sdiff1(Ny) / dy
        Ax = Dx' * Dx
        Ay = Dy' * Dy
        return kron(spdiagm(0=>ones(Ny)), Ax) + kron(Ay, spdiagm(0=>ones(Nx)))
    end

    T = Float64
    Lx, Ly = 1, 1
    Nx, Ny = 10, 10
    A = Laplacian(Nx, Ny, Lx, Ly)
    n, _ = size(A)
    b = rand(T, n)
    x = rand(T, n)

    # define the linear operator function
    mulbyA!(y, x) = (y .= A * x)

    # specify choice of solver and preconditoner
    solver_type = ConjugateGradientMethod(M = 100)
    pc_type = Jacobi()
    linearsolver = LinearSolver(
        mulbyA!,
        solver_type,
        x;
        pc_alg = pc_type
    )

    # solve Ax = b
    linearsolve!(linearsolver, x, b)
```
# References
@book{barrett1994cg,
    title = {Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods},
    author = {
            Barrett, Richard and Berry, Michael and Chan, Tony F. and Demmel, James
            and Donato, June and Dongarra, Jack and Eijkhout, Victor and Pozo, Roldan
            and Romine, Charles and van der Vorst, Henk
        },
    pages={12-15},
    year = {1994},
    publisher = {SIAM},
    doi = {10.1137/1.9781611971538}
    }

