using Test
using Solvent
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
using Printf

# Creates a Laplacian matrix based on the code from: http://math.mit.edu/~stevenj/18.303/lecture-10.html
# construct the (M+1)xM matrix D, not including the 1/dx factor
sdiff1(M) = [ [1.0 spzeros(1, M-1)]; spdiagm(1=>ones(M-1)) - I ]

# make the discrete -Laplacian in 2d, with Dirichlet boundaries
function Laplacian(Nx, Ny, Lx, Ly)
    dx = Lx / (Nx+1)
    dy = Ly / (Ny+1)
    Dx = sdiff1(Nx) / dx
    Dy = sdiff1(Ny) / dy
    Ax = Dx' * Dx
    Ay = Dy' * Dy
    return kron(spdiagm(0=>ones(Ny)), Ax) + kron(Ay, spdiagm(0=>ones(Nx)))
end

@testset "Solvent Laplacian: CG" begin
    T = Float64
    Lx = 1
    Ly = 1
    Nx = 10
    Ny = 10
    A = Laplacian(Nx, Ny, Lx, Ly)
    n, _ = size(A)
    @info "Size of matrix A: ($n, $n)"
    b = rand(T, n)
    x = rand(T, n)

    mulbyA!(y, x) = (y .= A * x)

    rtol = sqrt(eps(T))
    atol = eps(T)
    solver_type = ConjugateGradientMethod(M = 100)
    linearsolver = LinearSolver(
        mulbyA!,
        solver_type,
        x;
        rtol = rtol,
        atol = atol,
    )

    x0 = copy(x)
    linearsolve!(linearsolver, x, b)
    @test norm(A * x - b) < rtol * norm(A * x0 - b)

end
