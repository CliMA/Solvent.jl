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


# Tests effectiveness of Jacobi preconditioner using specially designed matrix from:
# http://www.math.iit.edu/~fass/477577_Chapter_16.pdf
@testset "Solvent Preconditioned Jacobi: CG" begin
    T = Float64
    n = 1000
    dv0 = .5 .+ .√(1:n)
    dv1 = ones(n - 1)
    dv2 = ones(n - 100)
    A = spdiagm(0=>dv0) + spdiagm(1=>dv1) + spdiagm(-1=>dv1) + 
        spdiagm(100=>dv2) + spdiagm(-100=>dv2)
    @info "Size of matrix A: ($n, $n)"
    b = ones(T, 1000)
    x = rand(T, n)

    mulbyA!(y, x) = (y .= A * x)

    rtol = sqrt(eps(T))
    atol = eps(T)
    solver_type = ConjugateGradientMethod(M = 100)
    identitysolver = LinearSolver(
        mulbyA!,
        solver_type,
        x;
        rtol = rtol,
        atol = atol,
    )
    jacobisolver = LinearSolver(
        mulbyA!,
        solver_type,
        x;
        pc_alg = Jacobi(),
        rtol = rtol,
        atol = atol,
    )

    x0 = copy(x)
    jacobiiters = linearsolve!(jacobisolver, x, b)
    @test norm(A * x - b) < rtol * norm(A * x0 - b)
    identityiters = linearsolve!(identitysolver, x0, b)
    @test jacobiiters < identityiters
end