using Test
using Solvent
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
using Printf

# this test setup is partly baed on IterativeSolvers.jl, see e.g
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/test/cg.jl
@testset "Solvent small full system: GCR" begin
    n = 10

    for T in [Float32, Float64]
        Random.seed!(44)

        A = @MMatrix rand(T, n, n)
        A = A' * A + I
        b = @MVector rand(T, n)

        mulbyA!(y, x) = (y .= A * x)

        x = @MVector rand(T, n)

        tol = sqrt(eps(T))
        solver_type = GeneralizedConjugateResidualMethod(M = n, K = 1)
        linearsolver = LinearSolver(
            mulbyA!,
            solver_type,
            x;
            rtol = tol,
            atol = tol,
        )

        x0 = copy(x)
        linearsolve!(linearsolver, x, b)
        @test norm(A * x - b) / norm(A * x0 - b) <= tol

        # test for convergence in 0 iterations by
        # initializing with the exact solution
        x = A \ b
        iters = linearsolve!(linearsolver, x, b)
        @test iters == 0
        @test norm(A * x - b) <= 100 * eps(T)

        newtol = 1000tol
        settolerance!(linearsolver, newtol)
        settolerance!(linearsolver, newtol; relative = true)

        x = @MVector rand(T, n)
        x0 = copy(x)
        linearsolve!(linearsolver, x, b)

        @test norm(A * x - b) / norm(A * x0 - b) <= newtol
        @test norm(A * x - b) / norm(A * x0 - b) >= tol

    end
end

@testset "Solvent large sparse system: GCR" begin
    n = 10000
    T = Float64
    Random.seed!(44)

    α = 1e-2
    A = I + α * sprandn(T, n, n, 0.05)
    b = rand(T, n)

    mulbyA!(y, x) = (y .= A * x)

    x = rand(T, n)

    rtol = sqrt(eps(T))
    atol = eps(T)
    solver_type = GeneralizedConjugateResidualMethod(M = 30, K = 10)
    preconditioner = Identity(pc_side=PCright())
    linearsolver = LinearSolver(
        mulbyA!,
        solver_type,
        x;
        pc_alg = preconditioner,
        rtol = rtol,
        atol = atol,
    )

    x0 = copy(x)
    linearsolve!(linearsolver, x, b)
    @test norm(A * x - b) < 10*rtol * norm(A * x0 - b)

    # test for convergence in 0 iterations by
    # initializing with the exact solution
    x = A \ b
    iters = linearsolve!(linearsolver, x, b)
    @test iters == 0
    @test norm(A * x - b) < 1000*atol

    newrtol = 1000rtol
    newatol = 1000atol
    settolerance!(linearsolver, newatol)
    settolerance!(linearsolver, newrtol; relative=true)

    x = rand(T, n)
    x0 = copy(x)
    linearsolve!(linearsolver, x, b)

    @test norm(A * x - b) < newrtol * norm(A * x0 - b)

end
