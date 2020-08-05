using Test
using SystemSolvers
using StaticArrays
using LinearAlgebra
using Random
using Printf

# this test setup is partly based on IterativeSolvers.jl, see e.g
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/test/cg.jl
@testset "SystemSolvers small full system" begin
    n = 10

    for T in [Float32, Float64]
        Random.seed!(44)

        A = @MMatrix rand(T, n, n)
        A = A' * A + I
        b = @MVector rand(T, n)

        mulbyA!(y, x) = (y .= A * x)

        x = @MVector rand(T, n)

        tol = sqrt(eps(T))
        solver_type = GeneralizedMinimalResidualMethod(M = 30, K = 5)
        preconditioner = Identity(pc_side=PCleft())
        linearsolver = LinearSolver(
            mulbyA!,
            solver_type,
            preconditioner,
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

@testset "SystemSolvers larger full system" begin
    n = 1000

    for T in [Float32, Float64]
        for (i, α) in enumerate(T[1e-2, 5e-3])
            Random.seed!(44)

            A = rand(T, 200, 1000)
            A = α * A' * A + I
            b = rand(T, n)

            mulbyA!(y, x) = (y .= A * x)

            x = rand(T, n)

            tol = sqrt(eps(T))
            solver_type = GeneralizedMinimalResidualMethod(M = 30, K = 10)
            preconditioner = Identity(pc_side=PCright())
            linearsolver = LinearSolver(
                mulbyA!,
                solver_type,
                preconditioner,
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
            @test norm(A * x - b) <= tol

            newtol = 1000tol
            settolerance!(linearsolver, newtol)
            settolerance!(linearsolver, newtol; relative=true)

            x = rand(T, n)
            x0 = copy(x)
            linearsolve!(linearsolver, x, b)
    
            @test norm(A * x - b) / norm(A * x0 - b) <= newtol

        end
    end
end
