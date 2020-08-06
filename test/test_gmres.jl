using Test
using Solvent
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
using Printf
using IterativeSolvers: gmres!
using KrylovKit: GMRES, linsolve
using Krylov: dqgmres

# this test setup is partly based on IterativeSolvers.jl, see e.g
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/test/cg.jl
# @testset "Solvent small full system" begin
#     n = 10

#     for T in [Float32, Float64]
#         Random.seed!(44)

#         A = @MMatrix rand(T, n, n)
#         A = A' * A + I
#         b = @MVector rand(T, n)

#         mulbyA!(y, x) = (y .= A * x)

#         x = @MVector rand(T, n)

#         tol = sqrt(eps(T))
#         solver_type = GeneralizedMinimalResidualMethod(M = n, K = 1)
#         preconditioner = Identity(pc_side=PCleft())
#         linearsolver = LinearSolver(
#             mulbyA!,
#             solver_type,
#             preconditioner,
#             x;
#             rtol = tol,
#             atol = tol,
#         )

#         x0 = copy(x)
#         linearsolve!(linearsolver, x, b)
#         @test norm(A * x - b) / norm(A * x0 - b) <= tol

#         # test for convergence in 0 iterations by
#         # initializing with the exact solution
#         x = A \ b
#         iters = linearsolve!(linearsolver, x, b)
#         @test iters == 0
#         @test norm(A * x - b) <= 100 * eps(T)

#         newtol = 1000tol
#         settolerance!(linearsolver, newtol)
#         settolerance!(linearsolver, newtol; relative = true)

#         x = @MVector rand(T, n)
#         x0 = copy(x)
#         linearsolve!(linearsolver, x, b)

#         @test norm(A * x - b) / norm(A * x0 - b) <= newtol
#         @test norm(A * x - b) / norm(A * x0 - b) >= tol

#     end
# end

@testset "Solvent large sparse system" begin
    n = 10000
    M = 20
    K = 10

    for T in [Float32, Float64]
        Random.seed!(44)

        α = 1f-2
        A = I + α * sprandn(T, n, n, 0.05)
        b = rand(T, n)

        mulbyA!(y, x) = (y .= A * x)

        x = rand(T, n)

        tol = sqrt(eps(T))
        solver_type = GeneralizedMinimalResidualMethod(M = M, K = K)
        preconditioner = Identity(pc_side=PCright())
        linearsolver = LinearSolver(
            mulbyA!,
            solver_type,
            preconditioner,
            x;
            rtol = tol,
            atol = tol,
        )

        x_is = copy(x)
        x_kk = copy(x)
        algorithm = GMRES(krylovdim = M, maxiter = K, tol = tol)
        _, time, alloc, gct, _             = @timed linearsolve!(linearsolver, x, b)
        _, time_is, alloc_is, gct_is, _    = @timed gmres!(x_is, A, b, tol = tol, restart = M, maxiter = K)
        x_kk, time_kk, alloc_kk, gct_kk, _ = @timed linsolve(A, b, x_kk, algorithm)[1]
        x_k, time_k, alloc_k, gct_k, _     = @timed dqgmres(A, b; atol = tol, rtol = tol, memory = M, itmax = K)[1]
        @printf "\nTesting type %s:\n" string(T)
        @printf "Method           | Diff from Solvent | Elapsed Time (s) | GC Time (s) | Allocations\n"
        @printf "-----------------------------------------------------------------------------------\n"
        @printf "Solvent          | %17g | %16g | %11g | %d\n" norm(x - x)    time    gct    alloc
        @printf "IterativeSolvers | %17g | %16g | %11g | %d\n" norm(x - x_is) time_is gct_is alloc_is
        @printf "KrylovKit        | %17g | %16g | %11g | %d\n" norm(x - x_kk) time_kk gct_kk alloc_kk
        @printf "Krylov           | %17g | %16g | %11g | %d\n" norm(x - x_k)  time_k  gct_k  alloc_k

        # x0 = copy(x)
        # linearsolve!(linearsolver, x, b)
        # @test norm(A * x - b) / norm(A * x0 - b) <= tol

        # # test for convergence in 0 iterations by
        # # initializing with the exact solution
        # x = A \ b
        # iters = linearsolve!(linearsolver, x, b)
        # @test iters == 0
        # @test norm(A * x - b) <= tol

        # newtol = 1000tol
        # settolerance!(linearsolver, newtol)
        # settolerance!(linearsolver, newtol; relative=true)

        # x = rand(T, n)
        # x0 = copy(x)
        # linearsolve!(linearsolver, x, b)

        # @test norm(A * x - b) / norm(A * x0 - b) <= newtol
    end
end
