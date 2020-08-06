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

        x_ref = A \ b
        
        x_is = copy(x)
        x_kk = copy(x)
        algorithm = GMRES(krylovdim = M, maxiter = K, tol = tol)
        # run once to trigger precompilation
        _, time, alloc, gct, _             = @timed linearsolve!(linearsolver, x, b)
        _, time_is, alloc_is, gct_is, _    = @timed gmres!(x_is, A, b, tol = tol, restart = M, maxiter = K)
        x_kk, time_kk, alloc_kk, gct_kk, _ = @timed linsolve(A, b, x_kk, algorithm)[1]
        x_k, time_k, alloc_k, gct_k, _     = @timed dqgmres(A, b; atol = tol, rtol = tol, memory = M, itmax = K)[1]
        # run 2nd time
        _, time, alloc, gct, _             = @timed linearsolve!(linearsolver, x, b)
        _, time_is, alloc_is, gct_is, _    = @timed gmres!(x_is, A, b, tol = tol, restart = M, maxiter = K)
        x_kk, time_kk, alloc_kk, gct_kk, _ = @timed linsolve(A, b, x_kk, algorithm)[1]
        x_k, time_k, alloc_k, gct_k, _     = @timed dqgmres(A, b; atol = tol, rtol = tol, memory = M, itmax = K)[1]
        @printf "\nTesting type %s:\n" string(T)
        @printf "Method           | Diff from ref | Elapsed Time (s) | GC Time (s) | Allocations\n"
        @printf "-----------------------------------------------------------------------------------\n"
        @printf "Solvent          | %13g | %16g | %11g | %d\n" norm(x_ref - x)    time    gct    alloc
        @printf "IterativeSolvers | %13g | %16g | %11g | %d\n" norm(x_ref - x_is) time_is gct_is alloc_is
        @printf "KrylovKit        | %13g | %16g | %11g | %d\n" norm(x_ref - x_kk) time_kk gct_kk alloc_kk
        @printf "Krylov           | %13g | %16g | %11g | %d\n" norm(x_ref - x_k)  time_k  gct_k  alloc_k
    end
end
