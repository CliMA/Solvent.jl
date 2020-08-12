using CUDA
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Random
using Printf
using TimerOutputs

import ConjGrad
import IterativeSolvers 
import Krylov
import KrylovKit
import Solvent

function invertiblesparse_test(::Type{T}, nx::Int, ny::Int=nx, α::Number=0.01,
        density::AbstractFloat=0.05) where T
    A = I + T(α) * sprandn(T, nx, ny, density)
    b = rand(T, nx)
    xinit = rand(T, ny)
    return (A, b, xinit)
end

function laplace_test(::Type{T}, nx::Int, ny::Int=nx, lx::Int=1,
        ly::Int=lx) where T
    dx = T(lx) / T(nx + 1)
    dy = T(ly) / T(ny + 1)
    Dx = [[T(1) spzeros(T, 1, nx - 1)]; spdiagm(1=>ones(T, nx - 1)) - I] / dx
    Dy = [[T(1) spzeros(T, 1, ny - 1)]; spdiagm(1=>ones(T, ny - 1)) - I] / dy
    Ax = Dx' * Dx
    Ay = Dy' * Dy
    A = kron(sparse(I, ny, ny), Ax) + kron(Ay, sparse(I, nx, nx))
    b = rand(T, nx)
    xinit = rand(T, ny)
    return (A, b, xinit)
end

function benchmark_gmres!(timer::TimerOutput, Aname::String, dims::Int,
        iters::Int, tol::Number, A::AbstractArray, b::Vector, xinit::Vector)
    names = [string(algname, "-GMRES-", Aname) for algname in
        ["IterativeSolvers", "Krylov", "KrylovKit", "Solvent"]]
    xfinals = [copy(xinit) for _ in 1:4]
    # Disable timer for precompilation, then run again with timer enabled.
    for timerfunc in [disable_timer!, enable_timer!]
        timerfunc(timer)
        @timeit timer names[1] begin
            IterativeSolvers.gmres!(xfinals[1], A, b, tol = tol, restart = dims,
                maxiter = iters)
        end
        @timeit timer names[2] begin
            xfinals[2] = Krylov.dqgmres(A, b; atol = tol, rtol = tol,
                memory = dims, itmax = iters)[1]
        end
        @timeit timer names[3] begin
            algorithm = KrylovKit.GMRES(krylovdim = dims, maxiter = iters,
                tol = tol)
            xfinals[3] = KrylovKit.linsolve(A, b, xinit, algorithm)[1]
        end
        @timeit timer names[4] begin
            linearsolver = Solvent.LinearSolver(
                (y, x) -> (y .= A * x),
                Solvent.GeneralizedMinimalResidualMethod(M = dims, K = iters),
                xinit;
                pc_alg = Solvent.Identity(pc_side=Solvent.PCright()),
                rtol = tol,
                atol = tol,
            )
            Solvent.linearsolve!(linearsolver, xfinals[4], b)
        end
    end
    # TODO: Print residuals in same table as times and allocations.
    # xexact = A \ b
    # for i in 1:4
    #     println(names[i], " Residual: ", norm(xfinals[i] - xexact))
    # end
end

const timer = TimerOutput()
for testfunc in [invertiblesparse_test]#, laplace_test]
    for T in [Float32, Float64]
        for _ in 1:3
            benchmark_gmres!(timer, string(testfunc, "-", T, "-10000-20-10"),
                20, 10, sqrt(eps(T)), testfunc(T, 10000)...)
        end
    end
end
print_timer(timer, sortby = :name)