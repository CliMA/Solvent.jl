
export AbstractLinearSolver, AbstractLinearSolverCache
export AbstractKrylovMethod
export LinearSolver
export settolerance!
export linearsolve!
export LSinitialize!, LSsolve!

"""
    AbstractLinearSolver

This is an abstract type representing a generic iterative
linear solver.
"""
abstract type AbstractLinearSolver <: AbstractSystemSolver end

abstract type AbstractLinearSolverCache end
abstract type AbstractKrylovMethod end

mutable struct LinearSolver{
    OP,
    krylovType <: AbstractKrylovMethod,
    pcType <: AbstractPreconditioner,
    fType,
    lscType <: AbstractLinearSolverCache,
} <: AbstractLinearSolver
    linop!::OP
    krylov_alg::krylovType
    pc::pcType
    rtol::fType
    atol::fType
    cache::lscType

    function LinearSolver(
        linearoperator!,
        krylov_alg::AbstractKrylovMethod,
        Q::AT;
        pc_alg = Identity(),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
    ) where {AT}
        lscache = cache(
            krylov_alg,
            Q,
        )
        pc = Preconditioner(pc_alg, linearoperator!, Q)

        OP = typeof(linearoperator!)
        krylovType = typeof(krylov_alg)
        pcType = typeof(pc)
        fType = typeof(rtol)
        lscType = typeof(lscache)

        return new{OP, krylovType, pcType, fType, lscType}(
            linearoperator!,
            krylov_alg,
            pc,
            rtol,
            atol,
            lscache,
        )
    end
end

function linearsolve!(
    linearsolver::LinearSolver,
    Q,
    Qrhs,
    args...;
    cvg = Ref{Bool}(),
)
    converged = false
    iters = 0
    krylov_alg = linearsolver.krylov_alg
    linearoperator! = linearsolver.linop!
    converged, threshold = LSinitialize!(
        krylov_alg,
        linearsolver,
        Q,
        Qrhs,
        args...,
    )
    converged && return iters
    
    converged, iters, residual_norm = LSsolve!(
        linearsolver,
        threshold,
        Q,
        Qrhs,
        args...,
    )
    converged || @warn "Linear solver did not attain convergence after $iters iterations"
    cvg[] = converged
    iters
end

function settolerance!(linearsolver::LinearSolver, tol; relative = false)
    if relative
        linearsolver.rtol = tol
    else
        linearsolver.atol = tol
    end
    nothing
end

function LSinitialize!(
    linearsolver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    PCinitialize!(linearsolver.pc, Q, Qrhs, args...)
    LSinitialize!(
        linearsolver.krylov_alg,
        linearsolver,
        Q,
        Qrhs,
        args...,
    )
end

function LSsolve!(
    linearsolver::LinearSolver,
    threshold,
    Q,
    Qrhs,
    args...,
)
    LSsolve!(
        linearsolver.krylov_alg,
        linearsolver,
        threshold,
        Q,
        Qrhs,
        args...,
    )
end

@kernel function linearcombination!(Q, cs, Xs, increment::Bool)
    i = @index(Global, Linear)
    if !increment
        @inbounds Q[i] = -zero(eltype(Q))
    end
    @inbounds for j in 1:length(cs)
        Q[i] += cs[j] * Xs[j][i]
    end
end

include("gmres.jl")
include("gcr.jl")
