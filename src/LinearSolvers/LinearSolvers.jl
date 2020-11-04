
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

The available concrete implementations are:

- [`Solvent.GeneralizedMinimalResidualMethod`](@ref)
- [`Solvent.ConjugateGradientMethod`](@ref)
- [`Solvent.GeneralizedConjugateResidualMethod`](@ref)
"""
abstract type AbstractLinearSolver <: AbstractSystemSolver end

abstract type AbstractLinearSolverCache end
abstract type AbstractKrylovMethod end

"""
    LinearSolver(
        linearoperator!,
        krylov_alg::AbstractKrylovMethod,
        Q::AT;
        pc_alg = Identity(),
        rtol = √eps(eltype(AT)),
        atol = eps(eltype(AT)),
    ) where {AT}

Object for a linear system solver with operator `linearoperator!`
i.e,
```math
L(Q) = Q_{rhs}
```
with solver method specified by `krylov_alg`. Preconditioning may
be specified by `pc_alg`.
"""
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

"""
    linearsolve!(
        linearsolver::LinearSolver,
        Q,
        Qrhs,
        args...;
        cvg = Ref{Bool}(),
    )

Solves a linear problem defined by the `linearsolver` object and the state
`Qrhs`, i.e,
```math
L(Q) = Q_{rhs}
```
using the initial guess `Q`. After the call `Q` contains the solution.
"""
function linearsolve!(
    linearsolver::LinearSolver,
    Q,
    Qrhs,
    args...;
    cvg = Ref{Bool}(),
)
    converged = false
    iters = 0
    converged, threshold = LSinitialize!(
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

"""
    settolerance!(linearsolver::LinearSolver, tol; relative=false)

Set the relative or absolute tolerance of `linearsolver` to `tol`.

Set `rtol` if `relative = true` or `atol` if `relative = false`.
"""
function settolerance!(linearsolver::LinearSolver, tol; relative = false)
    if relative
        linearsolver.rtol = tol
    else
        linearsolver.atol = tol
    end
    nothing
end

"""
    LSinitialize!(
        linearsolver::LinearSolver,
        Q,
        Qrhs,
        args...,
    )

Initialize the preconditioner and linear solver.
"""
function LSinitialize!(
    linearsolver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    # Initializes the preconditioner
    PCinitialize!(linearsolver.pc, Q, Qrhs, args...)
    LSinitialize!(
        linearsolver.krylov_alg,
        linearsolver,
        Q,
        Qrhs,
        args...,
    )
end

"""
    LSsolve!(
        linearsolver::LinearSolver,
        threshold,
        Q,
        Qrhs,
        args...,
    )

Implement linear solver iterations as specified by the solver type.
"""
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
include("cg.jl")
