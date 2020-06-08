module SystemSolvers

using Adapt
using CuArrays
using LinearAlgebra
using LazyArrays
using StaticArrays
using KernelAbstractions

# TODO: This is big ugly; will work on a better norm interface
const weighted_norm = false

# TODO: Should these helper functions belong to SystemSolvers?
# (Probably not; should belong to MPIStateArrays.jl or some Array package)
array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(::CuArray) = CUDA()
realview(x::Union{Array, SArray, MArray}) = x
realview(x::CuArray) = x

# Just for testing SystemSolvers
LinearAlgebra.norm(A::MVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::MVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::MVector, B::MVector, weighted) = dot(A, B)
LinearAlgebra.norm(A::AbstractVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::AbstractVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::AbstractVector, B::AbstractVector, weighted) = dot(A, B)

export linearsolve!, settolerance!, prefactorize
export AbstractSystemSolver, AbstractIterativeSystemSolver

"""
    AbstractSystemSolver

This is an abstract type representing a generic linear solver.
"""
abstract type AbstractSystemSolver end

"""
    AbstractIterativeSystemSolver

This is an abstract type representing a generic iterative
linear solver.
"""
abstract type AbstractIterativeSystemSolver <: AbstractSystemSolver end

"""
    settolerance!(solver::AbstractIterativeSystemSolver, tolerance, relative)

Sets the relative or absolute tolerance of the iterative linear solver
`solver` to `tolerance`.
"""
settolerance!(
    solver::AbstractIterativeSystemSolver,
    tolerance,
    relative = true,
) = (relative ? (solver.rtol = tolerance) : (solver.atol = tolerance))

doiteration!(
    linearoperator!,
    x,
    b,
    solver::AbstractIterativeSystemSolver,
    tolerance,
    args...,
) = throw(MethodError(
    doiteration!,
    (linearoperator!, x, b, solver, tolerance, args...),
))

initialize!(
    linearoperator!,
    x,
    b,
    solver::AbstractIterativeSystemSolver,
    args...,
) = throw(MethodError(initialize!, (linearoperator!, x, b, solver, args...)))

"""
    prefactorize(linop!, linearsolver, args...)

Prefactorize the in-place linear operator `linop!` for use with `linearsolver`.
"""
prefactorize(linop!, linearsolver::AbstractIterativeSystemSolver, args...) =
    linop!

"""
    linearsolve!(linearoperator!, solver::AbstractIterativeSystemSolver, x, b, args...)

Solves a linear problem defined by the `linearoperator!` function and the right-hand
side `b`, i.e,

```math
A x  = b
```

using the `solver` and the initial guess `x`. After the call `x` contains the
solution.  The arguments `args` is passed to `linearoperator!` when it is
called.
"""
function linearsolve!(
    linearoperator!,
    solver::AbstractIterativeSystemSolver,
    x,
    b,
    args...;
    max_iters = length(Q),
    cvg = Ref{Bool}(),
)
    converged = false
    iters = 0

    converged, threshold =
        initialize!(linearoperator!, x, b, solver, args...)
    converged && return iters

    while !converged && iters < max_iters
        converged, inner_iters, residual_norm =
            doiteration!(linearoperator!, x, b, solver, threshold, args...)

        iters += inner_iters

        if !isfinite(residual_norm)
            error("norm of residual is not finite after $iters iterations")
        end

        achieved_tolerance = residual_norm / threshold * solver.rtol
    end

    converged || @warn "Solver did not converge after $iters iterations"
    cvg[] = converged

    iters
end

@kernel function linearcombination!(x, cs, Xs, increment::Bool)
    i = @index(Global, Linear)
    if !increment
        @inbounds x[i] = -zero(eltype(x))
    end
    @inbounds for j in 1:length(cs)
        x[i] += cs[j] * Xs[j][i]
    end
end

# TODO: Include concrete implementations and interfaces
include("krylov_methods/gcr.jl")
include("krylov_methods/gmres.jl")

end  # End of module
