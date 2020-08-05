module Solvent

using Adapt
using CUDA
using LinearAlgebra
using LazyArrays
using StaticArrays
using KernelAbstractions

const weighted_norm = false

array_device(::Union{Array, SArray, MArray}) = CPU()
array_device(::CuArray) = CUDADevice()
realview(x::Union{Array, SArray, MArray}) = x
realview(x::CuArray) = x

# Just for testing Solvent
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

include(joinpath("Preconditioners", "Preconditioners.jl"))
include(joinpath("LinearSolvers", "LinearSolvers.jl"))

end  # End of module
