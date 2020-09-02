
export AbstractPreconditioner
export AbstractPreconditionerCache
export AbstractPreconditionerType
export PCinitialize!, PCapply!
export PCside, PCright, PCleft

"""
    AbstractPreconditioner

This is an abstract type representing a an abstract
preconditioner.
"""
abstract type AbstractPreconditioner end

abstract type AbstractPreconditionerCache end
abstract type AbstractPreconditionerType end

abstract type PCside end
struct PCright <: PCside end
struct PCleft <: PCside end

mutable struct Preconditioner{
    pcType <: AbstractPreconditionerType,
    OP,
    pccType <: AbstractPreconditionerCache,
    stype <: PCside,
} <: AbstractPreconditioner
    pc_type::pcType
    linop!::OP
    cache::pccType
    pc_side::stype

    function Preconditioner(
        pc_type::AbstractPreconditionerType,
        linearoperator!,
        Q,
    )
        pccache = cache(pc_type, linearoperator!, Q)
        OP = typeof(linearoperator!)
        pcType = typeof(pc_type)
        pccType = typeof(pccache)
        stype = typeof(pc_type.pc_side)

        return new{pcType, OP, pccType, stype}(
            pc_type,
            linearoperator!,
            pccache,
            pc_type.pc_side,
        )
    end
end

function PCinitialize!(
    pc::AbstractPreconditioner,
    Q,
    Qrhs,
    args...,
)
    PCinitialize!(pc.pc_type, pc, Q, Qrhs, args...)
end

function PCapply!(
    pc::AbstractPreconditioner,
    W,
    Q,
    args...,
)
    PCapply!(pc.pc_type, pc, W, Q, args...)
end

include("identity.jl")
include("jacobi.jl")
