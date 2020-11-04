
export Jacobi

"""
    Jacobi()

Preconditioner object for the Jacobi (diagonal) left-side preconditioner.

Effective for diagonally-dominant systems.
"""
struct Jacobi{PCleft} <: AbstractPreconditionerType
    pc_side::PCleft
    function Jacobi()
        return new{PCleft}()
    end
end

mutable struct JacobiCache{AT} <: AbstractPreconditionerCache 
    P_inv::AT
end

function cache(::Jacobi, linearoperator!, Q)
    x = similar(Q)
    AT = typeof(Q)
    return JacobiCache{AT}(x)
end

# Create P inverse preconditioner operator
function PCinitialize!(
    ::Jacobi,
    pc::Preconditioner,
    Q,
    Qrhs,
    args...,
)
    # extract matrix form of operator A
    n = length(Q)
    A = similar(Q, n, n)
    pc.linop!(A, I)
    # store inverse of diagonal preconditioner
    pc.cache.P_inv .= 1 ./ diag(A)
    return nothing
end

# Apply P inverse
function PCapply!(
    ::Jacobi,
    pc::Preconditioner,
    W,
    Q,
    args...,
)
    P_inv = pc.cache.P_inv
    W .= P_inv .* Q
end
