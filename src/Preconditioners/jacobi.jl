
export Jacobi

struct Jacobi{S <: PCside} <: AbstractPreconditionerType
    pc_side::S
    function Jacobi(; pc_side = PCleft())
        return new{typeof(pc_side)}(pc_side)
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

# Create P inverse operator
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
    d = diag(A)
    # store inverse of diagonal preconditioner
    pc.cache.P_inv .= 1 ./ d
    return nothing
end

# Apply P inverse to Q
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
