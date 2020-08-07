
export Identity

struct Identity{S <: PCside} <: AbstractPreconditionerType
    pc_side::S
    function Identity(; pc_side = PCleft())
        return new{typeof(pc_side)}(pc_side)
    end
end

struct IdentityCache <: AbstractPreconditionerCache end

function cache(::Identity, linearoperator!, Q)
    return IdentityCache()
end

function PCinitialize!(
    ::Identity,
    ::Preconditioner,
    Q,
    Qrhs,
    args...,
) 
    return nothing
end

function PCapply!(
    ::Identity,
    ::Preconditioner,
    W,
    Q,
    args...,
)
    W .= Q
end
