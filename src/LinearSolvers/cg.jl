
export ConjugateGradientMethod

struct ConjugateGradientMethod <: AbstractKrylovMethod
    "Maximum number of CG iterations"
    M::Int
    function ConjugateGradientMethod(; M=30)
        return new(M)
    end
end

mutable struct CGCache{AT} <: AbstractLinearSolverCache
    r::AT
    z::AT
    p::AT
    Ap::AT
end

function cache(
    krylov_alg::ConjugateGradientMethod,
    Q::AT,
) where {AT}
    r = similar(Q)
    z = similar(Q)
    p = similar(Q)
    Ap = similar(Q)

    return CGCache{AT}(
        r,
        z,
        p,
        Ap,
    )
end

function LSinitialize!(
    ::ConjugateGradientMethod,
    solver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    linearoperator! = solver.linop!
    pc = solver.pc
    PCinitialize!(pc, Q, Qrhs, args...)

    cache = solver.cache
    r = cache.r
    z = cache.z
    p = cache.p

    linearoperator!(r, Q, args...)
    r .= Qrhs .- r

    residual_norm = norm(r, weighted_norm)
    threshold = solver.rtol * residual_norm

    converged = false
    if threshold < solver.atol
        converged = true
        return converged, threshold
    end

    isa(pc.pc_side, PCleft) || error("Only supports left preconditioning")
    PCapply!(pc, z, r, args...)
    p .= z

    converged, max(threshold, solver.atol)
end

function LSsolve!(
    krylov_alg::ConjugateGradientMethod,
    solver::LinearSolver,
    threshold,
    Q,
    Qrhs,
    args...,
)
    converged = false
    iter = 0
    residual_norm = typemax(eltype(Q))

    linearoperator! = solver.linop!
    pc = solver.pc
    cache = solver.cache
    r = cache.r
    z = cache.z
    p = cache.p
    Ap = cache.Ap

    ω0 = dot(r, z)
    while !converged && iter < krylov_alg.M
        linearoperator!(Ap, p, args...)
        α = ω0 / dot(p, Ap)
        Q .+= α .* p
        r .= r .- α .* Ap
        residual_norm = norm(r, weighted_norm)
        if residual_norm < threshold
            converged = true
            break
        end
        # apply preconditioning
        PCapply!(pc, z, r, args...)
        ω1 = dot(r, z)
        β =  ω1 / ω0
        p .= z .+ β .* p
        ω0 = ω1
        iter += 1
    end

    (converged, iter, residual_norm)
end
