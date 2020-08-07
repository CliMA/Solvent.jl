
using Printf

export ConjugateGradientMethod


struct ConjugateGradientMethod <: AbstractKrylovMethod
    "Maximum number of CG iterations"
    M::Int
    function ConjugateGradientMethod(; M=30)
        return new(M)
    end
end

mutable struct CGCache{AT} <: AbstractLinearSolverCache
    r0::AT
    r1::AT
    z0::AT
    z1::AT
    p0::AT
end

function cache(
    krylov_alg::ConjugateGradientMethod,
    Q::AT,
) where {AT}
    r0 = similar(Q)
    r1 = similar(Q)
    z0 = similar(Q)
    z1 = similar(Q)
    p0 = similar(Q)

    return CGCache{AT}(
        r0,
        r1,
        z0,
        z1,
        p0,
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
    r0 = cache.r0
    z0 = cache.z0
    p0 = cache.p0

    linearoperator!(r0, Q, args...)
    @. r0 = Qrhs - r0

    residual_norm = norm(r0, weighted_norm)
    @info "Initial residual: $residual_norm"
    threshold = solver.rtol * residual_norm

    converged = false
    if threshold < solver.atol
        converged = true
        return converged, threshold
    end

    isa(pc.pc_side, PCleft) || error("Only supports left preconditioning")
    PCapply!(pc, z0, r0, args...)
    @. p0 = z0

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
    r0 = cache.r0
    r1 = cache.r1
    z0 = cache.z0
    z1 = cache.z1
    p0 = cache.p0

    Ap0 = similar(Q)

    while !converged && iter < krylov_alg.M
        linearoperator!(Ap0, p0, args...)
        α = r0' * z0 / (p0' * Ap0)
        @. Q += α * p0
        @. r1 = r0 - α * Ap0
        residual_norm = norm(r1, weighted_norm)
        @info "Residual at iteration $iter: $residual_norm"
        if residual_norm < threshold
            converged = true
            break
        end
        # apply preconditioning
        PCapply!(pc, cache.z1, r1, args...)
        β = r1' * z1 / (r0' * z0)
        @. p0 = z1 + β * p0
        iter += 1
    end

    (converged, iter, residual_norm)
end
