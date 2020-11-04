
export ConjugateGradientMethod

"""
    ConjugateGradientMethod(; M=30)

# CG
This object represents an iterative Krylov method for solving a linear system.
The constructor parameter `M` is the number maximum number of iterations. This
uses the preconditioned Conjugate Gradient method of Barrett et al. (1994).

## References
    @book{barrett1994cg,
    title = {Templates for the Solution of Linear Systems: Building Blocks for Iterative Methods},
    author = {
            Barrett, Richard and Berry, Michael and Chan, Tony F. and Demmel, James
            and Donato, June and Dongarra, Jack and Eijkhout, Victor and Pozo, Roldan
            and Romine, Charles and van der Vorst, Henk
        },
    pages={12-15},
    year = {1994},
    publisher = {SIAM},
    doi = {10.1137/1.9781611971538}
    }
"""
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

    cache = solver.cache
    r = cache.r
    z = cache.z
    p = cache.p

    linearoperator!(r, Q, args...)
    r .= Qrhs .- r

    residual_norm = norm(r)
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
        residual_norm = norm(r)
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
