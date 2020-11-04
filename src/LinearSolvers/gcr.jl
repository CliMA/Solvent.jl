
export GeneralizedConjugateResidualMethod

"""
    GeneralizedConjugateResidualMethod(; M=30, K=10)

# Conjugate Residual
This object represents an iterative Krylov method for solving a linear system.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `K` is the maximal number of restarts. 
The amount of memory required by the solver state is roughly `(2K + 2) * N`, where `N`
is the the number of unknowns. This uses the restarted Generalized Conjugate
Residual method of Eisenstat (1983).

## References
    @article{eisenstat1983variational,
        title={Variational iterative methods for nonsymmetric systems of linear equations},
        author={Eisenstat, Stanley C and Elman, Howard C and Schultz, Martin H},
        journal={SIAM Journal on Numerical Analysis},
        volume={20},
        number={2},
        pages={345--357},
        year={1983},
        publisher={SIAM}
    }
"""
struct GeneralizedConjugateResidualMethod <: AbstractKrylovMethod
    "Maximum number of Krylov iterations"
    M::Int
    "Maximum number of restarts"
    K::Int
    function GeneralizedConjugateResidualMethod(; M=30, K=10)
        return new(M, K)
    end
end

mutable struct GCRCache{M, T, AT} <: AbstractLinearSolverCache
    residual::AT
    L_residual::AT
    p::NTuple{M, AT}
    L_p::NTuple{M, AT}
    alpha::Vector{T}
    normsq::Vector{T}
end

function cache(
    krylov_alg::GeneralizedConjugateResidualMethod,
    Q::AT,
) where {AT}
    T = eltype(Q)
    M = krylov_alg.M
    residual = similar(Q)
    L_residual = similar(Q)
    p = ntuple(i -> similar(Q), M)
    L_p = ntuple(i -> similar(Q), M)
    alpha = zeros(M)
    normsq = zeros(M)

    return GCRCache{M, T, AT}(
        residual,
        L_residual,
        p,
        L_p,
        alpha,
        normsq,
    )
end

function LSinitialize!(
    ::GeneralizedConjugateResidualMethod,
    solver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    linearoperator! = solver.linop!
    pc = solver.pc

    cache = solver.cache
    residual = cache.residual
    p = cache.p
    L_p = cache.L_p

    @assert size(Q) == size(residual)

    rtol = solver.rtol
    atol = solver.atol

    threshold = rtol * norm(Qrhs)
    linearoperator!(residual, Q, args...)
    @. residual -= Qrhs

    converged = false
    residual_norm = norm(residual)
    if residual_norm < threshold
        converged = true
        return converged, threshold
    end

    @. p[1] = residual
    linearoperator!(L_p[1], p[1], args...)

    threshold = max(atol, threshold)

    converged, threshold
end

function LSsolve!(
    krylov_alg::GeneralizedConjugateResidualMethod,
    solver::LinearSolver,
    threshold,
    Q,
    Qrhs,
    args...,
)
    converged = false
    iter = 0
    total_iters = 0
    residual_norm = typemax(eltype(Q))

    while !converged && iter < krylov_alg.K
        converged, cycle_iters, residual_norm = gcr_cycle!(solver, threshold, Q, Qrhs, args...)

        iter += 1
        total_iters += cycle_iters

        # If we blow up, we want to know about it
        if !isfinite(residual_norm)
            error("norm of residual is not finite after $total_iters GCR iterations.")
        end
    end

    (converged, total_iters, residual_norm)
end

function gcr_cycle!(
    solver::LinearSolver,
    threshold,
    Q,
    Qrhs,
    args...,
)
    linearoperator! = solver.linop!
    pc = solver.pc
    cache = solver.cache
    residual = cache.residual
    p = cache.p
    L_residual = cache.L_residual
    L_p = cache.L_p
    normsq = cache.normsq
    alpha = cache.alpha
    rtol = solver.rtol
    atol = solver.atol

    converged = false
    residual_norm = typemax(eltype(Q))
    k = 1
    M = solver.krylov_alg.M
    for outer k in 1:M

        normsq[k] = norm(L_p[k])^2
        beta = -dot(residual, L_p[k]) / normsq[k]

        @. Q += beta * p[k]
        @. residual += beta * L_p[k]

        residual_norm = norm(residual)

        if residual_norm < threshold
            converged = true
            break
        end

        linearoperator!(L_residual, residual, args...)

        for l in 1:k
            alpha[l] = -dot(L_residual, L_p[l]) / normsq[l]
        end

        if k < M
            rv_nextp = realview(p[k + 1])
            rv_L_nextp = realview(L_p[k + 1])
        else # restart
            rv_nextp = realview(p[1])
            rv_L_nextp = realview(L_p[1])
        end

        rv_residual = realview(residual)
        rv_p = realview.(p)
        rv_L_p = realview.(L_p)
        rv_L_residual = realview(L_residual)

        groupsize = 256
        T = eltype(alpha)

        event = Event(array_device(Q))
        event = linearcombination!(array_device(Q), groupsize)(
            rv_nextp,
            (one(T), alpha[1:k]...),
            (rv_residual, rv_p[1:k]...),
            false;
            ndrange = length(rv_nextp),
            dependencies = (event,),
        )

        event = linearcombination!(array_device(Q), groupsize)(
            rv_L_nextp,
            (one(T), alpha[1:k]...),
            (rv_L_residual, rv_L_p[1:k]...),
            false;
            ndrange = length(rv_nextp),
            dependencies = (event,),
        )
        wait(array_device(Q), event)
    end

    (converged, k, residual_norm)
end
