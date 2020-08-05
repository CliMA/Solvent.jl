
export GeneralizedMinimalResidualMethod

struct GeneralizedMinimalResidualMethod <: AbstractKrylovMethod
    "Maximum number of Krylov iterations"
    M::Int
    "Maximum number of restarts"
    K::Int
    function GeneralizedMinimalResidualMethod(; M=30, K=10)
        return new(M, K)
    end
end

mutable struct GMRESCache{M, MP1, MMP1, T, AT} <: AbstractLinearSolverCache
    krylov_basis::NTuple{MP1, AT}
    "Hessenberg matrix"
    H::MArray{Tuple{MP1, M}, T, 2, MMP1}
    "rhs of the least squares problem"
    g0::MArray{Tuple{MP1, 1}, T, 2, MP1}
    "work vector for preconditioning"
    Wvec::AT
end

function cache(
    krylov_alg::GeneralizedMinimalResidualMethod,
    Q::AT,
) where {AT}
    max_iter = krylov_alg.M
    Wvec = similar(Q)
    krylov_basis = ntuple(i -> similar(Q), max_iter + 1)
    H = @MArray zeros(max_iter + 1, max_iter)
    g0 = @MArray zeros(max_iter + 1)

    return GMRESCache{max_iter, max_iter + 1, max_iter * (max_iter + 1), eltype(Q), AT}(
        krylov_basis,
        H,
        g0,
        Wvec,
    )
end

function LSinitialize!(
    ::GeneralizedMinimalResidualMethod,
    solver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    linearoperator! = solver.linop!
    pc = solver.pc
    PCinitialize!(pc, Q, Qrhs, args...)

    cache = solver.cache
    g0 = cache.g0
    krylov_basis = cache.krylov_basis
    atol = solver.atol

    @assert size(Q) == size(krylov_basis[1])

    # store the initial residual in krylov_basis[1]
    linearoperator!(krylov_basis[1], Q, args...)
    @. krylov_basis[1] = Qrhs - krylov_basis[1]

    # apply left preconditioning
    if isa(pc.pc_side, PCleft)
        PCapply!(pc, cache.Wvec, krylov_basis[1], args...)
        @. krylov_basis[1] = cache.Wvec
    end

    residual_norm = norm(krylov_basis[1], weighted_norm)

    converged = false
    if residual_norm < solver.atol
        converged = true
        return converged, residual_norm
    end

    fill!(g0, 0)
    g0[1] = residual_norm
    krylov_basis[1] ./= residual_norm

    converged, max(residual_norm, atol)
end

function LSsolve!(
    krylov_alg::GeneralizedMinimalResidualMethod,
    solver::LinearSolver,
    Q,
    Qrhs,
    args...,
)
    linearoperator! = solver.linop!
    pc = solver.pc
    cache = solver.cache
    krylov_basis = cache.krylov_basis
    H = cache.H
    g0 = cache.g0
    max_iter = krylov_alg.M
    max_restart_iter = krylov_alg.K
    rtol = solver.rtol
    atol = solver.atol

    converged = false
    residual_norm = typemax(eltype(Q))
    residual_norm0 = norm(krylov_basis[1], weighted_norm)

    Ω = LinearAlgebra.Rotation{eltype(Q)}([])
    j = 1
    k = 1
    total_iters = 1
    while !converged && k < max_restart_iter
        for outer j in 1:max_iter
            # apply right preconditioning
            if isa(pc.pc_side, PCright)
                PCapply!(pc, cache.Wvec, krylov_basis[j], args...)
                @. krylov_basis[j] = cache.Wvec
            end
            # apply the linear operator
            linearoperator!(krylov_basis[j + 1], krylov_basis[j], args...)
            # apply left preconditioning
            if isa(pc.pc_side, PCleft)
                PCapply!(pc, cache.Wvec, krylov_basis[j + 1], args...)
                @. krylov_basis[j + 1] = cache.Wvec
            end

            # Arnoldi using the Modified Gram Schmidt orthonormalization
            for i in 1:j
                H[i, j] = dot(krylov_basis[j + 1], krylov_basis[i], weighted_norm)
                @. krylov_basis[j + 1] -= H[i, j] * krylov_basis[i]
            end
            H[j + 1, j] = norm(krylov_basis[j + 1], weighted_norm)
            krylov_basis[j + 1] ./= H[j + 1, j]

            # apply the previous Givens rotations to the new column of H
            @views H[1:j, j:j] .= Ω * H[1:j, j:j]

            # compute a new Givens rotation to zero out H[j + 1, j]
            G, _ = givens(H, j, j + 1, j)

            # apply the new rotation to H and the rhs
            H .= G * H
            g0 .= G * g0

            # compose the new rotation with the others
            Ω = lmul!(G, Ω)

            residual_norm = abs(g0[j + 1])
            residual_reduction = residual_norm / residual_norm0

            if residual_norm < atol || residual_reduction < rtol
                converged = true
                break
            end
        end

        # If we blow up, we want to know about it
        if !isfinite(residual_norm)
            error("norm of residual is not finite after $j iterations.")
        end

        # solve the triangular system
        y = SVector{j}(@views UpperTriangular(H[1:j, 1:j]) \ g0[1:j])

        ## compose the solution
        rv_Q = realview(Q)
        rv_krylov_basis = realview.(krylov_basis)
        groupsize = 256
        event = Event(array_device(Q))
        event = linearcombination!(array_device(Q), groupsize)(
            rv_Q,
            y,
            rv_krylov_basis,
            true;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)
        # unwind right-preconditioning
        if isa(pc.pc_side, PCright)
            PCapply!(pc, cache.Wvec, Q, args...)
            @. Q = cache.Wvec
        end

        # if not converged, we restart
        if !converged
            # Reinitialize with current Q
            _, residual_norm = LSinitialize!(
                krylov_alg,
                solver,
                linearoperator!,
                Q,
                Qrhs,
                args...,
            )
        end
        k += 1
        total_iters += j
    end
    (converged, total_iters, residual_norm)
end
