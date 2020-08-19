using CUDA
using KernelAbstractions
using LinearAlgebra
using SparseArrays
using Random
using Printf
using TimerOutputs

import ConjGrad
import IterativeSolvers 
import Krylov
import KrylovKit
import Solvent

struct BenchmarkSetup{AT, T}
    A::AT
    b::Vector{T}
    xexact::Vector{T}
    xinit::Vector{T}

    function BenchmarkSetup(A::AbstractArray{<:Number})
        T = eltype(A)
        _, ncols = size(A)
        xexact = rand(T, ncols)
        xinit = rand(T, ncols)
        return new{typeof(A), T}(A, A * xexact, xexact, xinit)
    end
end

mutable struct StatTracker{T<:Number}
    min::T
    max::T
    tot::T

    function StatTracker(value::T) where T<:Number
        return new{T}(value, value, value)
    end
end

function addstat!(st::StatTracker{T}, value::T) where T<:Number
    st.tot += value
    if value < st.min
        st.min = value
    elseif value > st.max
        st.max = value
    end
end

mutable struct BenchmarkTracker
    runs::Int
    timetracker::StatTracker
    alloctracker::StatTracker
    itertracker::StatTracker
    restracker::StatTracker

    function BenchmarkTracker(time::AbstractFloat, allocs::AbstractFloat,
            iters::Integer, res::AbstractFloat)
        return new(1, StatTracker(time), StatTracker(allocs),
            StatTracker(iters), StatTracker(res))
    end
end

function addbenchmark!(bt::BenchmarkTracker, time::AbstractFloat,
        allocs::AbstractFloat, iters::Integer, res::AbstractFloat)
    bt.runs += 1
    addstat!(bt.timetracker, time)
    addstat!(bt.alloctracker, allocs)
    addstat!(bt.itertracker, iters)
    addstat!(bt.restracker, res)
end

mutable struct Benchmarker
    storage::Dict{String, BenchmarkTracker}
    enabled::Bool

    function Benchmarker(enabled::Bool = true)
        return new(Dict{String, BenchmarkTracker}(), enabled)
    end
end

function enable!(b::Benchmarker)
    b.enabled = true
end

function disable!(b::Benchmarker)
    b.enabled = false
end

function recordbenchmark!(b::Benchmarker, name::String, time::AbstractFloat,
        allocs::AbstractFloat, iters::Integer, res::AbstractFloat)
    if b.enabled
        if haskey(b.storage, name)
            addbenchmark!(b.storage[name], time, allocs, iters, res)
        else
            b.storage[name] = BenchmarkTracker(time, allocs, iters, res)
        end
    end
end

function printbenchmarks(b::Benchmarker)
    storage = b.storage
    maxnamelen = maximum([length(name) for name in keys(storage)])
    println(rpad("Benchmark Name", maxnamelen),
        "| Runs |       Time (ms)      |    Allocs (MiB)   |    Iters    |     Residual     ")
    println(" " ^ maxnamelen,
        "|      |   Min    Max    Avg  |  Min   Max   Avg  | Min Max Avg |  Min   Max   Avg ")
    println("-" ^ (maxnamelen + 83))
    for name in sort!(collect(keys(storage)))
        bt = storage[name]
        print(rpad(name, maxnamelen + 3))
        @printf(
            "%-3d   %6.1f %6.1f %6.1f   %5.1f %5.1f %5.1f   %3d %3d %3d   %.0e %.0e %.0e\n",
            bt.runs,
            bt.timetracker.min, bt.timetracker.max, bt.timetracker.tot / bt.runs,
            bt.alloctracker.min, bt.alloctracker.max, bt.alloctracker.tot / bt.runs,
            bt.itertracker.min, bt.itertracker.max, bt.itertracker.tot / bt.runs,
            bt.restracker.min, bt.restracker.max, bt.restracker.tot / bt.runs)
    end
end

function sparsesetup(::Type{T}, n::Integer, α::Number=0.01,
        density::AbstractFloat=0.05) where T
    return BenchmarkSetup(I + T(α) * sprandn(T, n, n, density))
end

function laplacesetup(::Type{T}, nx::Integer, ny::Integer=nx, lx::Integer=1,
        ly::Integer=lx) where T
    dx = T(lx) / T(nx + 1)
    dy = T(ly) / T(ny + 1)
    Dx = [[T(1) spzeros(T, 1, nx - 1)]; spdiagm(1=>ones(T, nx - 1)) - I] / dx
    Dy = [[T(1) spzeros(T, 1, ny - 1)]; spdiagm(1=>ones(T, ny - 1)) - I] / dy
    Ax = Dx' * Dx
    Ay = Dy' * Dy
    A = kron(sparse(I, ny, ny), Ax) + kron(Ay, sparse(I, nx, nx))
    return BenchmarkSetup(A)
end

function benchmarkcg!(benchmarker::Benchmarker, bs::BenchmarkSetup,
        setupname::String, maxiters::Integer, tol::AbstractFloat)
    A = bs.A
    b = bs.b
    xexact = bs.xexact
    xinit = bs.xinit
    algnames = ["ConjGrad", "IterativeSolvers", "Krylov", "KrylovKit",
        "Solvent"]
    times = Array{Float64, 1}(undef, 5)
    allocs = Array{Int, 1}(undef, 5)
    iters = Array{Int, 1}(undef, 5)
    xfinals = [copy(xinit) for _ in 1:5]
    result, times[1], allocs[1], _, _ = @timed begin
        ConjGrad.cg!((x) -> (A * x), b, xfinals[1]; tol = Float64(tol),
            maxIter = maxiters)
    end
    iters[1] = result[2]
    result, times[2], allocs[2], _, _ = @timed begin
        IterativeSolvers.cg!(xfinals[2], A, b, tol = tol, maxiter = maxiters,
            log = true)
    end
    iters[2] = result[2].iters
    result, times[3], allocs[3], _, _ = @timed begin
        Krylov.cg(A, b; atol = tol, rtol = tol, itmax = maxiters)
    end
    iters[3] = length(result[2].residuals) - 1 # 1st residual found before loop.
    xfinals[3] = result[1]
    result, times[4], allocs[4], _, _ = @timed begin
        KrylovKit.linsolve(A, b, xinit; isposdef = true, krylovdim = 1,
            maxiter = maxiters, atol = tol, rtol = tol)
    end
    iters[4] = result[2].numops - 1 # Linear operator applied once before loop.
    xfinals[4] = result[1]
    iters[5], times[5], allocs[5], _, _ = @timed begin
        linearsolver = Solvent.LinearSolver(
            (y, x) -> (y .= A * x),
            Solvent.ConjugateGradientMethod(M = maxiters),
            xinit;
            pc_alg = Solvent.Identity(pc_side=Solvent.PCleft()),
            rtol = tol,
            atol = tol,
        )
        Solvent.linearsolve!(linearsolver, xfinals[5], b)
    end
    prefix = join(["CG", setupname, eltype(xexact), size(A)[1], maxiters], '-')
    for i in 1:5
        recordbenchmark!(benchmarker, string(prefix, '-', algnames[i]),
            times[i] * 1000, allocs[i] / 2^20, iters[i],
            norm(xfinals[i] - xexact))
    end
end

function benchmarkgmres!(benchmarker::Benchmarker, bs::BenchmarkSetup,
        setupname::String, dims::Integer, maxiters::Integer, tol::AbstractFloat)
    A = bs.A
    b = bs.b
    xexact = bs.xexact
    xinit = bs.xinit
    algnames = ["IterativeSolvers", "Krylov", "KrylovKit", "Solvent"]
    times = Array{Float64, 1}(undef, 4)
    allocs = Array{Int, 1}(undef, 4)
    iters = Array{Int, 1}(undef, 4)
    xfinals = [copy(xinit) for _ in 1:4]
    result, times[1], allocs[1], _, _ = @timed begin
        IterativeSolvers.gmres!(xfinals[1], A, b, tol = tol, restart = dims,
            maxiter = dims * maxiters, log = true)
    end
    iters[1] = result[2].iters
    result, times[2], allocs[2], _, _ = @timed begin
        Krylov.dqgmres(A, b; atol = tol, rtol = tol, itmax = dims * maxiters)
    end
    iters[2] = length(result[2].residuals) - 1 # 1st residual found before loop.
    xfinals[2] = result[1]
    result, times[3], allocs[3], _, _ = @timed begin
        KrylovKit.linsolve(A, b, xinit; isposdef = false, krylovdim = dims,
            maxiter = maxiters, atol = tol, rtol = tol)
    end
    iters[3] = result[2].numops - 2 # Linear operator applied twice before loop.
    xfinals[3] = result[1]
    iters[4], times[4], allocs[4], _, _ = @timed begin
        linearsolver = Solvent.LinearSolver(
            (y, x) -> (y .= A * x),
            Solvent.GeneralizedMinimalResidualMethod(M = dims, K = maxiters),
            xinit;
            pc_alg = Solvent.Identity(pc_side=Solvent.PCright()),
            rtol = tol,
            atol = tol,
        )
        Solvent.linearsolve!(linearsolver, xfinals[4], b)
    end
    prefix = join(["GMRES", setupname, eltype(xexact), size(A)[1], dims,
        maxiters], '-')
    for i in 1:4
        recordbenchmark!(benchmarker, string(prefix, '-', algnames[i]),
            times[i] * 1000, allocs[i] / 2^20, iters[i],
            norm(xfinals[i] - xexact))
    end
end

const b = Benchmarker()
for T in [Float32, Float64]
    tol = sqrt(eps(T))
    for i in 0:100
        i == 0 && disable!(b)
        benchmarkcg!(b, laplacesetup(T, 100), "Laplace", 5000, tol)
        benchmarkgmres!(b, laplacesetup(T, 100), "Laplace", 500, 10, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 14, 1, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 12, 3, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 11, 5, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 10, 8, tol)
        # Solvent does not converge with M = 9, even when K = 10000.
        benchmarkgmres!(b, sparsesetup(T, 10123), "Sparse", 10, 90, tol)
        i == 0 && enable!(b)
    end
end
printbenchmarks(b)