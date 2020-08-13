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

mutable struct BenchmarkAccumulator
    runs::Int64
    mintime::Float64
    maxtime::Float64
    tottime::Float64
    minallocs::Float64
    maxallocs::Float64
    totallocs::Float64
    minres::Float64
    maxres::Float64
    totres::Float64

    function BenchmarkAccumulator(time::AbstractFloat, allocs::AbstractFloat,
            res::AbstractFloat)
        return new(1, time, time, time, allocs, allocs, allocs, res, res, res)
    end
end

mutable struct Benchmarker
    storage::Dict{String, BenchmarkAccumulator}
    disabled::Bool

    function Benchmarker(disabled::Bool = true)
        return new(Dict{String, BenchmarkAccumulator}(), disabled)
    end
end

function recordbenchmark(benchmarker::Benchmarker, name::String,
        time::AbstractFloat, allocs::AbstractFloat, res::AbstractFloat)
    if benchmarker.disabled
        return
    end
    storage = benchmarker.storage
    if haskey(storage, name)
        ba = storage[name]
        ba.runs += 1
        ba.tottime += time
        ba.totallocs += allocs
        ba.totres += res
        if time < ba.mintime
            ba.mintime = time
        elseif time > ba.maxtime
            ba.maxtime = time
        end
        if allocs < ba.minallocs
            ba.minallocs = allocs
        elseif allocs > ba.maxallocs
            ba.maxallocs = allocs
        end
        if res < ba.minres
            ba.minres = res
        elseif res > ba.maxres
            ba.maxres = res
        end
    else
        storage[name] = BenchmarkAccumulator(time, allocs, res)
    end
end

function printbenchmarks(benchmarker::Benchmarker)
    storage = benchmarker.storage
    maxnamelen = maximum([length(name) for name in keys(storage)])
    println(rpad("Benchmark Name", maxnamelen),
        "| Runs |     Time (ms)     |   Allocs (MiB)    |     Residual     ")
    println(" " ^ maxnamelen,
        "|      |  Min   Max   Avg  |  Min   Max   Avg  |  Min   Max   Avg ")
    println("-" ^ (maxnamelen + 63))
    for name in sort!(collect(keys(storage)))
        ba = storage[name]
        print(rpad(name, maxnamelen + 3))
        @printf(
            "%-3d   %5.1f %5.1f %5.1f   %5.1f %5.1f %5.1f   %.0e %.0e %.0e\n",
            ba.runs,
            ba.mintime, ba.maxtime, ba.tottime / ba.runs,
            ba.minallocs, ba.maxallocs, ba.totallocs / ba.runs,
            ba.minres, ba.maxres, ba.totres / ba.runs)
    end
end

function sparsesetup(::Type{T}, n::Int, α::Number=0.01,
        density::AbstractFloat=0.05) where T
    return BenchmarkSetup(I + T(α) * sprandn(T, n, n, density))
end

function laplacesetup(::Type{T}, nx::Int, ny::Int=nx, lx::Int=1,
        ly::Int=lx) where T
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
        setupname::String, iters::Int, tol::AbstractFloat)
    A = bs.A
    b = bs.b
    xexact = bs.xexact
    xinit = bs.xinit
    algnames = ["ConjGrad", "IterativeSolvers", "Krylov", "KrylovKit",
        "Solvent"]
    times = Array{Float64, 1}(undef, 5)
    allocs = Array{Int64, 1}(undef, 5)
    xfinals = [copy(xinit) for _ in 1:5]
    _, times[1], allocs[1], _, _ = @timed begin
        ConjGrad.cg!((x) -> (A * x), b, xfinals[1]; tol = Float64(tol),
            maxIter = iters)
    end
    _, times[2], allocs[2], _, _ = @timed begin
        IterativeSolvers.cg!(xfinals[2], A, b, tol = tol, maxiter = iters)
    end
    _, times[3], allocs[3], _, _ = @timed begin
        xfinals[3] = Krylov.cg(A, b; atol = tol, rtol = tol, itmax = iters)[1]
    end
    _, times[4], allocs[4], _, _ = @timed begin
        algorithm = KrylovKit.CG(maxiter = iters, tol = tol)
        xfinals[4] = KrylovKit.linsolve(A, b, xinit, algorithm)[1]
    end
    _, times[5], allocs[5], _, _ = @timed begin
        linearsolver = Solvent.LinearSolver(
            (y, x) -> (y .= A * x),
            Solvent.ConjugateGradientMethod(M = iters),
            xinit;
            pc_alg = Solvent.Identity(pc_side=Solvent.PCleft()),
            rtol = tol,
            atol = tol,
        )
        Solvent.linearsolve!(linearsolver, xfinals[5], b)
    end
    prefix = join(["CG", setupname, eltype(xexact), size(A)[1], iters], '-')
    for i in 1:5
        recordbenchmark(benchmarker, string(prefix, '-', algnames[i]),
            times[i] * 1000, allocs[i] / 2^20, norm(xfinals[i] - xexact))
    end
end

function benchmarkgmres!(benchmarker::Benchmarker, bs::BenchmarkSetup,
        setupname::String, dims::Int, iters::Int, tol::AbstractFloat)
    A = bs.A
    b = bs.b
    xexact = bs.xexact
    xinit = bs.xinit
    algnames = ["IterativeSolvers", "Krylov", "KrylovKit", "Solvent"]
    times = Array{Float64, 1}(undef, 4)
    allocs = Array{Int64, 1}(undef, 4)
    xfinals = [copy(xinit) for _ in 1:4]
    _, times[1], allocs[1], _, _ = @timed begin
        IterativeSolvers.gmres!(xfinals[1], A, b, tol = tol, restart = dims,
            maxiter = iters)
    end
    _, times[2], allocs[2], _, _ = @timed begin
        xfinals[2] = Krylov.dqgmres(A, b; atol = tol, rtol = tol,
            memory = dims, itmax = iters)[1]
    end
    _, times[3], allocs[3], _, _ = @timed begin
        algorithm = KrylovKit.GMRES(krylovdim = dims, maxiter = iters,
            tol = tol)
        xfinals[3] = KrylovKit.linsolve(A, b, xinit, algorithm)[1]
    end
    _, times[4], allocs[4], _, _ = @timed begin
        linearsolver = Solvent.LinearSolver(
            (y, x) -> (y .= A * x),
            Solvent.GeneralizedMinimalResidualMethod(M = dims, K = iters),
            xinit;
            pc_alg = Solvent.Identity(pc_side=Solvent.PCright()),
            rtol = tol,
            atol = tol,
        )
        Solvent.linearsolve!(linearsolver, xfinals[4], b)
    end
    prefix = join(["GMRES", setupname, eltype(xexact), size(A)[1], dims, iters],
        '-')
    for i in 1:4
        recordbenchmark(benchmarker, string(prefix, '-', algnames[i]),
            times[i] * 1000, allocs[i] / 2^20, norm(xfinals[i] - xexact))
    end
end

const b = Benchmarker()
for T in [Float32, Float64]
    tol = sqrt(eps(T))
    for i in 0:5
        if i == 0
            b.disabled = true
        end
        benchmarkcg!(b, laplacesetup(T, 100), "Laplace", 5000, tol)
        benchmarkgmres!(b, laplacesetup(T, 100), "Laplace", 500, 10, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 20, 10, tol)
        benchmarkgmres!(b, sparsesetup(T, 10000), "Sparse", 10, 20, tol)
        if i == 0
            b.disabled = false
        end
    end
end
printbenchmarks(b)