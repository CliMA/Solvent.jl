push!(LOAD_PATH, joinpath(dirname(dirname(@__DIR__)), "PETSc.jl"))
push!(LOAD_PATH, dirname(@__DIR__))
include("benchmark.jl")
