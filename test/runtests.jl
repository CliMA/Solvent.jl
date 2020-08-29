group = get(ARGS,1,"default")

if group == "default"
    include("gmres.jl")
    include("gcr.jl")
    include("cg.jl")
elseif group == "gpu"
    include("gmres_gpu.jl")
end
