push!(LOAD_PATH,"..")

using Documenter
using Solvent

makedocs(
    sitename = "Solvent",
    format = Documenter.HTML(),
    modules = [Solvent],
    pages = [
        "index.md",
        "algorithms.md",
        "preconditioners.md",
        "Background" => [
            "background/ConjugateGradient.md",
            "background/GeneralizedMinimumResidual.md",
            "background/GeneralizedConjugateResidual.md",
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/CliMA/Solvent.jl.git",
    push_preview = true,
)