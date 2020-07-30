using Dilemma
using Documenter

makedocs(;
    modules=[Dilemma],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/leferrad/Dilemma.jl/blob/{commit}{path}#L{line}",
    sitename="Dilemma.jl",
    authors="Leandro Ferrado",
    strict=false
)

deploydocs(;
    repo="github.com/leferrad/Dilemma.jl",
)
