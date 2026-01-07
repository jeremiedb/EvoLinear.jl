using Documenter
using DocumenterVitePress
using EvoLinear

makedocs(;
    sitename="EvoLinear.jl",
    authors="Jeremie Desgagne-Bouchard",
    modules=[EvoLinear],
    format=DocumenterVitepress.MarkdownVitepress(
        repo="github.com/jeremiedb/EvoLinear.jl",
        devbranch="main",
        devurl="dev"
    ),
    warnonly=true,
    checkdocs=:all,
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ]
)

DocumenterVitepress.deploydocs(
    repo="github.com/jeremiedb/EvoLinear.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)
