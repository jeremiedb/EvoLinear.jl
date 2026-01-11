using Documenter
using DocumenterVitepress
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
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    warnonly=true,
    checkdocs=:all,
)

DocumenterVitepress.deploydocs(
    repo="github.com/jeremiedb/EvoLinear.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)
