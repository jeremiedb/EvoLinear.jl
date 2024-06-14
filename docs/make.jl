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

deploydocs(
    repo="github.com/jeremiedb/EvoLinear.jl",
    target="build", # this is where Vitepress stores its output
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)
