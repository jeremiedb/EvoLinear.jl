using Documenter
using EvoLinear

makedocs(
    sitename="EvoLinear.jl",
    authors="Jeremie Desgagne-Bouchard",
    modules=[EvoLinear],
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
        "Internals" => "internals.md",
    ],
    format=Documenter.HTML(
        sidebar_sitename=true,
        edit_link="main",
        assets=["assets/style.css"]
    )
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/jeremiedb/EvoLinear.jl.git",
    target="build",
    devbranch="main"
)
