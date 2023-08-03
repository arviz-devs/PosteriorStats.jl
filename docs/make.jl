using PosteriorStats
using Documenter

DocMeta.setdocmeta!(PosteriorStats, :DocTestSetup, :(using PosteriorStats); recursive=true)

makedocs(;
    modules=[PosteriorStats],
    repo="https://github.com/arviz-devs/PosteriorStats.jl/blob/{commit}{path}#{line}",
    sitename="PosteriorStats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
