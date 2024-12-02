using PosteriorStats
using Documenter
using DocumenterInterLinks

DocMeta.setdocmeta!(PosteriorStats, :DocTestSetup, :(using PosteriorStats); recursive=true)

makedocs(;
    modules=[PosteriorStats],
    repo=Remotes.GitHub("arviz-devs", "PosteriorStats.jl"),
    sitename="PosteriorStats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true", edit_link="main", assets=String[]
    ),
    pages=["Home" => "index.md", "API" => "api.md"],
    warnonly=[:footnote, :missing_docs],
)

deploydocs(; repo="github.com/arviz-devs/PosteriorStats.jl.git", devbranch="main")
