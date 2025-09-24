using PosteriorStats
using Distributions
using Documenter
using DocumenterCitations
using DocumenterInterLinks

DocMeta.setdocmeta!(PosteriorStats, :DocTestSetup, :(using PosteriorStats); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:numeric)

links = InterLinks(
    "Distributions" => "https://juliastats.org/Distributions.jl/stable/",
    "IntervalSets" => (
        "https://juliamath.github.io/IntervalSets.jl/stable/",
        joinpath(@__DIR__, "inventories", "IntervalSets.toml"),
    ),
    "MCMCDiagnosticTools" => "https://julia.arviz.org/MCMCDiagnosticTools/stable/",
    "PSIS" => "https://julia.arviz.org/PSIS/stable/",
    "Statistics" => "https://docs.julialang.org/en/v1/",
    "StatsBase" => (
        "https://juliastats.org/StatsBase.jl/stable/",
        "https://juliastats.org/StatsBase.jl/dev/objects.inv",
    ),
)

# Increase the terminal width from 80 to 90 chars to avoid column truncation
withenv("COLUMNS" => 90) do
    makedocs(;
        modules=[PosteriorStats],
        repo=Remotes.GitHub("arviz-devs", "PosteriorStats.jl"),
        sitename="PosteriorStats.jl",
        format=Documenter.HTML(;
            prettyurls=get(ENV, "CI", "false") == "true",
            edit_link="main",
            assets=[joinpath("assets", "citations.css")],
        ),
        pages=["Home" => "index.md", "API" => "api.md", "References" => "references.md"],
        plugins=[bib, links],
    )
end

deploydocs(;
    repo="github.com/arviz-devs/PosteriorStats.jl.git", devbranch="main", push_preview=true
)
