```@meta
CurrentModule = PosteriorStats
```

# PosteriorStats

PosteriorStats implements widely-used and well-characterized statistical analyses for the Bayesian workflow.
These functions generally estimate properties of posterior and/or posterior predictive distributions.
The default implementations defined here operate on Monte Carlo samples.

See the [API](@ref) for details.
## Extending this package

The methods defined here are intended to be extended by two types of packages.
- packages that implement data types for storing Monte Carlo samples
- packages that implement other representations for posterior distributions than Monte Carlo draws
