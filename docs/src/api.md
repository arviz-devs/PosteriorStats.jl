# API

```@index
Pages = ["stats.md"]
```

## Summary statistics

```@docs
SummaryStats
summarize
default_summary_stats
```

## Credible intervals

```@docs
hdi
hdi!
eti
eti!
```

## LOO

```@docs
AbstractELPDResult
PSISLOOResult
elpd_estimates
loo
```

## Model comparison

```@docs
ModelComparisonResult
compare
```

The following model weighting methods are available
```@docs
BootstrappedPseudoBMA
PseudoBMA
Stacking
```

## Predictive checks

```@docs
loo_pit
r2_score
```

## Utilities

```@docs
PosteriorStats.kde_reflected
PosteriorStats.pointwise_conditional_loglikelihoods
```
