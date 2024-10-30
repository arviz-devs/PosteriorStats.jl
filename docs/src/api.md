# API

```@index
Pages = ["stats.md"]
```

## Summary statistics

```@docs
SummaryStats
default_diagnostics
default_stats
default_summary_stats
summarize
```

## Credible intervals

```@docs
hdi
hdi!
eti
eti!
```

## LOO and WAIC

```@docs
AbstractELPDResult
PSISLOOResult
WAICResult
elpd_estimates
information_criterion
loo
waic
```

## Model comparison

```@docs
ModelComparisonResult
compare
model_weights
```

The following model weighting methods are available
```@docs
AbstractModelWeightsMethod
BootstrappedPseudoBMA
PseudoBMA
Stacking
```

## Predictive checks

```@docs
loo_pit
r2_score
```

### Utilities

```@docs
PosteriorStats.kde_reflected
PosteriorStats.smooth_data
```
