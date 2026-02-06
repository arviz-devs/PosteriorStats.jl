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
information_criterion
loo
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

## Utilities

```@docs
PosteriorStats.kde_reflected
PosteriorStats.pointwise_conditional_loglikelihoods
```
