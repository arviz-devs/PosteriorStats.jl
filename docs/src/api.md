# API

```@index
Pages = ["stats.md"]
```

## Summary statistics

```@docs
```

## General statistics

```@docs
hdi
hdi!
r2_score
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
```

### Utilities

```@docs
PosteriorStats.smooth_data
```
