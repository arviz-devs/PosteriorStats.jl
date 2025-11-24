# Change Log

## v0.x.x (Unreleased)

### BREAKING CHANGES

### Features

### Fixes

### Maintenance

### Documentation

## v0.4.3 (2025-11-24)

### Features

- Support PrettyTables v3 (v2 temporarily still supported) ([#83](https://github.com/arviz-devs/PosteriorStats.jl/pull/83))

### Maintenance

- Store SummaryStats labels separately from data ([#81](https://github.com/arviz-devs/PosteriorStats.jl/pull/81))
- Add historical changelog ([#82](https://github.com/arviz-devs/PosteriorStats.jl/pull/82))
- Add pull request templates ([#82](https://github.com/arviz-devs/PosteriorStats.jl/pull/82))

## v0.4.2 (2025-10-13)

### Maintenance

- Bump actions/checkout from 4 to 5 ([#79](https://github.com/arviz-devs/PosteriorStats.jl/pull/79))
- Reorganize summarize/SummaryStats source ([#80](https://github.com/arviz-devs/PosteriorStats.jl/pull/80))

## v0.4.1 (2025-09-24)

### Fixes

- Fix broken doctests ([#78](https://github.com/arviz-devs/PosteriorStats.jl/pull/78))

## v0.4.0 (2025-09-22)

### BREAKING CHANGES

- Change default CI probability to 0.89 ([#72](https://github.com/arviz-devs/PosteriorStats.jl/pull/72))
- Update `r2_score` to return a credible interval instead of std. ([#73](https://github.com/arviz-devs/PosteriorStats.jl/pull/73))
- Remove `waic` ([#75](https://github.com/arviz-devs/PosteriorStats.jl/pull/75))

### Features

- `r2_score` can now optionally return raw scores. ([#73](https://github.com/arviz-devs/PosteriorStats.jl/pull/73))

### Fixes

- Avoid unnecessarily eltype promotion for CIs ([#77](https://github.com/arviz-devs/PosteriorStats.jl/pull/77))

## v0.3.0 (2025-09-10)

### BREAKING CHANGES

- Represent HDI and ETI using IntervalSets ([#36](https://github.com/arviz-devs/PosteriorStats.jl/pull/36))
- Rename `mcse` suffixes to `se` prefixes ([#47](https://github.com/arviz-devs/PosteriorStats.jl/pull/47))
- Refactor summarize ([#63](https://github.com/arviz-devs/PosteriorStats.jl/pull/63))
- Stop smoothing discrete data in loo_pit ([#68](https://github.com/arviz-devs/PosteriorStats.jl/pull/68))
- Remove SummaryStats's fields, type parameters, and indexing interface from the public API ([#69](https://github.com/arviz-devs/PosteriorStats.jl/pull/69))
- Refactor SummaryStats ([#70](https://github.com/arviz-devs/PosteriorStats.jl/pull/70))

### Features

- Make unimodal HDI more efficient ([#38](https://github.com/arviz-devs/PosteriorStats.jl/pull/38))
- Add eti to API ([#39](https://github.com/arviz-devs/PosteriorStats.jl/pull/39))
- Add multimodal HDI ([#40](https://github.com/arviz-devs/PosteriorStats.jl/pull/40))
- Pointwise log-likelihoods for non-factorizable models ([#58](https://github.com/arviz-devs/PosteriorStats.jl/pull/58))
- Add pointwise log-likelihoods for more observation models ([#61](https://github.com/arviz-devs/PosteriorStats.jl/pull/61))
- Reorganize and simplify utilities for showing tables ([#64](https://github.com/arviz-devs/PosteriorStats.jl/pull/64))

### Maintenance

- Run CI on pre-release versions instead of nightlies ([#34](https://github.com/arviz-devs/PosteriorStats.jl/pull/34))
- Support DataInterpolations v7 and v8 ([#44](https://github.com/arviz-devs/PosteriorStats.jl/pull/44), [#50](https://github.com/arviz-devs/PosteriorStats.jl/pull/50))
- Improvements to show method precision ([#46](https://github.com/arviz-devs/PosteriorStats.jl/pull/46))
- Bump test dependency compats ([#51](https://github.com/arviz-devs/PosteriorStats.jl/pull/51))
- Bump julia-actions/julia-downgrade-compat from 1 to 2 ([#57](https://github.com/arviz-devs/PosteriorStats.jl/pull/57))
- Bump Julia lower bound to v1.10 (LTS) ([#59](https://github.com/arviz-devs/PosteriorStats.jl/pull/59))
- Move R-loo reference tests to their own workflow ([#65](https://github.com/arviz-devs/PosteriorStats.jl/pull/65))
- Remove ArviZExampleData as a test dependency ([#66](https://github.com/arviz-devs/PosteriorStats.jl/pull/66))
- Cache R dependencies in CI ([#60](https://github.com/arviz-devs/PosteriorStats.jl/pull/60))

### Documentation

- Add docs interlinks ([#42](https://github.com/arviz-devs/PosteriorStats.jl/pull/42))
- Use DocumenterCitations in docs ([#43](https://github.com/arviz-devs/PosteriorStats.jl/pull/43))
- Fix relative effiency computation in loo docstring ([#67](https://github.com/arviz-devs/PosteriorStats.jl/pull/67))

## v0.2.8 (2025-07-25)

### Documentation

- Update loo_pit docstring example ([#56](https://github.com/arviz-devs/PosteriorStats.jl/pull/56))

## v0.2.7 (2025-04-10)

### Documentation

- Update doctests for new ArviZExampleData ([#52](https://github.com/arviz-devs/PosteriorStats.jl/pull/52))

## v0.2.6 (2025-04-10)

### Maintenance

- Bump Julia lower bound to v1.8 ([#53](https://github.com/arviz-devs/PosteriorStats.jl/pull/53))

## v0.2.5 (2024-08-05)

### Fixes

- Replace generated function with `nameof` ([#33](https://github.com/arviz-devs/PosteriorStats.jl/pull/33))

## v0.2.4 (2024-08-05)

### Maintenance

- Add support for DataInterpolations v5 and v6 ([#29](https://github.com/arviz-devs/PosteriorStats.jl/pull/29), [#31](https://github.com/arviz-devs/PosteriorStats.jl/pull/31))

## v0.2.3 (2024-08-04)

### Maintenance

- Fix downgrade CI workflow ([#32](https://github.com/arviz-devs/PosteriorStats.jl/pull/32))

## v0.2.2 (2024-03-13)

### Documentation

- Update doctests for DimensionalData v0.26 ([#27](https://github.com/arviz-devs/PosteriorStats.jl/pull/27))

## v0.2.1 (2024-02-13)

### Fixes

- Fix compat lower bounds ([#26](https://github.com/arviz-devs/PosteriorStats.jl/pull/26))

### Maintenance

- Run CI against downgraded dependencies ([#26](https://github.com/arviz-devs/PosteriorStats.jl/pull/26))

## v0.2.0 (2023-12-23)

### BREAKING CHANGES

- Refactor SummaryStats to wrap a Table ([#23](https://github.com/arviz-devs/PosteriorStats.jl/pull/23))

### Documentation

- Improve documentation of r2_score ([#22](https://github.com/arviz-devs/PosteriorStats.jl/pull/22))

## v0.1.4 (2023-10-29)

### Maintenance

- Support building docs with Documenter v1 ([#17](https://github.com/arviz-devs/PosteriorStats.jl/pull/17))
- Add v1.6 compat for all stdlib packages ([#21](https://github.com/arviz-devs/PosteriorStats.jl/pull/21))

## v0.1.3 (2023-08-21)

### Fixes

- Fix RCall set-up for CI ([#14](https://github.com/arviz-devs/PosteriorStats.jl/pull/14))
- Fix showing of pretty tables on x86 ([#16](https://github.com/arviz-devs/PosteriorStats.jl/pull/16))

## v0.1.2 (2023-08-08)

### Documentation

- Don't load PosteriorStats in doctests ([#11](https://github.com/arviz-devs/PosteriorStats.jl/pull/11))

## v0.1.1 (2023-08-08)

### Features

- Add `summarize` for computing summary statistics ([#5](https://github.com/arviz-devs/PosteriorStats.jl/pull/5))

## v0.1.0 (2023-08-05)

Initial release - history retained from [ArviZ.jl](https://github.com/arviz-devs/ArviZ.jl).
