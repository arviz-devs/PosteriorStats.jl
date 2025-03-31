var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Pages = [\"stats.md\"]","category":"page"},{"location":"api/#Summary-statistics","page":"API","title":"Summary statistics","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"SummaryStats\ndefault_diagnostics\ndefault_stats\ndefault_summary_stats\nsummarize","category":"page"},{"location":"api/#PosteriorStats.SummaryStats","page":"API","title":"PosteriorStats.SummaryStats","text":"A container for a column table of values computed by summarize.\n\nThis object implements the Tables and TableTraits interfaces and has a custom show method.\n\nname: The name of the collection of summary statistics, used as the table title in display.\ndata: The summary statistics for each parameter, with an optional first column parameter containing the parameter names.\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.default_diagnostics","page":"API","title":"PosteriorStats.default_diagnostics","text":"default_diagnostics(focus=Statistics.mean; kwargs...)\n\nDefault diagnostics to be computed with summarize.\n\nThe value of focus determines the diagnostics to be returned:\n\nStatistics.mean: mcse_mean, mcse_std, ess_tail, ess_bulk, rhat\nStatistics.median: mcse_median, ess_tail, ess_bulk, rhat\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.default_stats","page":"API","title":"PosteriorStats.default_stats","text":"default_stats(focus=Statistics.mean; prob_interval=0.94, kwargs...)\n\nDefault statistics to be computed with summarize.\n\nThe value of focus determines the statistics to be returned:\n\nStatistics.mean: mean, std, hdi_3%, hdi_97%\nStatistics.median: median, mad, eti_3%, eti_97%\n\nIf prob_interval is set to a different value than the default, then different HDI and ETI statistics are computed accordingly. hdi refers to the highest-density interval, while eti refers to the equal-tailed interval (i.e. the credible interval computed from symmetric quantiles).\n\nSee also: hdi\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.default_summary_stats","page":"API","title":"PosteriorStats.default_summary_stats","text":"default_summary_stats(focus=Statistics.mean; kwargs...)\n\nCombinatiton of default_stats and default_diagnostics to be used with summarize.\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.summarize","page":"API","title":"PosteriorStats.summarize","text":"summarize(data, stats_funs...; name=\"SummaryStats\", [var_names]) -> SummaryStats\n\nCompute the summary statistics in stats_funs on each param in data.\n\nstats_funs is a collection of functions that reduces a matrix with shape (draws, chains) to a scalar or a collection of scalars. Alternatively, an item in stats_funs may be a Pair of the form name => fun specifying the name to be used for the statistic or of the form (name1, ...) => fun when the function returns a collection. When the function returns a collection, the names in this latter format must be provided.\n\nIf no stats functions are provided, then those specified in default_summary_stats are computed.\n\nvar_names specifies the names of the parameters in data. If not provided, the names are inferred from data.\n\nTo support computing summary statistics from a custom object, overload this method specifying the type of data.\n\nSee also SummaryStats, default_summary_stats, default_stats, default_diagnostics.\n\nExamples\n\nCompute mean, std and the Monte Carlo standard error (MCSE) of the mean estimate:\n\njulia> using Statistics, StatsBase\n\njulia> x = randn(1000, 4, 3) .+ reshape(0:10:20, 1, 1, :);\n\njulia> summarize(x, mean, std, :mcse_mean => sem; name=\"Mean/Std\")\nMean/Std\n       mean    std  mcse_mean\n 1   0.0003  0.990      0.016\n 2  10.02    0.988      0.016\n 3  19.98    0.988      0.016\n\nAvoid recomputing the mean by using mean_and_std, and provide parameter names:\n\njulia> summarize(x, (:mean, :std) => mean_and_std, mad; var_names=[:a, :b, :c])\nSummaryStats\n         mean    std    mad\n a   0.000305  0.990  0.978\n b  10.0       0.988  0.995\n c  20.0       0.988  0.979\n\nNote that when an estimator and its MCSE are both computed, the MCSE is used to determine the number of significant digits that will be displayed.\n\njulia> summarize(x; var_names=[:a, :b, :c])\nSummaryStats\n       mean   std  hdi_3%  hdi_97%  mcse_mean  mcse_std  ess_tail  ess_bulk  r ⋯\n a   0.0003  0.99   -1.92     1.78      0.016     0.012      3567      3663  1 ⋯\n b  10.02    0.99    8.17    11.9       0.016     0.011      3841      3906  1 ⋯\n c  19.98    0.99   18.1     21.9       0.016     0.012      3892      3749  1 ⋯\n                                                                1 column omitted\n\nCompute just the statistics with an 89% HDI on all parameters, and provide the parameter names:\n\njulia> summarize(x, default_stats(; prob_interval=0.89)...; var_names=[:a, :b, :c])\nSummaryStats\n         mean    std  hdi_5.5%  hdi_94.5%\n a   0.000305  0.990     -1.63       1.52\n b  10.0       0.988      8.53      11.6\n c  20.0       0.988     18.5       21.6\n\nCompute the summary stats focusing on Statistics.median:\n\njulia> summarize(x, default_summary_stats(median)...; var_names=[:a, :b, :c])\nSummaryStats\n    median    mad  eti_3%  eti_97%  mcse_median  ess_tail  ess_median  rhat\n a   0.004  0.978   -1.83     1.89        0.020      3567        3336  1.00\n b  10.02   0.995    8.17    11.9         0.023      3841        3787  1.00\n c  19.99   0.979   18.1     21.9         0.020      3892        3829  1.00\n\n\n\n\n\n","category":"function"},{"location":"api/#General-statistics","page":"API","title":"General statistics","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"hdi\nhdi!\nr2_score","category":"page"},{"location":"api/#PosteriorStats.hdi","page":"API","title":"PosteriorStats.hdi","text":"hdi(samples::AbstractArray{<:Real}; prob=0.94) -> (; lower, upper)\n\nEstimate the unimodal highest density interval (HDI) of samples for the probability prob.\n\nThe HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest possible interval containing (100*prob)% of the probability mass.[Hyndman1996]\n\nsamples is an array of shape (draws[, chains[, params...]]). If multiple parameters are present, then lower and upper are arrays with the shape (params...,), computed separately for each marginal.\n\nThis implementation uses the algorithm of [ChenShao1999].\n\nnote: Note\nAny default value of prob is arbitrary. The default value of prob=0.94 instead of a more common default like prob=0.95 is chosen to reminder the user of this arbitrariness.\n\n[Hyndman1996]: Rob J. Hyndman (1996) Computing and Graphing Highest Density Regions,             Amer. Stat., 50(2): 120-6.             DOI: 10.1080/00031305.1996.10474359             jstor.\n\n[ChenShao1999]: Ming-Hui Chen & Qi-Man Shao (1999)              Monte Carlo Estimation of Bayesian Credible and HPD Intervals,              J Comput. Graph. Stat., 8:1, 69-92.              DOI: 10.1080/10618600.1999.10474802              jstor.\n\nExamples\n\nHere we calculate the 83% HDI for a normal random variable:\n\njulia> x = randn(2_000);\n\njulia> hdi(x; prob=0.83) |> pairs\npairs(::NamedTuple) with 2 entries:\n  :lower => -1.38266\n  :upper => 1.25982\n\nWe can also calculate the HDI for a 3-dimensional array of samples:\n\njulia> x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :);\n\njulia> hdi(x) |> pairs\npairs(::NamedTuple) with 2 entries:\n  :lower => [-1.9674, 3.0326, 8.0326]\n  :upper => [1.90028, 6.90028, 11.9003]\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.hdi!","page":"API","title":"PosteriorStats.hdi!","text":"hdi!(samples::AbstractArray{<:Real}; prob=0.94) -> (; lower, upper)\n\nA version of hdi that sorts samples in-place while computing the HDI.\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.r2_score","page":"API","title":"PosteriorStats.r2_score","text":"r2_score(y_true::AbstractVector, y_pred::AbstractVecOrMat) -> (; r2, r2_std)\n\nR² for linear Bayesian regression models.[GelmanGoodrich2019]\n\nArguments\n\ny_true: Observed data of length noutputs\ny_pred: Predicted data with size (ndraws[, nchains], noutputs)\n\n[GelmanGoodrich2019]: Andrew Gelman, Ben Goodrich, Jonah Gabry & Aki Vehtari (2019) R-squared for Bayesian Regression Models, The American Statistician, 73:3, 307-9, DOI: 10.1080/00031305.2018.1549100.\n\nExamples\n\njulia> using ArviZExampleData\n\njulia> idata = load_example_data(\"regression1d\");\n\njulia> y_true = idata.observed_data.y;\n\njulia> y_pred = PermutedDimsArray(idata.posterior_predictive.y, (:draw, :chain, :y_dim_0));\n\njulia> r2_score(y_true, y_pred) |> pairs\npairs(::NamedTuple) with 2 entries:\n  :r2     => 0.683197\n  :r2_std => 0.0368838\n\n\n\n\n\n","category":"function"},{"location":"api/#LOO-and-WAIC","page":"API","title":"LOO and WAIC","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"AbstractELPDResult\nPSISLOOResult\nWAICResult\nelpd_estimates\ninformation_criterion\nloo\nwaic","category":"page"},{"location":"api/#PosteriorStats.AbstractELPDResult","page":"API","title":"PosteriorStats.AbstractELPDResult","text":"abstract type AbstractELPDResult\n\nAn abstract type representing the result of an ELPD computation.\n\nEvery subtype stores estimates of both the expected log predictive density (elpd) and the effective number of parameters p, as well as standard errors and pointwise estimates of each, from which other relevant estimates can be computed.\n\nSubtypes implement the following functions:\n\nelpd_estimates\ninformation_criterion\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.PSISLOOResult","page":"API","title":"PosteriorStats.PSISLOOResult","text":"Results of Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO).\n\nSee also: loo, AbstractELPDResult\n\nestimates: Estimates of the expected log pointwise predictive density (ELPD) and effective number of parameters (p)\npointwise: Pointwise estimates\npsis_result: Pareto-smoothed importance sampling (PSIS) results\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.WAICResult","page":"API","title":"PosteriorStats.WAICResult","text":"Results of computing the widely applicable information criterion (WAIC).\n\nSee also: waic, AbstractELPDResult\n\nestimates: Estimates of the expected log pointwise predictive density (ELPD) and effective number of parameters (p)\npointwise: Pointwise estimates\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.elpd_estimates","page":"API","title":"PosteriorStats.elpd_estimates","text":"elpd_estimates(result::AbstractELPDResult; pointwise=false) -> (; elpd, elpd_mcse, lpd)\n\nReturn the (E)LPD estimates from the result.\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.information_criterion","page":"API","title":"PosteriorStats.information_criterion","text":"information_criterion(elpd, scale::Symbol)\n\nCompute the information criterion for the given scale from the elpd estimate.\n\nscale must be one of (:deviance, :log, :negative_log).\n\nSee also: loo, waic\n\n\n\n\n\ninformation_criterion(result::AbstractELPDResult, scale::Symbol; pointwise=false)\n\nCompute information criterion for the given scale from the existing ELPD result.\n\nscale must be one of (:deviance, :log, :negative_log).\n\nIf pointwise=true, then pointwise estimates are returned.\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.loo","page":"API","title":"PosteriorStats.loo","text":"loo(log_likelihood; reff=nothing, kwargs...) -> PSISLOOResult{<:NamedTuple,<:NamedTuple}\n\nCompute the Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO). [Vehtari2017][LOOFAQ]\n\nlog_likelihood must be an array of log-likelihood values with shape (chains, draws[, params...]).\n\nKeywords\n\nreff::Union{Real,AbstractArray{<:Real}}: The relative effective sample size(s) of the likelihood values. If an array, it must have the same data dimensions as the corresponding log-likelihood variable. If not provided, then this is estimated using MCMCDiagnosticTools.ess.\nkwargs: Remaining keywords are forwarded to [PSIS.psis].\n\nSee also: PSISLOOResult, waic\n\n[Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017). doi: 10.1007/s11222-016-9696-4 arXiv: 1507.04544\n\n[LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html\n\nExamples\n\nManually compute R_mathrmeff and calculate PSIS-LOO of a model:\n\njulia> using ArviZExampleData, MCMCDiagnosticTools\n\njulia> idata = load_example_data(\"centered_eight\");\n\njulia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));\n\njulia> reff = ess(log_like; kind=:basic, split_chains=1, relative=true);\n\njulia> loo(log_like; reff)\nPSISLOOResult with estimates\n elpd  elpd_mcse    p  p_mcse\n  -31        1.4  0.9    0.34\n\nand PSISResult with 500 draws, 4 chains, and 8 parameters\nPareto shape (k) diagnostic values:\n                    Count      Min. ESS\n (-Inf, 0.5]  good  7 (87.5%)  151\n  (0.5, 0.7]  okay  1 (12.5%)  446\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.waic","page":"API","title":"PosteriorStats.waic","text":"waic(log_likelihood::AbstractArray) -> WAICResult{<:NamedTuple,<:NamedTuple}\n\nCompute the widely applicable information criterion (WAIC).[Watanabe2010][Vehtari2017][LOOFAQ]\n\nlog_likelihood must be an array of log-likelihood values with shape (chains, draws[, params...]).\n\nSee also: WAICResult, loo\n\n[Watanabe2010]: Watanabe, S. Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory. 11(116):3571−3594, 2010. https://jmlr.csail.mit.edu/papers/v11/watanabe10a.html\n\n[Vehtari2017]: Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017). doi: 10.1007/s11222-016-9696-4 arXiv: 1507.04544\n\n[LOOFAQ]: Aki Vehtari. Cross-validation FAQ. https://mc-stan.org/loo/articles/online-only/faq.html\n\nExamples\n\nCalculate WAIC of a model:\n\njulia> using ArviZExampleData\n\njulia> idata = load_example_data(\"centered_eight\");\n\njulia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));\n\njulia> waic(log_like)\nWAICResult with estimates\n elpd  elpd_mcse    p  p_mcse\n  -31        1.4  0.9    0.33\n\n\n\n\n\n","category":"function"},{"location":"api/#Model-comparison","page":"API","title":"Model comparison","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"ModelComparisonResult\ncompare\nmodel_weights","category":"page"},{"location":"api/#PosteriorStats.ModelComparisonResult","page":"API","title":"PosteriorStats.ModelComparisonResult","text":"ModelComparisonResult\n\nResult of model comparison using ELPD.\n\nThis struct implements the Tables and TableTraits interfaces.\n\nEach field returns a collection of the corresponding entry for each model:\n\nname: Names of the models, if provided.\nrank: Ranks of the models (ordered by decreasing ELPD)\nelpd_diff: ELPD of a model subtracted from the largest ELPD of any model\nelpd_diff_mcse: Monte Carlo standard error of the ELPD difference\nweight: Model weights computed with weights_method\nelpd_result: AbstactELPDResults for each model, which can be used to access useful stats like ELPD estimates, pointwise estimates, and Pareto shape values for PSIS-LOO\nweights_method: Method used to compute model weights with model_weights\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.compare","page":"API","title":"PosteriorStats.compare","text":"compare(models; kwargs...) -> ModelComparisonResult\n\nCompare models based on their expected log pointwise predictive density (ELPD).\n\nThe ELPD is estimated either by Pareto smoothed importance sampling leave-one-out cross-validation (LOO) or using the widely applicable information criterion (WAIC). We recommend loo. Read more theory here - in a paper by some of the leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353\n\nArguments\n\nmodels: a Tuple, NamedTuple, or AbstractVector whose values are either AbstractELPDResult entries or any argument to elpd_method.\n\nKeywords\n\nweights_method::AbstractModelWeightsMethod=Stacking(): the method to be used to weight the models. See model_weights for details\nelpd_method=loo: a method that computes an AbstractELPDResult from an argument in models.\nsort::Bool=true: Whether to sort models by decreasing ELPD.\n\nReturns\n\nModelComparisonResult: A container for the model comparison results. The fields contain a similar collection to models.\n\nExamples\n\nCompare the centered and non centered models of the eight school problem using the defaults: loo and Stacking weights. A custom myloo method formates the inputs as expected by loo.\n\njulia> using ArviZExampleData\n\njulia> models = (\n           centered=load_example_data(\"centered_eight\"),\n           non_centered=load_example_data(\"non_centered_eight\"),\n       );\n\njulia> function myloo(idata)\n           log_like = PermutedDimsArray(idata.log_likelihood.obs, (2, 3, 1))\n           return loo(log_like)\n       end;\n\njulia> mc = compare(models; elpd_method=myloo)\n┌ Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.\n└ @ PSIS ~/.julia/packages/PSIS/...\nModelComparisonResult with Stacking weights\n               rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p   ⋯\n non_centered     1   -31        1.4       0              0.0       1.0  0.9   ⋯\n centered         2   -31        1.4       0.06           0.067     0.0  0.9   ⋯\n                                                                1 column omitted\njulia> mc.weight |> pairs\npairs(::NamedTuple) with 2 entries:\n  :non_centered => 1.0\n  :centered     => 5.34175e-19\n\nCompare the same models from pre-computed PSIS-LOO results and computing BootstrappedPseudoBMA weights:\n\njulia> elpd_results = mc.elpd_result;\n\njulia> compare(elpd_results; weights_method=BootstrappedPseudoBMA())\nModelComparisonResult with BootstrappedPseudoBMA weights\n               rank  elpd  elpd_mcse  elpd_diff  elpd_diff_mcse  weight    p   ⋯\n non_centered     1   -31        1.4       0              0.0      0.52  0.9   ⋯\n centered         2   -31        1.4       0.06           0.067    0.48  0.9   ⋯\n                                                                1 column omitted\n\n\n\n\n\n","category":"function"},{"location":"api/#PosteriorStats.model_weights","page":"API","title":"PosteriorStats.model_weights","text":"model_weights(elpd_results; method=Stacking())\nmodel_weights(method::AbstractModelWeightsMethod, elpd_results)\n\nCompute weights for each model in elpd_results using method.\n\nelpd_results is a Tuple, NamedTuple, or AbstractVector with AbstractELPDResult entries. The weights are returned in the same type of collection.\n\nStacking is the recommended approach, as it performs well even when the true data generating process is not included among the candidate models. See [YaoVehtari2018] for details.\n\nSee also: AbstractModelWeightsMethod, compare\n\n[YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.                Using Stacking to Average Bayesian Predictive Distributions.                2018. Bayesian Analysis. 13, 3, 917–1007.                doi: 10.1214/17-BA1091                arXiv: 1704.02030\n\nExamples\n\nCompute Stacking weights for two models:\n\njulia> using ArviZExampleData\n\njulia> models = (\n           centered=load_example_data(\"centered_eight\"),\n           non_centered=load_example_data(\"non_centered_eight\"),\n       );\n\njulia> elpd_results = map(models) do idata\n           log_like = PermutedDimsArray(idata.log_likelihood.obs, (2, 3, 1))\n           return loo(log_like)\n       end;\n┌ Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.\n└ @ PSIS ~/.julia/packages/PSIS/...\n\njulia> model_weights(elpd_results; method=Stacking()) |> pairs\npairs(::NamedTuple) with 2 entries:\n  :centered     => 5.34175e-19\n  :non_centered => 1.0\n\nNow we compute BootstrappedPseudoBMA weights for the same models:\n\njulia> model_weights(elpd_results; method=BootstrappedPseudoBMA()) |> pairs\npairs(::NamedTuple) with 2 entries:\n  :centered     => 0.483723\n  :non_centered => 0.516277\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"The following model weighting methods are available","category":"page"},{"location":"api/","page":"API","title":"API","text":"AbstractModelWeightsMethod\nBootstrappedPseudoBMA\nPseudoBMA\nStacking","category":"page"},{"location":"api/#PosteriorStats.AbstractModelWeightsMethod","page":"API","title":"PosteriorStats.AbstractModelWeightsMethod","text":"abstract type AbstractModelWeightsMethod\n\nAn abstract type representing methods for computing model weights.\n\nSubtypes implement model_weights(method, elpd_results).\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.BootstrappedPseudoBMA","page":"API","title":"PosteriorStats.BootstrappedPseudoBMA","text":"struct BootstrappedPseudoBMA{R<:Random.AbstractRNG, T<:Real} <: AbstractModelWeightsMethod\n\nModel weighting method using pseudo Bayesian Model Averaging using Akaike-type weighting with the Bayesian bootstrap (pseudo-BMA+)[YaoVehtari2018].\n\nThe Bayesian bootstrap stabilizes the model weights.\n\nBootstrappedPseudoBMA(; rng=Random.default_rng(), samples=1_000, alpha=1)\nBootstrappedPseudoBMA(rng, samples, alpha)\n\nConstruct the method.\n\nrng::Random.AbstractRNG: The random number generator to use for the Bayesian bootstrap\nsamples::Int64: The number of samples to draw for bootstrapping\nalpha::Real: The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap. The default (1) corresponds to a uniform distribution on the simplex.\n\nSee also: Stacking\n\n[YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.                Using Stacking to Average Bayesian Predictive Distributions.                2018. Bayesian Analysis. 13, 3, 917–1007.                doi: 10.1214/17-BA1091                arXiv: 1704.02030\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.PseudoBMA","page":"API","title":"PosteriorStats.PseudoBMA","text":"struct PseudoBMA <: AbstractModelWeightsMethod\n\nModel weighting method using pseudo Bayesian Model Averaging (pseudo-BMA) and Akaike-type weighting.\n\nPseudoBMA(; regularize=false)\nPseudoBMA(regularize)\n\nConstruct the method with optional regularization of the weights using the standard error of the ELPD estimate.\n\nnote: Note\nThis approach is not recommended, as it produces unstable weight estimates. It is recommended to instead use BootstrappedPseudoBMA to stabilize the weights or Stacking. For details, see [YaoVehtari2018].\n\n[YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.                Using Stacking to Average Bayesian Predictive Distributions.                2018. Bayesian Analysis. 13, 3, 917–1007.                doi: 10.1214/17-BA1091                arXiv: 1704.02030\n\nSee also: Stacking\n\n\n\n\n\n","category":"type"},{"location":"api/#PosteriorStats.Stacking","page":"API","title":"PosteriorStats.Stacking","text":"struct Stacking{O<:Optim.AbstractOptimizer} <: AbstractModelWeightsMethod\n\nModel weighting using stacking of predictive distributions[YaoVehtari2018].\n\nStacking(; optimizer=Optim.LBFGS(), options=Optim.Options()\nStacking(optimizer[, options])\n\nConstruct the method, optionally customizing the optimization.\n\noptimizer::Optim.AbstractOptimizer: The optimizer to use for the optimization of the weights. The optimizer must support projected gradient optimization via a manifold field.\noptions::Optim.Options: The Optim options to use for the optimization of the weights.\n\nSee also: BootstrappedPseudoBMA\n\n[YaoVehtari2018]: Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman.                Using Stacking to Average Bayesian Predictive Distributions.                2018. Bayesian Analysis. 13, 3, 917–1007.                doi: 10.1214/17-BA1091                arXiv: 1704.02030\n\n\n\n\n\n","category":"type"},{"location":"api/#Predictive-checks","page":"API","title":"Predictive checks","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"loo_pit","category":"page"},{"location":"api/#PosteriorStats.loo_pit","page":"API","title":"PosteriorStats.loo_pit","text":"loo_pit(y, y_pred, log_weights; kwargs...) -> Union{Real,AbstractArray}\n\nCompute leave-one-out probability integral transform (LOO-PIT) checks.\n\nArguments\n\ny: array of observations with shape (params...,)\ny_pred: array of posterior predictive samples with shape (draws, chains, params...).\nlog_weights: array of normalized log LOO importance weights with shape (draws, chains, params...).\n\nKeywords\n\nis_discrete: If not provided, then it is set to true iff elements of y and y_pred are all integer-valued. If true, then data are smoothed using smooth_data to make them non-discrete before estimating LOO-PIT values.\nkwargs: Remaining keywords are forwarded to smooth_data if data is discrete.\n\nReturns\n\npitvals: LOO-PIT values with same size as y. If y is a scalar, then pitvals is a scalar.\n\nLOO-PIT is a marginal posterior predictive check. If y_-i is the array y of observations with the ith observation left out, and y_i^* is a posterior prediction of the ith observation, then the LOO-PIT value for the ith observation is defined as\n\nP(y_i^* le y_i mid y_-i) = int_-infty^y_i p(y_i^* mid y_-i) mathrmd y_i^*\n\nThe LOO posterior predictions and the corresponding observations should have similar distributions, so if conditional predictive distributions are well-calibrated, then all LOO-PIT values should be approximately uniformly distributed on 0 1.[Gabry2019]\n\n[Gabry2019]: Gabry, J., Simpson, D., Vehtari, A., Betancourt, M. & Gelman, A. Visualization in Bayesian Workflow. J. R. Stat. Soc. Ser. A Stat. Soc. 182, 389–402 (2019). doi: 10.1111/rssa.12378 arXiv: 1709.01449\n\nExamples\n\nCalculate LOO-PIT values using as test quantity the observed values themselves.\n\njulia> using ArviZExampleData\n\njulia> idata = load_example_data(\"centered_eight\");\n\njulia> y = idata.observed_data.obs;\n\njulia> y_pred = PermutedDimsArray(idata.posterior_predictive.obs, (:draw, :chain, :school));\n\njulia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));\n\njulia> log_weights = loo(log_like).psis_result.log_weights;\n\njulia> loo_pit(y, y_pred, log_weights)\n8-element DimArray{Float64,1} with dimensions:\n  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered\n \"Choate\"            0.943511\n \"Deerfield\"         0.63797\n \"Phillips Andover\"  0.316697\n \"Phillips Exeter\"   0.582252\n \"Hotchkiss\"         0.295321\n \"Lawrenceville\"     0.403318\n \"St. Paul's\"        0.902508\n \"Mt. Hermon\"        0.655275\n\nCalculate LOO-PIT values using as test quantity the square of the difference between each observation and mu.\n\njulia> using Statistics\n\njulia> mu = idata.posterior.mu;\n\njulia> T = y .- median(mu);\n\njulia> T_pred = y_pred .- mu;\n\njulia> loo_pit(T .^ 2, T_pred .^ 2, log_weights)\n8-element DimArray{Float64,1} with dimensions:\n  Dim{:school} Categorical{String} String[Choate, Deerfield, …, St. Paul's, Mt. Hermon] Unordered\n \"Choate\"            0.873577\n \"Deerfield\"         0.243686\n \"Phillips Andover\"  0.357563\n \"Phillips Exeter\"   0.149908\n \"Hotchkiss\"         0.435094\n \"Lawrenceville\"     0.220627\n \"St. Paul's\"        0.775086\n \"Mt. Hermon\"        0.296706\n\n\n\n\n\n","category":"function"},{"location":"api/#Utilities","page":"API","title":"Utilities","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"PosteriorStats.smooth_data","category":"page"},{"location":"api/#PosteriorStats.smooth_data","page":"API","title":"PosteriorStats.smooth_data","text":"smooth_data(y; dims=:, interp_method=CubicSpline, offset_frac=0.01)\n\nSmooth y along dims using interp_method.\n\ninterp_method is a 2-argument callabale that takes the arguments y and x and returns a DataInterpolations.jl interpolation method, defaulting to a cubic spline interpolator.\n\noffset_frac is the fraction of the length of y to use as an offset when interpolating.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = PosteriorStats","category":"page"},{"location":"#PosteriorStats","page":"Home","title":"PosteriorStats","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"PosteriorStats implements widely-used and well-characterized statistical analyses for the Bayesian workflow. These functions generally estimate properties of posterior and/or posterior predictive distributions. The default implementations defined here operate on Monte Carlo samples.","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the API for details.","category":"page"},{"location":"#Extending-this-package","page":"Home","title":"Extending this package","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The methods defined here are intended to be extended by two types of packages.","category":"page"},{"location":"","page":"Home","title":"Home","text":"packages that implement data types for storing Monte Carlo samples\npackages that implement other representations for posterior distributions than Monte Carlo draws","category":"page"}]
}
