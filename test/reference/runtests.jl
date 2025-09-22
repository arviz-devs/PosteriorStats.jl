using Distributions
using PosteriorStats
using Random
using RCall
using Test

# R loo with our API
function loo_r(log_likelihood; reff=nothing)
    R"require('loo')"
    if reff === nothing
        reff = rcopy(R"loo::relative_eff(exp($(log_likelihood)))")
    end
    result = R"loo::loo($log_likelihood, r_eff=$reff)"
    estimates = rcopy(R"$(result)$estimates")
    estimates = (
        elpd=estimates[1, 1],
        se_elpd=estimates[1, 2],
        p=estimates[2, 1],
        se_p=estimates[2, 2],
    )
    pointwise = rcopy(R"$(result)$pointwise")
    pointwise = (
        elpd=pointwise[:, 1],
        se_elpd=pointwise[:, 2],
        p=pointwise[:, 3],
        reff=reff,
        pareto_shape=pointwise[:, 5],
    )
    return (; estimates, pointwise)
end

function generate_log_likelihoods(proposal, target, ndraws, nchains, nparams)
    draws = rand(proposal, ndraws, nchains, nparams)
    log_likelihood = loglikelihood.(target, draws)
    return log_likelihood
end

Random.seed!(24)

@testset "Consistency with R loo" begin
    proposal = Normal()
    target = TDist(7)

    @testset "loo" begin
        log_likelihood = generate_log_likelihoods(proposal, target, 1000, 4, 10)
        reff_rand = rand(size(log_likelihood, 3))
        @testset for reff in (nothing, reff_rand)
            result_r = loo_r(log_likelihood; reff)
            result = loo(log_likelihood; reff)
            @test result.estimates.elpd ≈ result_r.estimates.elpd
            @test result.estimates.se_elpd ≈ result_r.estimates.se_elpd
            @test result.estimates.p ≈ result_r.estimates.p
            @test result.estimates.se_p ≈ result_r.estimates.se_p
            @test result.pointwise.elpd ≈ result_r.pointwise.elpd
            # increased tolerance for se_elpd, since we use a different approach
            @test result.pointwise.se_elpd ≈ result_r.pointwise.se_elpd rtol = 0.01
            @test result.pointwise.p ≈ result_r.pointwise.p
            @test result.pointwise.reff ≈ result_r.pointwise.reff
            @test result.pointwise.pareto_shape ≈ result_r.pointwise.pareto_shape
        end
    end
end
