using ArviZExampleData
using IntervalSets
using RCall

r_loo_installed() = !isempty(rcopy(R"system.file(package='loo')"))

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
        elpd_mcse=estimates[1, 2],
        p=estimates[2, 1],
        p_mcse=estimates[2, 2],
    )
    pointwise = rcopy(R"$(result)$pointwise")
    pointwise = (
        elpd=pointwise[:, 1],
        elpd_mcse=pointwise[:, 2],
        p=pointwise[:, 3],
        reff=reff,
        pareto_shape=pointwise[:, 5],
    )
    return (; estimates, pointwise)
end

# R loo with our API
function waic_r(log_likelihood)
    R"require('loo')"
    result = R"loo::waic($log_likelihood)"
    estimates = rcopy(R"$(result)$estimates")
    estimates = (
        elpd=estimates[1, 1],
        elpd_mcse=estimates[1, 2],
        p=estimates[2, 1],
        p_mcse=estimates[2, 2],
    )
    pointwise = rcopy(R"$(result)$pointwise")
    pointwise = (elpd=pointwise[:, 1], p=pointwise[:, 2])
    return (; estimates, pointwise)
end

function log_likelihood_eight_schools(idata)
    # convert to Array to keep compile times low
    return PermutedDimsArray(collect(idata.log_likelihood.obs), (2, 3, 1))
end

function eight_schools_data()
    return (
        centered=load_example_data("centered_eight"),
        non_centered=load_example_data("non_centered_eight"),
    )
end

function _isapprox(x::AbstractArray{<:Number}, y::AbstractArray{<:Number}; kwargs...)
    return isapprox(collect(x), collect(y); kwargs...)
end
function _isapprox(x::AbstractInterval, y::AbstractInterval; kwargs...)
    return isleftclosed(x) == isleftclosed(y) &&
           isrightclosed(x) == isrightclosed(y) &&
           _isapprox(endpoints(x), endpoints(y); kwargs...)
end
function _isapprox(x::AbstractArray, y::AbstractArray; kwargs...)
    return all(map((x, y) -> _isapprox(x, y; kwargs...), x, y))
end
function _isapprox(x::Tuple, y::Tuple; kwargs...)
    length(x) == length(y) || return false
    return all(map((x, y) -> _isapprox(x, y; kwargs...), x, y))
end
function _isapprox(x::NamedTuple, y::NamedTuple; kwargs...)
    return keys(x) == keys(y) && _isapprox(values(x), values(y); kwargs...)
end
_isapprox(x, y; kwargs...) = isapprox(x, y; kwargs...)
