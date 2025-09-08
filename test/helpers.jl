using ArviZExampleData
using IntervalSets

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
