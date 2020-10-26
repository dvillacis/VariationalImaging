export L2UpperLevelCost

struct L2UpperLevelCost{T} <: Function
    target::AbstractArray{T,1}
    λ::Real
end

function L2UpperLevelCost(target::AbstractArray{T,1},λ::Real) where {T}
    L2UpperLevelCost{Float64}(target,λ)
end

L2UpperLevelCost(target::AbstractArray{T,1}) where {T} = L2UpperLevelCost(target,1.0)

function (f::L2UpperLevelCost{T})(x::AbstractArray{T,1}) where {T}
    return (0.5*f.λ)*norm(x-f.target)^2
end