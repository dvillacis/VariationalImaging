export SqrNormL2

struct SqrNormL2{T} <: AbstractImagingFunction
    b::AbstractArray{T,1}
    λ::Real
end

function SqrNormL2(b::AbstractArray{T,1},λ::Real) where {T}
    SqrNormL2{Float64}(b,λ)
end

SqrNormL2(b::AbstractArray{T,1}) where {T} = SqrNormL2(b,1.0)

function (f::SqrNormL2{T})(x::AbstractArray{T,1}) where {T}
    return (0.5*f.λ)*norm(x-f.b)^2
end

function prox!(y::AbstractArray{T,1},f::SqrNormL2{T},x::AbstractArray{T,1},τ::Real) where {T}
    τλ = τ*f.λ
    n = length(x)
    for k =1:n
        y[k] = (x[k]+τλ*f.b[k])/(1+τλ)
    end
end