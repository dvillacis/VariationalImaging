export L2DataTerm

struct L2DataTerm{T} <: AbstractDataTerm
    b::AbstractArray{T,1}
    λ::Real
end

function L2DataTerm(b::AbstractArray{T,1},λ::Real) where {T}
    L2DataTerm{Float64}(b,λ)
end

L2DataTerm(b::AbstractArray{T,1}) where {T} = L2DataTerm(b,1.0)

function (f::L2DataTerm{T})(x::AbstractArray{T,1}) where {T}
    return (0.5*f.λ)*norm(x-f.b)^2
end

function prox!(y::AbstractArray{T,1},f::L2DataTerm{T},x::AbstractArray{T,1},τ::Real) where {T}
    τλ = τ*f.λ
    n = length(x)
    for k =1:n
        y[k] = (x[k]+τλ*f.b[k])/(1+τλ)
    end
end