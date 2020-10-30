export NormL21

struct NormL21{T} <: AbstractRegularizationTerm
    α::Union{Real,AbstractArray{T,1}}
end

function NormL21(α::AbstractArray{T,2}) where {T}
    return NormL21(α[:])
end

function (f::NormL21)(x::AbstractArray{T,1}) where {T}
    y = 0
    n = Int(length(x)/2)
    @assert n == length(f.α)
    for i = 1:n
        y += f.α[i]*norm([x[i] x[i+n]])
    end
    return y
end

function cprox!(y::AbstractArray{T,1},f::NormL21,x::AbstractArray{T,1},τ::Real) where {T}
    n = Int(length(x)/2)
    @assert n == length(f.α)
    for i = 1:n
        α² = f.α[i]*f.α[i]
        n² = x[i]^2 + x[i+n]^2
        if n²>α²
            v = f.α[i]/√n²
            y[i] = x[i] * v
            y[i+n] = x[i+n] * v
        end
    end
    y = x
end