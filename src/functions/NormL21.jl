export NormL21

struct NormL21 <: AbstractRegularizationTerm
    α::Real
end

function (f::NormL21)(x::AbstractArray{T,1}) where {T}
    y = 0
    n = Int(length(x)/2)
    for i = 1:n
        y += norm([x[i] x[i+n]])
    end
    return y
end

function cprox!(y::AbstractArray{T,1},f::NormL21,x::AbstractArray{T,1},τ::Real) where {T}
    n = Int(length(x)/2)
    α² = f.α*f.α
    for i = 1:n
        n² = x[i]^2 + x[i+n]^2
        if n²>α²
            v = f.α/√n²
            y[i] = x[i] * v
            y[i+n] = x[i+n] * v
        end
    end
    y = x
end