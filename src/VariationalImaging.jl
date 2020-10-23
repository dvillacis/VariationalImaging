module VariationalImaging

using LinearOperators
using Krylov
using LinearAlgebra
using Printf

import Base.show

abstract type AbstractImagingFunction end

# Write your package code here.
include("operators/Gradient.jl")
include("functions/SqrNormL2.jl")
include("functions/NormL21.jl")
include("solvers/pdhg.jl")

export prox!, cprox!
export TVDenoising, TikhonovDenoising

function TVDenoising(img::AbstractArray{T,2},α::Real) where {T}
    M,N = size(img)
    L = Gradient(M,N)
    f = SqrNormL2(img[:])
    g = NormL21(α)
    x = pdhg(f,g,L;x0=f.b,y0=L*f.b)
    return reshape(x,M,N)
end

function TikhonovDenoising(img::AbstractArray{T,2},α::Real) where {T}
    M,N = size(img)
    L = Gradient(M,N)
    I = opEye(M*N,M*N)
    (x,stats) = cg((I+α*L'*L),img[:])
    println(stats)
    return reshape(x,M,N)
end

end
