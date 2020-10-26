module VariationalImaging

using LinearOperators
using Krylov
using LinearAlgebra
using Printf

import Base.show

abstract type AbstractDataTerm end
abstract type AbstractRegularizationTerm end

# Write your package code here.
include("operators/Gradient.jl")

include("functions/L2DataTerm.jl")
include("functions/NormL21.jl")

include("utilities/IterationTools.jl")

include("solvers/PDHG.jl")

export prox!, cprox!
export TVDenoising, TikhonovDenoising

function TVDenoising(img::AbstractArray{T,2},α::Real) where {T}
    M,N = size(img)
    L = Gradient(M,N)
    f = L2DataTerm(img[:])
    g = NormL21(α)
    solver = PDHG(;maxit=1000, verbose=true)
    x,res,iters = solver(f.b,L*f.b,f,g,L)
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
