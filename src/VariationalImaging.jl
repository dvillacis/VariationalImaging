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

include("bilevel_learning/L2UpperLevelCost.jl")
include("bilevel_learning/NSTR.jl")

export prox!, cprox!
export TVDenoising, TikhonovDenoising
export find_optimal_parameter

# Image Denoising

function TVDenoising(img::AbstractArray{T,2},α::Real;maxit=1000,verbose=false) where {T}
    M,N = size(img)
    L = Gradient(M,N)
    f = L2DataTerm(img[:])
    g = NormL21(α)
    solver = PDHG(;maxit, verbose)
    x,res,iters = solver(f.b,L*f.b,f,g,L)
    return x
end

function TikhonovDenoising(img::AbstractArray{T,2},α::Real) where {T}
    M,N = size(img)
    L = Gradient(M,N)
    I = opEye(M*N,M*N)
    (x,stats) = cg((I+α*L'*L),img[:])
    println(stats)
    return x
end

# Parameter Learning

function find_optimal_parameter(x0::Real,img::AbstractArray{T,2},noisy::AbstractArray{T,2}) where {T}
    lower_level_solver = x -> TVDenoising(noisy,x;maxit=100)
    gradient_solver = x->x*-1
    upper_level_cost = L2UpperLevelCost(img[:])
    solver = NSTR(;verbose=true,freq=1)
    x,res,iters = solver(x0,noisy[:],upper_level_cost,gradient_solver,lower_level_solver)
    return x
end

end
