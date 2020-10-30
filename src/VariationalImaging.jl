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

function TVDenoising(img::AbstractArray{T,2},α::AbstractArray{T,2};maxit=1000,verbose=false) where {T}
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

function rof_function(x::Real,noisy::AbstractArray{T,2},img::AbstractArray{T,2}) where {T}
    u = TVDenoising(noisy,x)
    cost_fun = L2UpperLevelCost(img[:])
    return (u=u,c=cost_fun(u),g=x)
end

function find_optimal_parameter(x0::R,img::AbstractArray{T,2},noisy::AbstractArray{T,2}) where {R,T}
    #f = x -> (u=x,c=(x+1)^2*exp(x),g=2*(x+1)*exp(x)+(x+1)^2*exp(x))
    #f = x -> (u=x,c=(x+1)^2,g=2*(x+1))
    #f = x -> (u=x,c=(1-x[1])^2+100*(x[2]-x[1]^2)^2,g=[-2*(1-x[1])-200*x[1]*(x[2]-x[1]^2);200*(x[2]-x[1]^2)])
    #f = x -> rof_function(x,noisy,img)
    f = x -> (u=x,c=abs(x),g=sign(x))
    solver = NSTR(;verbose=true,freq=5,maxit=100)
    x,res,iters = solver(f,x0)
    return x
end

end
