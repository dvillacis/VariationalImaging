module VariationalImaging

using LinearAlgebra
using Printf

using ColorTypes: Gray
import ColorVectorSpace

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise

import Base.show

abstract type AbstractDataTerm end
abstract type AbstractRegularizationTerm end

# Write your package code here.
include("operators/PatchOperator.jl")

#include("utilities/IterationTools.jl")
#include("utilities/BatchMul.jl")
#include("utilities/Experiment.jl")

include("GradientOps.jl")
include("RegDenoise.jl")
include("SDDenoise.jl")
include("Util.jl")

include("bilevel_learning/MatrixUtils.jl")
#include("bilevel_learning/NSTR.jl")

export TVl₂Denoising, TikhonovDenoising
export find_optimal_parameter

# Image Denoising

function TVl₂Denoising(img::AbstractArray{T,2},α::Real;maxit=1000,verbose=false,freq=10) where {T}

    img = Float64.(Gray{Float64}.(img))

    if verbose == false
        verbose_iter = maxit+1
    else
        verbose_iter = freq
    end

    params = (
        α = α,
        τ₀ = 5,
        σ₀ = 0.99/5,
        ρ = 0,
        accel = true,
        verbose_iter = verbose_iter,
        maxiter = maxit,
        save_iterations = false
    )

    st, iterate = initialise_visualisation(false)
    x, y, st = denoise_pdps(img; iterate=iterate, params=params)
    finalise_visualisation(st)

    return x
end

# Dataset Denoising

function TVl₂Denoising(img::AbstractArray{T,3},α::Real;maxit=1000,verbose=false,freq=10) where {T}

    img = Float64.(Gray{Float64}.(img))

    M,N,O = size(img)

    if verbose == false
        verbose_iter = maxit+1
    else
        verbose_iter = freq
    end

    params = (
        α = α,
        τ₀ = 5,
        σ₀ = 0.99/5,
        ρ = 0,
        accel = true,
        verbose_iter = verbose_iter,
        maxiter = maxit,
        save_iterations = false
    )

    x = zeros(size(img))
    
    for i=1:O
        #print(".")
        st, iterate = initialise_visualisation(false)
        x_, y_, st_ = denoise_pdps(img[:,:,i]; iterate=iterate, params=params)
        x[:,:,i] = x_
        finalise_visualisation(st)
    end
    #print("*")
    return x
end

function TVl₂Denoising(img::AbstractArray{T,2},α::AbstractArray{S,2};maxit=1000,verbose=false,freq=10,visualize=false) where {T,S}

    img = Float64.(Gray{Float64}.(img))

    if verbose == false
        verbose_iter = maxit+1
    else
        verbose_iter = freq
    end

    params = (
        α = α,
        τ₀ = 5,
        σ₀ = 0.99/5,
        ρ = 0,
        accel = true,
        verbose_iter = verbose_iter,
        maxiter = maxit,
        save_iterations = false
    )

    st, iterate = initialise_visualisation(visualize)
    x, y, st = denoise_sd_pdps(img; iterate=iterate, params=params)
    finalise_visualisation(st)

    return x
end

function TVl₂Denoising(img::AbstractArray{T,3},α::AbstractArray{S,2};maxit=1000,verbose=false,freq=10,visualize=false) where {T,S}

    img = Float64.(Gray{Float64}.(img))

    M,N,O = size(img)
    m,n = size(α)
    if M > m || N > n
        p = PatchOperator(α,img[:,:,1])
        α = patch(p,α)
    end

    if verbose == false
        verbose_iter = maxit+1
    else
        verbose_iter = freq
    end

    params = (
        α = α,
        τ₀ = 5,
        σ₀ = 0.99/5,
        ρ = 0,
        accel = true,
        verbose_iter = verbose_iter,
        maxiter = maxit,
        save_iterations = false
    )

    x = zeros(size(img))
    
    for i=1:O
        st, iterate = initialise_visualisation(visualize)
        x_, y_, st = denoise_sd_pdps(img[:,:,i]; iterate=iterate, params=params)
        x[:,:,i] = x_
        finalise_visualisation(st)
    end

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
