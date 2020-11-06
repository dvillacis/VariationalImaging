########################################################
# Basic TV denoising via primal–dual proximal splitting
########################################################

__precompile__()

using AlgTools.Util
import AlgTools.Iterate
using ImageTools.Gradient

##############
# Our exports
##############

export denoise_sd_pdps,
       denoise_sd_fista

#############
# Data types
#############

ImageSize = Tuple{Integer,Integer}
Image = Array{Float64,2}
Primal = Image
Dual = Array{Float64,3}

#########################
# Iterate initialisation
#########################

function init_rest(x::Primal)
    imdim=size(x)

    y = zeros(2, imdim...)
    Δx = copy(x)
    Δy = copy(y)
    x̄ = copy(x)

    return x, y, Δx, Δy, x̄
end

function init_primal(xinit::Image, b)
    return copy(xinit)
end

function init_primal(xinit::Nothing, b :: Image)
    return zeros(size(b)...)
end


############
# Algorithm
############

function denoise_sd_pdps(b :: Image;
                      xinit :: Union{Image,Nothing} = nothing,
                      iterate = AlgTools.simple_iterate,
                      params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################                    

    α, ρ = params.α, params.ρ
    τ₀, σ₀ =  params.τ₀, params.σ₀

    R_K = ∇₂_norm₂₂_est
    γ = 1

    @assert(τ₀*σ₀ < 1)
    σ = σ₀/R_K
    τ = τ₀/R_K
    
    ######################
    # Initialise iterates
    ######################

    x, y, Δx, Δy, x̄ = init_rest(init_primal(xinit, b))

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        ω = params.accel ? 1/√(1+2*γ*τ) : 1
        
        ∇₂ᵀ!(Δx, y)                    # primal step:
        @. x̄ = x                       # |  save old x for over-relax
        @. x = (x-τ*(Δx-b))/(1+τ)      # |  prox
        @. x̄ = (1+ω)*x - ω*x̄           # over-relax: x̄ = 2x-x_old
        ∇₂!(Δy, x̄)                     # dual step: y
        @. y = (y + σ*Δy)               # |
        @. y[1,:,:] = y[1,:,:] ./(1 + σ*ρ ./α)
        @. y[2,:,:] = y[2,:,:] ./(1 + σ*ρ ./α)
        proj_norm₂₁ball!(y, α)         # |  prox
        

        if params.accel
            τ, σ = τ*ω, σ/ω
        end
                
        ################################
        # Give function value if needed
        ################################
        v = verbose() do    
            ∇₂!(Δy, x)
            value = norm₂²(b-x)/2 + norm₂₁w(Δy, params.α)
            value, x  
        end
        v
    end

    return x, y, v
end

function denoise_sd_fista(b :: Image;
                       xinit :: Union{Image,Nothing} = nothing,
                       iterate = AlgTools.simple_iterate,
                       params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################                    

    α, ρ = params.α, params.ρ
    τ₀ =  params.τ₀
    τ = τ₀/∇₂_norm₂₂_est²
    
    ######################
    # Initialise iterates
    ######################

    x = init_primal(xinit, b)
    imdim = size(x)
    Δx = similar(x)
    y = zeros(2, imdim...)
    ỹ = copy(y)
    y⁻ = similar(y)
    Δy = similar(y)

    ####################
    # Run the algorithm
    ####################

    t = 0

    v = iterate(params) do verbose :: Function                    
        ∇₂ᵀ!(Δx, ỹ)
        @. Δx .-= b
        ∇₂!(Δy, Δx)
        @. y⁻ = y
        @. y = (ỹ - τ*Δy)./(1 + τ*ρ ./α)
        proj_norm₂₁ball!(y, α)
        t⁺ = (1+√(1+4*t^2))/2
        @. ỹ = y+((t-1)/t⁺)*(y-y⁻)
        t = t⁺

        ################################
        # Give function value if needed
        ################################
        v = verbose() do
            ∇₂ᵀ!(Δx, y)
            @. x = b - Δx
            ∇₂!(Δy, x)
            value = norm₂²(b-x)/2 + params.α*γnorm₂₁(Δy, params.ρ)
            value, x
        end

        v
    end

    ∇₂ᵀ!(Δx, y)
    @. x = b - Δx

    return x, y, v
end


