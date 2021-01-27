########################################################
# Image denoising with generic regularizer via primal–dual proximal splitting
########################################################

__precompile__()

module RegDenoise

using AlgTools.Util
using AlgTools.LinOps
import AlgTools.Iterate
using VariationalImaging.GradientOps

export gen_denoise_pdps

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

function gen_denoise_pdps(b :: Image, op :: LinOp{Image,Data};
    xinit :: Union{Image,Nothing} = nothing,
    iterate = Iterate.simple_iterate,
    params::NamedTuple) where Data

    ################################                                        
    # Extract and set up parameters
    ################################                    

    α, ρ = params.α, params.ρ
    τ₀, σ₀ =  params.τ₀, params.σ₀

    R_K = opnorm_estimate(op)
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


        inplace!(Δx,op',y)             # primal step:
        @. x̄ = x                       # |  save old x for over-relax
        @. x = (x-τ*(Δx-b))/(1+τ)      # |  prox
        @. x̄ = (1+ω)*x - ω*x̄           # over-relax: x̄ = 2x-x_old
        inplace!(Δy,op,x̄)              # dual step: y
        @. y = (y + σ*Δy)/(1 + σ*ρ/α)  # |
        proj_norm₂₁ball!(y, α)         # |  prox

        if params.accel
            τ, σ = τ*ω, σ/ω
        end

        ################################
        # Give function value if needed
        ################################
        v = verbose() do     
            inplace!(Δy,op,x)
            value = norm₂²(b-x)/2 + params.α*γnorm₂₁(Δy, params.ρ)
            value, x
        end

        v
    end

    return x, y, v
end

end # Module