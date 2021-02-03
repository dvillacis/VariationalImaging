########################################################
# Image denoising with generic regularizer summatory via primal–dual proximal splitting
########################################################

__precompile__()

module SumRegsDenoise

using AlgTools.Util
using AlgTools.LinOps
import AlgTools.Iterate
using VariationalImaging.GradientOps

export sumregs_denoise_pdps

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

    y₁ = zeros(2, imdim...)
    y₂ = zeros(2, imdim...)
    y₃ = zeros(2, imdim...)
    Δx₁ = copy(x)
    Δx₂ = copy(x)
    Δx₃ = copy(x)
    Δy₁ = copy(y₁)
    Δy₂ = copy(y₂)
    Δy₃ = copy(y₃)
    x̄ = copy(x)

    return x, y₁, y₂, y₃, Δx₁, Δx₂, Δx₃, Δy₁, Δy₂, Δy₃, x̄
end

function init_primal(xinit::Image, b)
    return copy(xinit)
end

function init_primal(xinit::Nothing, b :: Image)
    return zeros(size(b)...)
end


############
# Algorithm Paper
############

function sumregs_denoise_pdps(b :: Image, 
    op₁ :: LinOp{Image,Data}, 
    op₂ :: LinOp{Image,Data}, 
    op₃ :: LinOp{Image,Data};
    xinit :: Union{Image,Nothing} = nothing,
    iterate = Iterate.simple_iterate,
    params::NamedTuple) where Data

    ################################                                        
    # Extract and set up parameters
    ################################                    

    α₁, α₂, α₃, ρ = params.α₁, params.α₂, params.α₃, params.ρ
    τ₀, σ₀ =  params.τ₀, params.σ₀

    R_K = opnorm_estimate(op₁) + opnorm_estimate(op₂) + opnorm_estimate(op₃)
    γ = 1

    @assert(τ₀*σ₀ < 1)
    σ = σ₀/R_K
    τ = τ₀/R_K

    ######################
    # Initialise iterates
    ######################

    x, y₁, y₂, y₃, Δx₁, Δx₂, Δx₃, Δy₁, Δy₂, Δy₃, x̄ = init_rest(init_primal(xinit, b))

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        ω = params.accel ? 1/√(1+2*γ*τ) : 1

        inplace!(Δx₁,op₁',y₁)                   # primal step:
        inplace!(Δx₂,op₂',y₂)
        inplace!(Δx₃,op₃',y₃)

        @. x̄ = x                                # |  save old x for over-relax

        @. x = (x-τ*(Δx₁+Δx₂+Δx₃-b))/(1+τ)      # |  prox

        @. x̄ = (1+ω)*x - ω*x̄                    # over-relax: x̄ = 2x-x_old

        inplace!(Δy₁,op₁,x̄)                     # dual step: y
        inplace!(Δy₂,op₂,x̄) 
        inplace!(Δy₃,op₃,x̄) 

        @. y₁ = (y₁ + σ*Δy₁) #/(1 + σ*ρ/α₁)
        @. y₂ = (y₂ + σ*Δy₂) #/(1 + σ*ρ/α₂)
        @. y₃ = (y₃ + σ*Δy₃) #/(1 + σ*ρ/α₃)

        proj_norm₂₁ball!(y₁, α₁)                # |  prox
        proj_norm₂₁ball!(y₂, α₂)
        proj_norm₂₁ball!(y₃, α₃)

        if params.accel
            τ, σ = τ*ω, σ/ω
        end

        ################################
        # Give function value if needed
        ################################
        v = verbose() do     
            inplace!(Δy₁,op₁,x)
            inplace!(Δy₂,op₂,x) 
            inplace!(Δy₃,op₃,x) 
            value = norm₂²(b-x)/2 + norm₂₁w(Δy₁, params.α₁) + norm₂₁w(Δy₂, params.α₂) + norm₂₁w(Δy₃, params.α₃)
            value, x
        end

        v
    end

    return x, y₁, y₂, y₃, v
end

end # Module