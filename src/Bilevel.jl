########################################################
# Bilevel Parameter Learning via Nonsmooth Trust Region 
########################################################

__precompile__()

module Bilevel

using AlgTools.Util
using AlgTools.LinOps
import AlgTools.Iterate

using LinearAlgebra

export bilevel_learn

#############
# Data types
#############

ImageSize = Tuple{Integer,Integer}
Image = Array{Float64,2}
Primal = Image
Dual = Array{Float64,3}
Parameter = Union{Real,AbstractArray}
TrainingPair = Tuple{Image,Image}

#########################
# Iterate initialisation
#########################

function init_rest(x::Parameter,learning_function::Function,Δ)
    x̄ = copy(x)
    fx,gx = learning_function(x)
    fx̄ = copy(fx)
    gx̄ = copy(gx)
    B = 0.1*Diagonal(ones(size(x)))
    return x, x̄, fx, gx, fx̄, gx̄, Δ, B
end

############
# Auxiliary Functions
############

function cauchy_point(Δ,g,B)
    t = 0
    gᵗBg = g'*(B*g)
    if gᵗBg ≤ 0 # Negative curvature detected
        t = Δ/norm(g)
    else
        t = min(norm(g)^2/gᵗBg,Δ/norm(g))
    end
    return -t*g
end

############
# Algorithm
############

function bilevel_learn(pair :: TrainingPair,
    learning_function::Function;
    xinit :: Parameter,
    iterate = Iterate.simple_iterate,
    params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################                    

    η₁, η₂ = params.η₁, params.η₂
    β₁, β₂ =  params.β₁, params.β₂
    Δ₀ = params.Δ₀

    ######################
    # Initialise iterates
    ######################

    x, x̄, fx, gx, fx̄, gx̄, Δ, B = init_rest(xinit,learning_function,Δ₀)

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        
        p = cauchy_point(Δ,gx,B) # solve tr subproblem

        @. x̄ = x = p  # test new point

        fx̄,gx̄ = learning_function(x̄,pair)
        ρ = (-p'*gx - 0.5*p'*(B*p))/(fx-fx̄) # pred/ared

        if ρ < η₁               # radius update
            Δ = β₁*Δ
        elseif ρ > η₂
            Δ = β₂*Δ
        end

        if ρ > η₂
            @. x = x̄
            @. fx = fx̄
            @. gx = gx̄
        end

        ################################
        # Give function value if needed
        ################################
        v = verbose() do     
            value = fx
            value, x
        end

        v
    end

    return x, y, v
end

end # Module