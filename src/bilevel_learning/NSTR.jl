export NSTR

using Base.Iterators
using Krylov

# Iterable
struct NSTR_iterable{R,Tx}
    x0::Tx
    noisy::AbstractArray
    upper_level_cost::Function
    gradient_solver::Function
    lower_level_solver::Function
    η₁::R # Radius decrease
    η₂::R
    ρ₁::R # Quality measure
    ρ₂::R
    Δ₀::R # Initial radius
end

Base.IteratorSize(::Type{<:NSTR_iterable}) = Base.IsInfinite()

# State
mutable struct NSTR_state{R,Tx,Timg}
    x::Tx
    denoised::Timg
    cost::Real
    cost_::Real
    grad::Real
    grad_::Real
    hess::LBFGSOperator
    Δ::R
    step::R
    res::R
end

function Base.iterate(iter::NSTR_iterable)
    x = copy(iter.x0)
    x̄ = copy(iter.x0)
    Δ = copy(iter.Δ₀)
    denoised = iter.lower_level_solver(iter.x0)
    cost = iter.upper_level_cost(denoised)
    cost_ = copy(cost)
    grad = iter.gradient_solver(iter.x0)
    grad_ = copy(grad)
    hess = LBFGSOperator(1)
    step= -0.1*grad
    res = copy(cost)
    state = NSTR_state(x,denoised, cost, cost_,grad,grad_,hess,Δ,step,res)
    return state,state
end

function Base.iterate(iter::NSTR_iterable{R}, state::NSTR_state) where {R}
    
    pred = -state.grad*state.step-0.5*state.step*state.hess*state.step

    state.denoised = iter.lower_level_solver(state.x+state.step)
    state.cost_ = iter.upper_level_cost(state.denoised)
    ared = state.cost-state.cost_

    println(ared/pred)

    state.x += iter.ρ₁*state.grad*state.Δ
    state.res = state.cost
    return state,state
end


# Solver
struct NSTR{R}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function NSTR{R}(;
        maxit::Int = 100,
        tol::R = 1e-5,
        verbose::Bool = false,
        freq::Int = 10,
    ) where {R}
        @assert maxit > 0
        @assert tol > 0
        @assert freq > 0
        new(maxit,tol,verbose,freq)
    end
end

function (solver::NSTR{R})(x0::Real,
    noisy::AbstractArray{R,1},
    upper_level_cost::Function,
    gradient_solver::Function,
    lower_level_solver::Function) where {R}
    stop(state::NSTR_state) = state.res <= solver.tol
    disp((it,state)) = @printf(
        "%5d | %.3e | %.3e | %.3e\n",
        it,
        state.x,
        state.Δ,
        state.cost
    )
    η₁ = 0.5
    η₂ = 2.0
    ρ₁ = 0.1
    ρ₂ = 0.9
    Δ₀ = 1.0
    iter = NSTR_iterable(x0,noisy,upper_level_cost,gradient_solver,lower_level_solver,η₁,η₂,ρ₁,ρ₂,Δ₀)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
    iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.res,num_iters
end

NSTR(::Type{R};kwargs...) where{R} = NSTR{R}(;kwargs...)
NSTR(;kwargs...) = NSTR(Float64;kwargs...)