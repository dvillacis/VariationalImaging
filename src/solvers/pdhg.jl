export PDHG

using LinearAlgebra
using Base.Iterators

# Iterable
struct PDHG_iterable{R,Tf,Tg,TL,Tx,Ty}
    f::Tf
    g::Tg
    L::TL
    τ::R
    σ::R
    x0::Tx
    y0::Ty
end

Base.IteratorSize(::Type{<:PDHG_iterable}) = Base.IsInfinite()

mutable struct PDHG_state{R,Tx,Ty}
    x::Tx
    y::Ty
    temp_x::Tx
    temp_y::Ty
    x̄::Tx
    Δx::Tx
    Δy::Ty
    res::R
end

function Base.iterate(iter::PDHG_iterable)
    x = copy(iter.x0)
    y = copy(iter.y0)
    temp_y = copy(iter.y0)
    temp_x = copy(iter.x0)
    x̄ = copy(iter.x0)
    Δx = copy(iter.x0)
    Δy = copy(iter.y0)
    res = 1.0
    state = PDHG_state(x,y,temp_x,temp_y,x̄,Δx,Δy,res)
    return state,state
end

function Base.iterate(iter::PDHG_iterable{R}, state::PDHG_state) where {R}
    state.Δy = iter.L*state.x̄
    state.temp_y .= state.y .+ iter.σ .* state.Δy 
    cprox!(state.y,iter.g,state.temp_y, iter.σ)
    state.x̄ .= state.x
    state.Δx = iter.L'*state.y
    state.temp_x .= state.x .- iter.τ .* state.Δx
    prox!(state.x,iter.f,state.temp_x, iter.τ)
    state.x̄ .= 2 .* state.x .- state.x̄

    # Residual calculation (PD gap)
    primal = iter.f(state.x̄) + iter.g(iter.L*state.x̄)
    dual = -conjugate(iter.f,iter.L'*state.y)
    state.res = primal
    return state,state
end


# Solver
struct PDHG{R}
    maxit::Int
    tol::R
    verbose::Bool
    freq::Int

    function PDHG{R}(;
        maxit::Int = 10000,
        tol::R = 1e-5,
        verbose::Bool = false,
        freq::Int = 100,
    ) where {R}
       @assert maxit > 0
       @assert tol > 0
       @assert freq > 0
       new(maxit,tol,verbose,freq) 
    end
end

function (solver::PDHG{R})(x0::AbstractArray{T,1},
            y0::AbstractArray{T,1},
            f::AbstractDataTerm,
            g::AbstractRegularizationTerm,
            L::LinearOperator) where {R,T}
    stop(state::PDHG_state) = state.res <= solver.tol
    disp((it,state)) = @printf(
        "%5d | %.3e\n",
        it,
        state.res
    )
    τ = 0.01
    σ = 1 / τ / √8 # Calculate operator norm
    iter = PDHG_iterable(f,g,L,τ,σ,x0,y0)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.res,num_iters
end

PDHG(::Type{R};kwargs...) where{R} = PDHG{R}(;kwargs...)
PDHG(;kwargs...) = PDHG(Float64;kwargs...)

function pdhg(f::AbstractDataTerm,
                g::AbstractRegularizationTerm,
                L::LinearOperator;
                x0::AbstractArray{T,1}=zeros(L.ncol),
                y0::AbstractArray{T,1}=zeros(L.nrow)) where {T}
    
    τ = 0.01
    σ = 1 / τ / √8 # Calculate operator norm

    y = y0
    x = x0
    x̄ = x
    res = 1
    freq = 100
    maxit = 1000
    tol = 1e-5

    for i = 1:maxit

        Δy = L * x̄ 
        cprox!(y,g,y + σ * Δy,σ)

        x̄ .= x

        Δx = L'* y
        prox!(x,f,x - τ * Δx,τ)

        x̄ .= 2*x - x̄
        
        #res = pd_gap(L,x,y,f,g,g.α)
        # if res < tol
        #     println("$i: $res")
        #     break
        # end
        if i%freq == 0
            println("$i: $res")
        end
    end

    return x
end

function pd_gap(L,x,y,f,g,α)
    Lty = L'*y
    Lx = L*x
    primal = f(x)+α*g(Lx)
    dual = -0.5*norm(Lty)^2 + f.b'*Lty
    return primal-dual
end