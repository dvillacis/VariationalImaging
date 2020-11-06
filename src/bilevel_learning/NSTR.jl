export NSTR

using Base.Iterators
using Krylov
using Random

struct TrustRegionModel{TG,TB} 
    g::TG
    B::TB
end

# Iterable
struct NSTR_iterable{R,Tx,Tf}
    f::Tf
    x₀::Tx
    η::R
    Δ₀::R
end

Base.IteratorSize(::Type{<:NSTR_iterable}) = Base.IsInfinite()

# State
mutable struct NSTR_state{R,Tx,Tfx,TB}
    x::Tx
    Δ::R
    fx::Tfx
    B::TB
    res::R
    ρ::R
end

function Base.iterate(iter::NSTR_iterable)
    x = copy(iter.x₀)
    Δ = copy(iter.Δ₀)
    fx = iter.f(x)
    if length(x) > 1
        B = LBFGSOperator(length(x))
    else
        B = 0.1
    end
    res = 1.0
    ρ = 0.0
    state = NSTR_state(x,Δ,fx,B,res,ρ)
    return state,state
end

function unconstrained_optimum(model)
    if isa(model.B,LBFGSOperator)
        p,_ = cg(model.B,-model.g) # TODO B puede estar mal condicionada
    else
        p = -model.B\model.g
    end
    return p, norm(p,2)
end

function cauchy_point(Δ,model)
    t = 0
    gᵗBg = model.g'*(model.B*model.g)
    if gᵗBg ≤ 0
        t = Δ/norm(model.g)
    else
        t = min(norm(model.g)^2/gᵗBg,Δ/norm(model.g))
    end
    return -t*model.g
end


function solve_model(Δ,model)
    pU,pU_norm = unconstrained_optimum(model)
    if pU_norm ≤ Δ
        return pU,pU_norm,false
    else
        pC = cauchy_point(Δ,model)
        return pC,norm(pC,2),false
    end
end

function reduction_ratio(model,p,fx,fx̄)
    #println("$(fx.c), $(fx̄.c), $p")
    pred = - p'*model.g - 0.5*p'*(model.B*p)
    ared = fx.c - fx̄.c
    #println("pred = $pred, ared = $ared")
    return ared/pred
end

function Base.iterate(iter::NSTR_iterable{R}, state::NSTR_state) where {R}
    
    model = TrustRegionModel(state.fx.g,state.B)
    p,p_norm,on_boundary = solve_model(state.Δ,model)
    x̄ = state.x .+ p
    fx̄ = iter.f(x̄)
    state.ρ = reduction_ratio(model,p,state.fx,fx̄)
    
    # update SR1 matrix
    y = fx̄.g-state.fx.g
    s = x̄-state.x
    if isa(state.B,LBFGSOperator)
        push!(state.B,s,y)
    elseif abs(y) > 0
        yBs = y - state.B*s
        state.B += ((yBs)*(yBs)')/((yBs)'*y) #SR1 Update
        if bitrand(1) == [1]
            state.B = 0.1
        end
        println(state.B)
    end

    # Radius update
    Δ̄ = 
        if state.ρ < 0.25
            0.5*state.Δ
        elseif state.ρ > 0.75
            2*state.Δ
        else
            state.Δ
        end

    # Point update
    #println("ρ = $ρ")
    if state.ρ ≥ iter.η
        state.Δ = Δ̄
        state.x = x̄
        state.fx = fx̄
    else
        state.Δ = Δ̄
    end

    state.res = norm(state.fx.g)
    
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

function (solver::NSTR{R})(f,x₀::Real;η=0.1,Δ₀=0.1) where {R}
    stop(state::NSTR_state) = state.res <= solver.tol || state.Δ ≤  solver.tol
    @printf("iter | x | cost | grad | radius | ρ\n")
    disp((it,state)) = @printf(
        "%4d | %.3e | %.3e | %.3e | %.3e | %.3e\n",
        it,
        norm(state.x),
        state.fx.c,
        # norm(state.fx.g),
        state.fx.g,
        state.Δ,
        state.ρ
    )
    iter = NSTR_iterable(f,x₀,η,Δ₀)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.fx,state_final.res,num_iters
end

function (solver::NSTR{R})(f,x₀::AbstractArray{T};η=0.1,Δ₀=0.1) where {R,T}
    stop(state::NSTR_state) = state.res <= solver.tol || state.Δ ≤  solver.tol
    @printf("iter | x | cost | grad | radius\n")
    disp((it,state)) = @printf(
        "%4d | %.3e | %.3e | %.3e | %.3e\n",
        it,
        norm(state.x),
        state.fx.c,
        norm(state.fx.g),
        state.Δ
    )
    iter = NSTR_iterable(f,x₀,η,Δ₀)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.fx,state_final.res,num_iters
end

NSTR(::Type{R};kwargs...) where{R} = NSTR{R}(;kwargs...)
NSTR(;kwargs...) = NSTR(Float64;kwargs...)