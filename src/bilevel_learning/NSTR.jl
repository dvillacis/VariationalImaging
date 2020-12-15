export NSTR

using Base.Iterators
using Krylov

struct TrustRegionModel{TG,TB} 
    g::TG
    B::TB
end

# Iterable
abstract type NTSR_iterable end

struct NSTR_scalar_iterable{Tf}
    f::Tf
    x₀::Real
    η::Real
    Δ₀::Real
end

struct NSTR_sd_iterable{R,Tx,Tf}
    f::Tf
    x₀::Tx
    η::R
    Δ₀::R
end

Base.IteratorSize(::Type{<:NSTR_scalar_iterable}) = Base.IsInfinite()
Base.IteratorSize(::Type{<:NSTR_sd_iterable}) = Base.IsInfinite()

# State
abstract type NTSR_state end

mutable struct NSTR_sd_state{R,Tx,Tfx,TB} <: NTSR_state
    x::Tx
    Δ::R
    fx::Tfx
    B::TB
    res::R
    ρ::R
    error_flag::Bool
end

mutable struct NSTR_scalar_state{Tfx,TB} <: NTSR_state
    x::Real
    Δ::Real
    fx::Tfx
    B::TB
    res::Real
    ρ::Real
    error_flag::Bool
end

function Base.iterate(iter::NSTR_scalar_iterable)
    x = copy(iter.x₀)
    Δ = copy(iter.Δ₀)
    fx = iter.f(x)
    B = 0.1
    res = norm(fx.g,2)
    ρ = 0.0
    state = NSTR_scalar_state(x,Δ,fx,B,res,ρ,false)
    return state,state
end

function Base.iterate(iter::NSTR_sd_iterable)
    x = copy(iter.x₀)
    Δ = copy(iter.Δ₀)
    fx = iter.f(x)
    #B = LSR1Operator(length(x))
    B = LBFGSOperator(length(x))
    res = norm(fx.g[:],2)
    ρ = 0.0
    state = NSTR_sd_state(x,Δ,fx,B,res,ρ,false)
    return state,state
end

function unconstrained_optimum(g,B::Real)
    p = -B\g
    return p, norm(p,2)
end
function unconstrained_optimum(g,B::LSR1Operator)
    p,_ = cg(B,-g) # TODO B puede estar mal condicionada
    return p, norm(p,2)
end

function unconstrained_optimum(g,B::LBFGSOperator)
    if g'*(B*g) > 0
        p,_ = cg(B,-g) # TODO B puede estar mal condicionada
    else
        p = Inf*ones(size(g))
    end
    return p, norm(p,2)
end

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

function cauchy_point_box(x::AbstractArray,Δ,g,B,lb,ub)
    Δmax = 10.0
    γ = min(1,Δmax/norm(g))
    t = 0
    gᵗBg = g'*(B*g)
    if gᵗBg ≤ 0 # Negative curvature detected
        t = (Δ/10.0)*γ
    else
        t = min(norm(g)^2/gᵗBg,(Δ/10.0)*γ)
    end
    d = -t*g
    x_ = x + d
    Px = clamp!(x_,eps(),Inf)
    return Px-x
end

function cauchy_point_box(x::Real,Δ,g,B,lb,ub)
    Δmax = 10.0
    γ = min(1,Δmax/norm(g))
    t = 0
    gᵗBg = g'*(B*g)
    if gᵗBg ≤ 0 # Negative curvature detected
        t = (Δ/10.0)*γ
    else
        t = min(norm(g)^2/gᵗBg,(Δ/10.0)*γ)
    end
    d = -t*g
    x_ = x + d
    if x_ <= 0
        x_ = eps()
    end
    return x_-x
end

function find_intersection(x,Δ)
    lb_total = max.(sqrt(eps()) .-x, -Δ)
    return lb_total,Δ*ones(size(x))
end

# L∞ bound check
function in_bounds(x,lb,ub)
    return all(x .>= lb) & all(x .<= ub)
end

function solve_model(Δ,model)
    pU,pU_norm = unconstrained_optimum(model.g,model.B)
    if pU_norm ≤ Δ
        println("Newton")
        return pU,pU_norm,false
    else
        println("Cauchy")
        pC = cauchy_point(Δ,model.g,model.B)
        return pC,norm(pC,2),false
    end
end

function solve_model_constrained(x,Δ,g,B)
    lb,ub = find_intersection(x,Δ)
    pC = cauchy_point_box(x,Δ,g,B,lb,ub)
    return pC,norm(pC,2),false
    # Identify the bounds
    # lb,ub = find_intersection(x,Δ)

    # pU,pU_norm = unconstrained_optimum(g,B)
    # if in_bounds(pU,lb,ub)
    #     println("Newton")
    #     return pU,pU_norm,false
    # else
    #     pC = cauchy_point_box(x,Δ,g,B,lb,ub)
    #     return pC,norm(pC,2),false
    # end
end

function reduction_ratio(g,B,p,c,c̄)
    pred = - p'*g - 0.5*p'*(B*p)
    ared = c - c̄
    return ared/pred
end

function Base.iterate(iter::NSTR_scalar_iterable{R}, state::NSTR_scalar_state) where {R}

    g = state.fx.g
    p,p_norm,on_boundary = solve_model_constrained(state.x,state.Δ,g,state.B)
    x̄ = state.x + p
    fx̄ = iter.f(x̄)
    state.ρ = reduction_ratio(g,state.B,p,state.fx.c,fx̄.c)
    
    # update SR1 matrix
    y = fx̄.g-g
    s = x̄-x
    if any(isnan.(y))
        @error "Error in gradient calculation"
        state.error_flag=true
    end
    yBs = y - state.B*s
    if abs(yBs'*s) > 0.1*norm₂²(yBs)# Guarantee boundedness of the hessian approximation
        state.B += ((yBs)*(yBs)')/((yBs)'*y) #SR1 Update
    end

    # Radius update
    Δ̄ = 
        if state.ρ < 0.1
            0.25*state.Δ
        elseif state.ρ > 0.75
            min(1e10,2*state.Δ) # Radius update modification for nonsmooth problems
        else
            state.Δ
        end

    # Point update
    if state.ρ ≥ iter.η
        state.Δ = Δ̄
        state.x = x̄
        state.fx = fx̄
    else
        state.Δ = Δ̄
    end

    state.res = norm(state.fx.g,2)
    
    return state,state
end

function Base.iterate(iter::NSTR_sd_iterable{R}, state::NSTR_sd_state) where {R}

    g = state.fx.g[:]
    p,p_norm,on_boundary = solve_model_constrained(state.x[:],state.Δ,g,state.B)
    x̄ = state.x + reshape(p,size(state.x))
    fx̄ = iter.f(x̄)
    state.ρ = reduction_ratio(state.fx.g[:],state.B,p[:],state.fx.c,fx̄.c)
    
    # update SR1 matrix
    y = fx̄.g[:]-g
    s = (x̄-state.x)[:]
    if any(isnan.(y))
        @error "Error in gradient calculation"
        state.error_flag=true
    end
    yBs = y - state.B*s
    if isa(state.B,LBFGSOperator)
        push!(state.B,s,y)
    elseif abs(yBs'*s) > 0.1*norm₂²(yBs)# Guarantee boundedness of the hessian approximation
        state.B += ((yBs)*(yBs)')/((yBs)'*y) #SR1 Update
    end

    # Radius update
    Δ̄ = 
        if state.ρ < 0.1
            0.25*state.Δ
        elseif state.ρ > 0.75
            min(1e10,2*state.Δ) # Radius update modification for nonsmooth problems
        else
            state.Δ
        end

    # Point update
    if state.ρ ≥ iter.η
        state.Δ = Δ̄
        state.x = x̄
        state.fx = fx̄
    else
        state.Δ = Δ̄
    end

    state.res = norm(state.fx.g[:],2)
    
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

"""
solver(NSTR{R})(f,x₀)

Non-smooth trust region solver for problems with box constraints.

This algorithm takes an evaluation function that has to return a tuple (u,c,g) including 
the current point u, the cost at the evaluation point c and the function gradient g.
x₀ is a mandatory initial value. Finally, it is optional to specify the box contraints bounds
that will be used for a in a reflective strategy as described in [X].
"""
function (solver::NSTR{R})(f,x₀::Real;η=0.1,Δ₀=0.1,xmin=1e-8,xmax=Inf) where {R}
    stop(state::NSTR_scalar_state) = state.res <= solver.tol || state.Δ ≤  solver.tol || state.error_flag == true
    @printf("iter | x | cost | grad | radius | ρ\n")
    disp((it,state)) = @printf(
        "%4d | %.3e | %.3e | %.3e | %.3e | %.3e\n",
        it,
        norm(state.x),
        state.fx.c,
        state.res,
        state.Δ,
        state.ρ
    )
    iter = NSTR_scalar_iterable(f,x₀,η,Δ₀)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.fx,state_final.res,num_iters
end

function (solver::NSTR{R})(f,x₀::AbstractArray{T};η=0.1,Δ₀=0.1,xmin=1e-8,xmax=Inf) where {R,T}
    stop(state::NSTR_sd_state) = state.res <= solver.tol || state.Δ ≤  solver.tol || state.error_flag == true
    @printf("iter | x | cost | grad | radius | ρ \n")
    disp((it,state)) = @printf(
        "%4d | %.3e | %.3e | %.3e | %.3e | %.3e\n",
        it,
        norm(state.x),
        state.fx.c,
        norm(state.res),
        state.Δ,
        state.ρ
    )
    iter = NSTR_sd_iterable(f,x₀,η,Δ₀)
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