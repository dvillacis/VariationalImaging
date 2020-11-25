export NSTR

using Base.Iterators
using Krylov

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
    error_flag::Bool
end

function Base.iterate(iter::NSTR_iterable)
    x = copy(iter.x₀)
    Δ = copy(iter.Δ₀)
    fx = iter.f(x)
    if length(x) > 1
        #B = LSR1Operator(length(x))
        B = LBFGSOperator(length(x))
    else
        B = 0.1
    end
    res = norm(fx.g[:],2)
    ρ = 0.0
    state = NSTR_state(x,Δ,fx,B,res,ρ,false)
    return state,state
end

function unconstrained_optimum(g,B::Real)
    p = -B\g
    return p, norm(p,2)
end
function unconstrained_optimum(g,B::LSR1Operator)
    if g'*(B*g) > 0
        p,_ = cg(B,-g) # TODO B puede estar mal condicionada
    else
        p = Inf*ones(size(g))
    end
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

function cauchy_point_2(x,Δ,g,B,lb,ub)
    t = 0
    #println("lb=$lb")
    gᵗBg = g'*(B*g)
    if gᵗBg ≤ 0 # Negative curvature detected
        println("neg. curvature")
        t = step_size_to_bound(zeros(size(x)),-g,lb,ub)
    else
        #println("pos. curvature: a=$(norm(g)^2/gᵗBg), b=$(Δ/norm(g))")
        min_step = step_size_to_bound(zeros(size(x)),-g,lb,ub)
        t = min(norm(g)^2/gᵗBg,min_step)
    end
    s = -t*g
    return s
end

function find_intersection(x,Δ)
    lb_total = max.(sqrt(eps()) .-x, -Δ)
    return lb_total,Δ*ones(size(x))
end

# L∞ bound check
function in_bounds(x,lb,ub)
    return all(x .>= lb) & all(x .<= ub)
end

function step_size_to_bound(x,s,lb,ub)
    nz = findall(iszero,isapprox.(s,0;atol=sqrt(eps())))
    snz = s[nz]
    steps = Inf .* ones(size(x))
    lbx = lb .-x
    steps[nz] = max.(lbx[nz] ./snz, (ub .- x)[nz] ./snz)
    println("steps = $(minimum(steps[nz]))")
    min_step = minimum(steps[nz])
    min_step2 = 2e-8*mean(steps[nz])
    return min_step2
end

function cauchy_point_box(x,g,lb,ub)
    to_bounds = step_size_to_bound(zeros(size(x)),-g,lb,ub)
    return -to_bounds*g
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
    # Identify the bounds
    lb,ub = find_intersection(x,Δ)

    pU,pU_norm = unconstrained_optimum(g,B)
    if in_bounds(pU,lb,ub)
        println("Newton")
        return pU,pU_norm,false
    else
        pC = cauchy_point_2(x,Δ,g,B,lb,ub)
        return pC,norm(pC,2),false
    end
end

function reduction_ratio(g,B,p,fx,fx̄)
    println("p=$(norm(p))")
    pred = - p'*g - 0.5*p'*(B*p)
    ared = fx.c - fx̄.c
    return ared/pred
end

function Base.iterate(iter::NSTR_iterable{R}, state::NSTR_state) where {R}

    #model = TrustRegionModel(state.fx.g[:],state.B)
    #p,p_norm,on_boundary = solve_model(state.Δ,model)
    p,p_norm,on_boundary = solve_model_constrained(state.x[:],state.Δ,state.fx.g[:],state.B)
    #println("p=$(norm(p))")
    x̄ = state.x + reshape(p,size(state.x))
    #clamp!(x̄,eps(),Inf)
    #println("x̄=$x̄")
    fx̄ = iter.f(x̄)
    state.ρ = reduction_ratio(state.fx.g[:],state.B,p,state.fx,fx̄)
    
    # update SR1 matrix
    #println("$x̄, $(fx̄.g), $(state.fx.g)")
    y = fx̄.g[:]-state.fx.g[:]
    s = x̄[:]-state.x[:]
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
    stop(state::NSTR_state) = state.res <= solver.tol || state.Δ ≤  solver.tol || state.error_flag == true
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
    iter = NSTR_iterable(f,x₀,η,Δ₀)
    iter = take(halt(iter,stop),solver.maxit)
    iter = enumerate(iter)
    if solver.verbose
        iter = tee(sample(iter,solver.freq),disp)
    end

    num_iters, state_final = loop(iter)
    return state_final.x,state_final.fx,state_final.res,num_iters
end

function (solver::NSTR{R})(f,x₀::AbstractArray{T};η=0.1,Δ₀=0.1,xmin=1e-8,xmax=Inf) where {R,T}
    stop(state::NSTR_state) = state.res <= solver.tol || state.Δ ≤  solver.tol || state.error_flag == true
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