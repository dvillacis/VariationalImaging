
########################
# Discretised gradients operators
########################

__precompile__()

module GradientOps

using AlgTools.Util
using AlgTools.LinOps
using ImageTools.Gradient

export FwdGradientOp, BwdGradientOp, CenteredGradientOp

Primal = Array{Float64,2}
Dual = Array{Float64,3}

###########################
# Forward differences gradient linear operator
###########################

struct FwdGradientOp <: AdjointableOp{Primal, Dual}

end

function (op::FwdGradientOp)(x::Primal)
    y = zeros(2,size(x)...)
    ∇₂!(y,x)
    return y
end

function LinOps.inplace!(y::Dual, op::FwdGradientOp, x::Primal)
    ∇₂!(y,x)
end

function LinOps.calc_adjoint(op::FwdGradientOp, y::Dual)
    res = zeros(size(y,2),size(y,3))
    calc_adjoint!(res,op,y)
    return res
end

function LinOps.calc_adjoint!(res::Primal, op::FwdGradientOp, y::Dual)
    ∇₂ᵀ!(res,y)
end

function LinOps.opnorm_estimate(op::FwdGradientOp)
    return sqrt(8)
end


###########################
# Backward differences gradient linear operator
###########################

function ∇₂b!(u₁, u₂, u)
    @. @views begin
        u₁[2:end, :] = u[1:(end-1), :] - u[2:end, :]
        u₁[1, :, :] = 0

        u₂[:, 2:end] = u[:, 1:(end-1)] - u[:, 2:end]
        u₂[:, 1] = 0
    end
    return u₁, u₂
end

function ∇₂b!(v, u)
    ∇₂b!(@view(v[1, :, :]), @view(v[2, :, :]), u)
end

function ∇₂bᵀ!(v, v₁, v₂)
    @. @views begin
        v[2:(end-1), :] = v₁[3:end, :] - v₁[2:(end-1), :]
        v[1, :] = v₁[2, :]
        v[end, :] = -v₁[end, :]

        v[:, 2:(end-1)] += v₂[:, 3:end] - v₂[:, 2:(end-1)]
        v[:, 1] += v₂[:, 2]
        v[:, end] += -v₂[:, end]
    end
    return v
end

function ∇₂bᵀ!(u, v)
    ∇₂bᵀ!(u, @view(v[1, :, :]), @view(v[2, :, :]))
end

struct BwdGradientOp <: AdjointableOp{Primal, Dual}

end

function (op::BwdGradientOp)(x::Primal)
    y = zeros(2,size(x)...)
    ∇₂b!(y,x)
    return y
end

function LinOps.inplace!(y::Dual, op::BwdGradientOp, x::Primal)
    ∇₂b!(y,x)
end

function LinOps.calc_adjoint(op::BwdGradientOp, y::Dual)
    res = zeros(size(y,2),size(y,3))
    calc_adjoint!(res,op,y)
    return res
end

function LinOps.calc_adjoint!(res::Primal, op::BwdGradientOp, y::Dual)
    ∇₂bᵀ!(res,y)
end

function LinOps.opnorm_estimate(op::BwdGradientOp)
    return sqrt(8)
end


###########################
# Centered differences gradient linear operator
###########################

function ∇₂cᵀ!(v, v₁, v₂)
    @. @views begin
        v[2:(end-1), :] = (v₁[1:(end-2), :] - v₁[3:end, :])/2
        v[1, :] = (-v₁[1, :] - v₁[2,:])/2
        v[end, :] = (v₁[end-1, :] + v₁[end,:])/2

        v[:, 2:(end-1)] += (v₂[:, 1:(end-2)] - v₂[:, 3:end])/2
        v[:, 1] += (-v₂[:, 1] - v₂[:, 2])/2
        v[:, end] += (v₂[:, end] + v₂[:, end-1])/2
    end
    return v
end

function ∇₂cᵀ!(u, v)
    ∇₂cᵀ!(u, @view(v[1, :, :]), @view(v[2, :, :]))
end

struct CenteredGradientOp <: AdjointableOp{Primal, Dual}

end

function (op::CenteredGradientOp)(x::Primal)
    y = zeros(2,size(x)...)
    ∇₂c!(y,x)
    return y
end

function LinOps.inplace!(y::Dual, op::CenteredGradientOp, x::Primal)
    ∇₂c!(y,x)
end

function LinOps.calc_adjoint(op::CenteredGradientOp, y::Dual)
    res = zeros(size(y,2),size(y,3))
    calc_adjoint!(res,op,y)
    return res
end

function LinOps.calc_adjoint!(res::Primal, op::CenteredGradientOp, y::Dual)
    ∇₂cᵀ!(res,y)
end

function LinOps.opnorm_estimate(op::CenteredGradientOp)
    return sqrt(8)
end

end # Module
