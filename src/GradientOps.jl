
########################
# Discretised gradients operators
########################

__precompile__()

module GradientOps

using SparseArrays
using AlgTools.Util
using AlgTools.LinOps
using ImageTools.Gradient

export FwdGradientOp, BwdGradientOp, CenteredGradientOp, ZeroOp, PatchOp, matrix

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

# Sparse Matrix representation of the linear operator, assumimg a vectorial image
function matrix(op::FwdGradientOp,n)
    Hx = spdiagm(0=>-ones(n-1),1=>ones(n-1))
	Gx = kron(spdiagm(0=>ones(n)),Hx)
	Gy = spdiagm(0=>-ones(n^2-n),n=>ones(n^2-n))
	return [Gx;Gy]
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

# Sparse Matrix representation of the linear operator, assumimg a vectorial image
function matrix(op::BwdGradientOp,n)
    Hx = spdiagm(-1=>-ones(n-1),0=>[0;ones(n-1)])
	Gx = kron(spdiagm(0=>ones(n)),Hx)
	Gy = spdiagm(-n=>-ones(n^2-n),0=>[spzeros(n);ones(n^2-n)])
	return [Gx;Gy]
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

# Sparse Matrix representation of the linear operator, assumimg a vectorial image
function matrix(op::CenteredGradientOp,n)
    Hx = spdiagm(-1=>-ones(n-1),0=>[-1;zeros(n-2);1],1=>ones(n-1))
	Gx = kron(spdiagm(0=>ones(n)),Hx)
	Gy = spdiagm(-n=>-ones(n^2-n),0=>[-ones(n);spzeros(n);ones(n)],n=>ones(n^2-n))
	return [Gx;Gy]
end

###########################
# Zero operator
###########################

struct ZeroOp{X} <: LinOp{X,X}
end

function (op::ZeroOp{X})(x::X) where X
    return 0 .*x
end

function Base.adjoint(op::ZeroOp{X}) where X
    return op
end

function LinOps.opnorm_estimate(op::ZeroOp{X}) where X
    return 0
end

###########################
# Patch Operator
###########################

struct PatchOp <: AdjointableOp{Primal,Primal}
    size_in::Tuple
    size_out::Tuple
    size_ratio::Tuple
end

function PatchOp(param::AbstractArray{T,2},ref::AbstractArray{T,2}) where T
    sz_in = size(param)
    sz_out = size(ref)
    sz_ratio = (Int(sz_out[1]/sz_in[1]),Int(sz_out[2]/sz_in[2]))
    return PatchOp(sz_in,sz_out,sz_ratio)
end

function PatchOp(param::AbstractArray{T,3},ref::AbstractArray{T,2}) where T
    m,n,o = size(param)
    sz_in = (m,n)
    sz_out = size(ref)
    sz_ratio = (Int(sz_out[1]/sz_in[1]),Int(sz_out[2]/sz_in[2]))
    return PatchOp(sz_in,sz_out,sz_ratio)
end

function (op::PatchOp)(x::Primal)
    y = zeros(op.size_out)
    inplace!(y,op,x)
    return y
end

function (op::PatchOp)(x::AbstractArray{T,3}) where T
    m,n,o = size(x)
    y = zeros(op.size_out...,o)
    for i=1:o
        r = @view y[:,:,i]
        inplace!(r,op,x[:,:,i])
    end
    return y
end

function LinOps.inplace!(y::Union{Primal,SubArray}, op::PatchOp, x::Primal)
    template = ones(op.size_ratio)
    y .= kron(x,template)
end

function LinOps.calc_adjoint(op::PatchOp, y::Primal)
    res = zeros(op.size_in)
    calc_adjoint!(res,op,y)
    return res
end

function LinOps.calc_adjoint!(res::Primal, op::PatchOp, y::Primal)
    M,N = op.size_ratio
    n = 0
    m = 0
    if op.size_out == size(y)
        for i=1:op.size_in[1]
            for j=1:op.size_in[2]
                res[i,j] = sum(y[m*M+1:(m+1)*M,n*N+1:(n+1)*N])
                n += 1
            end
            m += 1
            n = 0
        end
    else
        @error "wrong dimensions on output matrix"
    end
end

function LinOps.opnorm_estimate(op::PatchOp)
    return 1
end



end # Module
