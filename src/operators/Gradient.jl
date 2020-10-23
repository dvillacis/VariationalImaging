
import LinearOperators.AbstractLinearOperator

export Gradient

"""`Gradient()`
Finite Differences Gradient Operator for Imaging
```
L = Gradient(M,N)
v = rand(M*N)
```
"""
struct Gradient <: AbstractLinearOperator{Any} end

# function show(io:IO, op::Gradient)
#     println(io,"∇")
# end

function product(x::AbstractArray{T,1},M,N) where {T}
    @assert length(x) == M*N
    y = zeros(2*M*N)
    for i = 1:M*N-N
        y[i] = x[i+N] - x[i]
    end
    for i = M*N-N+1:M*N
        y[i] = x[i]
    end
    for i = 1:M*N
        if (i%M)==0
            y[i+M*N] = x[i]
        else
            y[i+M*N] = x[i+1]-x[i]
        end
    end
    return y
end

function tproduct(x::AbstractArray{T,1},M,N) where {T}
    @assert length(x) == 2*M*N
    y = zeros(M*N)
    y[1] = -x[1]-x[1+M*N]
    for i = 2:M
        y[i] = -x[i]-x[i+M*N]+x[i+M*N-1]
    end
    for i = M+1:M*N
        y[i] = x[i-M]-x[i]-x[i+M*N]+x[i+M*N-1]
    end
    return y
end

# Constructors
function Gradient(T::DataType,M::Int,N::Int)
    p = v -> product(v,M,N)
    t = v -> tproduct(v,M,N)
    LinearOperator{T}(2*M*N,M*N,false,false,p,t,t)
end

Gradient(M::Int,N::Int) = Gradient(Float64,M,N)