export pdhg

function pdhg(f::AbstractImagingFunction,g::AbstractImagingFunction,L::AbstractLinearOperator;
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