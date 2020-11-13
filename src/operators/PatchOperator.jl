
export  PatchOperator,
        patch,
        reverse_patch

struct PatchOperator
    size_in::Tuple
    size_out::Tuple
    size_tmp::Tuple
end

# Constructor
function PatchOperator(param,ref)
    sz_in = size(param)
    sz_out = size(ref)
    sz_tmp = (Int(sz_out[1]/sz_in[1]),Int(sz_out[2]/sz_in[2]))
    return PatchOperator(sz_in,sz_out,sz_tmp)
end

function patch(p::PatchOperator,param)
    template = ones(p.size_tmp)
    return kron(param,template)
end

function reverse_patch(p::PatchOperator,x)
    out = zeros(p.size_in)
    els = sum(ones(p.size_tmp))
    M,N = p.size_tmp
    n = 0
    m = 0
    if p.size_out == size(x)
        for i=1:p.size_in[1]
            for j=1:p.size_in[2]
                #println("i=$(m*M+1:(m+1)*M), j=$(n*N+1:(n+1)*N)")
                out[i,j] = sum(x[m*M+1:(m+1)*M,n*N+1:(n+1)*N])/els
                n += 1
            end
            m += 1
            n = 0
        end
    else
        @error "wrong dimensions on output matrix"
    end
    return out
end

