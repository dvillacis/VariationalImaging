
export  proj_norm₂₁ball!,
        norm₂₁w

@inline function proj_norm₂₁ball!(y, α::AbstractArray{T,2}) where {T}
    α²=α .*α
    if ndims(y)==3 && size(y, 1)==2
        @inbounds for i=1:size(y, 2)
            @simd for j=1:size(y, 3)
                n² = y[1,i,j]*y[1,i,j]+y[2,i,j]*y[2,i,j]
                if n² .> α²[i,j]
                    v = α[i,j] ./√n²
                    y[1, i, j] = y[1, i, j] * v
                    y[2, i, j] = y[2, i, j] * v
                end
            end
        end
    else
        y′=reshape(y, (size(y, 1), prod(size(y)[2:end])))

        @inbounds @simd for i=1:size(y′, 2)# in CartesianIndices(size(y)[2:end])
            n² = norm₂²(@view(y′[:, i]))
            if n²>α²[i,j]
                y′[:, i] .*= (α[i,j]/√n²)
            end
        end
    end
end

function norm₂₁w(y::AbstractArray{T,3},w::AbstractArray{T,2}) where {T}
    accum = 0
    if ndims(y)==3 && size(y, 1)==2
        @inbounds for i=1:size(y, 2)
            @simd for j=1:size(y, 3)
                n = √(y[1,i,j]*y[1,i,j]+y[2,i,j]*y[2,i,j])
                accum += w[i,j] * n
            end
        end
    else
        @error "Problems with dimensions"
    end

    return accum
end