
export batchmul!, batchmul

function batchmul!(C, A, B)
	@views C[1,:, :] = A[1,:, :] .* B
	@views C[2,:, :] = A[2,:, :] .* B
    return C
end

function batchmul(A, B)
    C = zeros(size(A))
    return batchmul!(C, A, B)
end