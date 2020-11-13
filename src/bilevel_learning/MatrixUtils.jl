
using SparseArrays

export 	createDivMatrix,
		prodesc,
		xi,
		patch,
		invpatch

function createDivMatrix(n)
	Hx = spdiagm(-1=>-ones(n-1),1=>ones(n-1))
	Gx = kron(spdiagm(0=>ones(n)),Hx)
	Gy = spdiagm(-n=>-ones(n^2-n),n=>ones(n^2-n))
	return [Gy;Gx]
end

function prodesc(q,p)
	n=Int(size(q,1)/2)
	q1=q[1:n]
	q2=q[n+1:2*n]

	p1=p[1:n]
	p2=p[n+1:2*n]

	return [spdiagm(0=>p1.*q1) spdiagm(0=>p2.*q1);
		  spdiagm(0=>p1.*q2) spdiagm(0=>p2.*q2)]
end

function xi(x)
	y = zeros(size(x))
	n = Int(size(x,1)/2)
	for i = 1:n
		y[i] = norm([x[i];x[i+n]],2)
		y[i+n] = norm([x[i];x[i+n]],2)
	end
	return y
end	