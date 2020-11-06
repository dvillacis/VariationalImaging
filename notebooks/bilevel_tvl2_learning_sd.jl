### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 8aeec1f4-1ae6-11eb-3434-09070607d869
using Revise

# ╔═╡ a25eb5e2-1ae6-11eb-1e09-9956b46d95df
using VariationalImaging, Noise, TestImages, Images, ImageQualityIndexes, LinearAlgebra, LinearOperators, Krylov

# ╔═╡ b6057fd8-1ae6-11eb-2515-cb91ef2bf540
begin
	M,N = 50,50
	img = 0.2*LowerTriangular(ones(M,N))+0.6*UpperTriangular(ones(M,N));
	noisy = img .* LowerTriangular(ones(M,N)) .+ UpperTriangular(add_gauss(img,0.1));
end

# ╔═╡ d0aef4f4-1ae6-11eb-2048-557123397904
Gray.([img noisy])

# ╔═╡ d723e342-1ae6-11eb-0e70-4f847b7ef66f
α=0.2*LowerTriangular(ones(M,N))

# ╔═╡ eb1a4b8e-1ae6-11eb-171f-7bb21c7185e4
u = TVDenoising(noisy,α;maxit=200,verbose=true)

# ╔═╡ 07d44e20-1ae7-11eb-254e-fdbd68333e9e
Gray.([img noisy u])

# ╔═╡ 0eee319c-1ae7-11eb-389f-7feaf7261bb1
[assess_ssim(img,noisy) assess_ssim(img,u)]

# ╔═╡ 9ece2eee-1aeb-11eb-1f36-eb8c9d55f380
function gamma(x)
	y = zeros(size(x))
	n = Int(length(x)/2)
	for i = 1:n
		a = (sqrt(x[i]^2+x[1+n]^2))
		y[i] = x[i]/a
		y[i+n]=x[i+n]/a
	end
	return y
end

# ╔═╡ 274e1b4e-1ae7-11eb-01a5-79bf72293145
function tr_fun(x)
	m,n = size(img)
    L = Gradient(m,n)
    f = L2DataTerm(noisy[:])
    g = NormL21(x)
    s = PDHG(;maxit=200, verbose=false)
    u,r,i = s(f.b,L*f.b,f,g,L)
	c = 0.5*norm(u[:]-img[:])^2
	Lu = L*u
	A = [opEye(m*n) Diagonal(x[:])*L']
	track = u[:]-img[:]
	adj,status = lsqr(A,track)
	if status.solved == false
		@error "$status"
	end
	adj = adj[1:m*n]
	g = Diagonal(adj)*(L'*gamma(Lu))
	return (u=x,c=c,g=g)
end

# ╔═╡ 714d5c92-1aeb-11eb-3235-f7e786b7e462
tr_fun(α)

# ╔═╡ ce5ca852-1aeb-11eb-0245-25ed2098551d
solver = NSTR(;verbose=true,freq=5,maxit=100)

# ╔═╡ d4321596-1aeb-11eb-0921-ed7544d57886
opt_par,res,iters = solver(tr_fun,0.6*ones(M*N))

# ╔═╡ c0458396-1aec-11eb-042e-11ce6bb647c3
Gray.(reshape(opt_par,M,N))

# ╔═╡ 0009ccc2-1aef-11eb-0d2e-eff4191e57b2
sol = TVDenoising(noisy,reshape(opt_par,M,N))

# ╔═╡ 1696d1c6-1aef-11eb-049b-133770cd96c6
Gray.([img noisy sol])

# ╔═╡ Cell order:
# ╠═8aeec1f4-1ae6-11eb-3434-09070607d869
# ╠═a25eb5e2-1ae6-11eb-1e09-9956b46d95df
# ╠═b6057fd8-1ae6-11eb-2515-cb91ef2bf540
# ╠═d0aef4f4-1ae6-11eb-2048-557123397904
# ╠═d723e342-1ae6-11eb-0e70-4f847b7ef66f
# ╠═eb1a4b8e-1ae6-11eb-171f-7bb21c7185e4
# ╠═07d44e20-1ae7-11eb-254e-fdbd68333e9e
# ╠═0eee319c-1ae7-11eb-389f-7feaf7261bb1
# ╠═9ece2eee-1aeb-11eb-1f36-eb8c9d55f380
# ╠═274e1b4e-1ae7-11eb-01a5-79bf72293145
# ╠═714d5c92-1aeb-11eb-3235-f7e786b7e462
# ╠═ce5ca852-1aeb-11eb-0245-25ed2098551d
# ╠═d4321596-1aeb-11eb-0921-ed7544d57886
# ╠═c0458396-1aec-11eb-042e-11ce6bb647c3
# ╠═0009ccc2-1aef-11eb-0d2e-eff4191e57b2
# ╠═1696d1c6-1aef-11eb-049b-133770cd96c6
