### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 8da2c26a-1fa0-11eb-2f91-216fa29571de
using Revise

# ╔═╡ 9751886e-1fa0-11eb-0c9e-ad8f7c66d3b9
using AlgTools.Util, JLD2, FileIO, Plots

# ╔═╡ a55a8e12-1fa0-11eb-2438-798a1c4ac0ef
using VariationalImaging

# ╔═╡ e16f734c-1fab-11eb-3182-55a956044637
using ImageQualityIndexes, ImageContrastAdjustment, ImageShow

# ╔═╡ 678d847c-2057-11eb-0609-e517d1990028
using ImageTools.Gradient

# ╔═╡ be3b79a2-22a8-11eb-24f6-2f469ff92a30
using LinearAlgebra, SparseArrays, BenchmarkTools

# ╔═╡ 32b60ce0-1fa0-11eb-29d3-1d3df6525a7d
md"""
# TV-l₂ Scalar Parameter Learning
In this notebook we will make use of a trust-region algorithm for finding the optimal scalar regularization parameter for the TV-l₂ image denoising model. Fisrt let us recall the form of the cost function.
"""

# ╔═╡ b35dd92e-1fa0-11eb-15bb-9723015e8379
begin
	data = FileIO.load("tvl2_lena_experiment.jld2")
	img = data["img"]
	noisy = data["noisy"]
	costs= data["costs"]
	αrange = data["αrange"]
end

# ╔═╡ 0d9c3818-1fa1-11eb-2d69-bd2d53d4bf65
plot(αrange[1000:9000], costs[1000:9000],legend=:bottomright,label="lena")

# ╔═╡ 656d23c2-1fa1-11eb-3b50-6d6eed3b00dc
md" To perform this optimal parameter search we will make use a nonsmooth trust region algorithm provided in VariationalImaging. This parameter requires a function that return a tuple containing the parameter value, the cost function and the gradient at a particular value. In this first experiment let us consider a gradient approximation using finite differences."

# ╔═╡ 9d090074-1fa1-11eb-2aab-89fa5e07012f
function tr_function(α, img, noisy)
	u = TVl₂Denoising(noisy,α)
	u_ = TVl₂Denoising(noisy,α+1e-8)
	c = 0.5*norm₂²(u-img)
	c_ = 0.5*norm₂²(u_-img)
	g = (c_-c)/1e-8
	return (u=u,c=c,g=g)
end

# ╔═╡ e5dd585e-1fa1-11eb-0dec-c5dca91a7cbf
solver = NSTR(;verbose=true,freq=10,maxit=100)

# ╔═╡ 7b6ee326-204f-11eb-3faf-b311902be290
M,N = size(img)

# ╔═╡ 9b7549bc-204f-11eb-3a84-8987324d9d14
x₀ = 0.1/(M*N) # Inital value heuristic

# ╔═╡ f3cfd7c0-1fa1-11eb-30b5-9faaa4154f18
opt,fx,res,iters = solver(x->tr_function(x,img,noisy),x₀)

# ╔═╡ c53edac0-1fb3-11eb-2a88-276a0d7eb702
Gray.([img noisy adjust_histogram(fx.u,LinearStretching())])

# ╔═╡ a2946062-1fb3-11eb-1ead-3bac6978cde2
[assess_ssim(img,noisy) assess_ssim(img,adjust_histogram(fx.u,LinearStretching()))]

# ╔═╡ 3bf1ae34-2047-11eb-3314-2b46b6dad513
md"""
## Optimality System Gradient
Now, let us use the optimality system obtained for the bilevel problem described in De Los Reyes, Villacís [x]. This optimality system makes the assumption of the biactive set being empty, given that this cost function is Bouligand differentiable, it makes use of the characterization of an element of its Bouligand subdifferential to set it as the gradient of the function.

``\phi''(u) + diag(\hat{\alpha})\mathbb{K}^\top q = 0 ``

``<q_j,\hat{\alpha}_j(\mathbb{K} u)_j>-\hat{\alpha}_j\|(\mathbb{K} u)_j\|=0,\;\forall j=1,\dots,n,``

``\|q_j\|\le 1,\;\forall j=1,\dots,n,``

``\phi''(u)\hat{p} + diag(\hat{\alpha})\mathbb{K}^\top \nu = J_u(u,\hat{\alpha}),``

``\nu_j - T_j(\mathbb{K} \hat{p})_j = 0,\;\forall j\in\mathcal{I}(u),``

``(\mathbb{K} \hat{p})_j=0,\;\forall j\in\mathcal{A}(u),``

``J_\alpha(u,\hat{\alpha}) -diag(\hat{p})\mathbb{K}^\top \gamma=0.``

"""

# ╔═╡ 51e28cde-2436-11eb-0bc6-b5a6f526e750
function createDivMatrix(n)
	Hx = spdiagm(-1=>-ones(n-1),1=>ones(n-1))
	Gx = kron(spdiagm(0=>ones(n)),Hx)
	Gy = spdiagm(-n=>-ones(n^2-n),n=>ones(n^2-n))
	return [Gy;Gx]
end

# ╔═╡ 64345a82-2438-11eb-19fc-a5bcd3c6207b
function prodesc(q,p)
	n=Int(size(q,1)/2)
	q1=q[1:n]
	q2=q[n+1:2*n]

	p1=p[1:n]
	p2=p[n+1:2*n]

	return [spdiagm(0=>p1.*q1) spdiagm(0=>p2.*q1);
		  spdiagm(0=>p1.*q2) spdiagm(0=>p2.*q2)]
end

# ╔═╡ f2c7c5e4-2470-11eb-39e2-b12eaae13730
function xi(x)
	y = zeros(size(x))
	n = Int(size(x,1)/2)
	for i = 1:n
		y[i] = norm([x[i];x[i+n]],2)
		y[i+n] = norm([x[i];x[i+n]],2)
	end
	return y
end		

# ╔═╡ 2d6b340e-2435-11eb-193b-8f9600fcf433
function gradient(α,u,ū)
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	
	# Generate centered gradient matrix
	G = createDivMatrix(n)
	Gu = G*u[:]
	nGu = xi(Gu)
	act = nGu .< 1e-12
	inact = 1 .- act
	Act = spdiagm(0=>act)
	Inact = spdiagm(0=>inact)
	
	# Vector with grad norm in inactive components and one in the active
	den = Inact*nGu+act
	Den = spdiagm(0=>1 ./den)
	
	# prod KuKuᵗ/norm³
	prodKuKu = prodesc(Gu ./den.^3,Gu)
	
	Adj = [spdiagm(0=>ones(n^2)) α*G';
			Act*G+Inact*(prodKuKu-Den)*G Inact+sqrt(eps())*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	@time mult = Adj\Track
	p = @view mult[1:n^2]
	return -p'*(G'*Inact*Den*Gu)
end

# ╔═╡ 39135c0e-2370-11eb-30d2-99637f8643f8
function T(Ku,invnKu,v)
	invnKu³ = invnKu.^3
	a = batchmul(v,invnKu)
	b = zeros(size(v))
	m = Ku[1,:,:] .* Ku[2,:,:]
	@views b[1,:,:] = Ku[1,:,:].^2*v[1,:,:] + m .* v[2,:,:]
	@views b[2,:,:] = Ku[2,:,:].^2*v[2,:,:] + m .* v[1,:,:]
	c = batchmul(b,invnKu³)
	return a# - c
end

# ╔═╡ ec6d509e-2392-11eb-1ac3-05879bb31a36
function S2(x,y,Ku,invnKu,Inact,Act;ϵ=0.01)
	Kx = zeros(size(y))
	∇₂!(Kx,x)
	TKx = T(Ku,invnKu,Kx)
	return batchmul(Kx,Act) - batchmul(TKx,Inact) + batchmul(y,(Inact + ϵ*Act))
end

# ╔═╡ f883545c-2052-11eb-3ab3-27e0e9627e58
function gradient_uzawa(α,u,ū;maxiter=10) #Uzawa for saddle point problems
	Ku = zeros(2,size(u)...)
	Kuū = zeros(2,size(u)...)
	∇₂!(Ku,u)
	∇₂!(Kuū,u-ū)
	nKu = sqrt.(Ku[1,:,:].^2 + Ku[2,:,:].^2)
	Inact = nKu .> 1e-7
	Act = 1 .- Inact
	invKu = 1 ./(Inact .*nKu .+ Act)
	batchmul!(Ku,Ku,Inact)
	KtKu = zeros(size(u))
	∇₂ᵀ!(KtKu,Ku)
	
	y = zeros(2,size(u)...)
	x = zeros(size(u))
	Δx = zeros(size(u))
	for i = 1:maxiter
		∇₂ᵀ!(Δx,y)	
		x = ū-u-α*Δx
		y += 0.00001*S2(x,y,Ku,invKu,Inact,Act)
	end
	
	return x[:]'*(Inact .*KtKu)[:]
end

# ╔═╡ 7c46e6aa-2369-11eb-1892-ebb1d305b956
#Schur complement operator
function S(α,x,Ku,invnKu,Inact,Act;ϵ=10.1)
	Kᵗx = zeros(size(x,2),size(x,3))
	KKᵗx = zeros(size(x))
	∇₂ᵀ!(Kᵗx,x)
	∇₂!(KKᵗx,Kᵗx)
	TKKᵗx = T(Ku,invnKu,KKᵗx)
	return α*batchmul(KKᵗx,Act) - α*batchmul(TKKᵗx,Inact) - batchmul(x,(Inact + ϵ*Act))
end

# ╔═╡ 19af755e-2053-11eb-0fe8-9f10affcdbb2
function os_tr_function(α,img,noisy)
	u = TVl₂Denoising(noisy,α)
	c = 0.5*norm₂²(u-img)
	g = gradient(α,u,img)
	println("Gradient at $α = $g")
	return (u=u,c=c,g=g)
end

# ╔═╡ 4da59ad0-2053-11eb-3cf9-c74b460c7a9d
αtest = 20.1

# ╔═╡ 682c9bba-2053-11eb-1031-a59210db816b
fd_approx = tr_function(αtest,img,noisy)

# ╔═╡ b70ef3c2-2053-11eb-3a6e-0fc254c1b5cd
os_approx = os_tr_function(αtest,img,noisy)

# ╔═╡ 4a26c068-2077-11eb-2946-21d3654fcca5
opt_os,fx_os,res_os,iters_os = solver(x->os_tr_function(x,img,noisy),x₀)

# ╔═╡ 86f6b782-2077-11eb-1c8b-2ffbd1660c03
Gray.([img noisy adjust_histogram(fx_os.u,LinearStretching())])

# ╔═╡ a7f9929c-2077-11eb-30df-5de48fd58ace
[assess_ssim(img,noisy) assess_ssim(img,adjust_histogram(fx_os.u,LinearStretching()))]

# ╔═╡ Cell order:
# ╟─32b60ce0-1fa0-11eb-29d3-1d3df6525a7d
# ╠═8da2c26a-1fa0-11eb-2f91-216fa29571de
# ╠═9751886e-1fa0-11eb-0c9e-ad8f7c66d3b9
# ╠═a55a8e12-1fa0-11eb-2438-798a1c4ac0ef
# ╠═b35dd92e-1fa0-11eb-15bb-9723015e8379
# ╠═0d9c3818-1fa1-11eb-2d69-bd2d53d4bf65
# ╟─656d23c2-1fa1-11eb-3b50-6d6eed3b00dc
# ╠═9d090074-1fa1-11eb-2aab-89fa5e07012f
# ╠═e5dd585e-1fa1-11eb-0dec-c5dca91a7cbf
# ╠═7b6ee326-204f-11eb-3faf-b311902be290
# ╠═9b7549bc-204f-11eb-3a84-8987324d9d14
# ╠═f3cfd7c0-1fa1-11eb-30b5-9faaa4154f18
# ╠═e16f734c-1fab-11eb-3182-55a956044637
# ╠═c53edac0-1fb3-11eb-2a88-276a0d7eb702
# ╠═a2946062-1fb3-11eb-1ead-3bac6978cde2
# ╟─3bf1ae34-2047-11eb-3314-2b46b6dad513
# ╠═678d847c-2057-11eb-0609-e517d1990028
# ╠═ec6d509e-2392-11eb-1ac3-05879bb31a36
# ╠═f883545c-2052-11eb-3ab3-27e0e9627e58
# ╠═be3b79a2-22a8-11eb-24f6-2f469ff92a30
# ╠═51e28cde-2436-11eb-0bc6-b5a6f526e750
# ╠═64345a82-2438-11eb-19fc-a5bcd3c6207b
# ╠═f2c7c5e4-2470-11eb-39e2-b12eaae13730
# ╠═2d6b340e-2435-11eb-193b-8f9600fcf433
# ╠═39135c0e-2370-11eb-30d2-99637f8643f8
# ╠═7c46e6aa-2369-11eb-1892-ebb1d305b956
# ╠═19af755e-2053-11eb-0fe8-9f10affcdbb2
# ╠═4da59ad0-2053-11eb-3cf9-c74b460c7a9d
# ╠═682c9bba-2053-11eb-1031-a59210db816b
# ╠═b70ef3c2-2053-11eb-3a6e-0fc254c1b5cd
# ╠═4a26c068-2077-11eb-2946-21d3654fcca5
# ╠═86f6b782-2077-11eb-1c8b-2ffbd1660c03
# ╠═a7f9929c-2077-11eb-30df-5de48fd58ace
