### A Pluto.jl notebook ###
# v0.12.7

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
plot(αrange[1:900], costs[1:900],legend=:bottomright,label="lena")

# ╔═╡ 656d23c2-1fa1-11eb-3b50-6d6eed3b00dc
md" To perform this optimal parameter search we will make use a nonsmooth trust region algorithm provided in VariationalImaging. This parameter requires a function that return a tuple containing the parameter value, the cost function and the gradient at a particular value. In this first experiment let us consider a gradient approximation using finite differences."

# ╔═╡ 9d090074-1fa1-11eb-2aab-89fa5e07012f
function tr_function(α, img, noisy)
	u = TVl₂Denoising(noisy,α)
	u_ = TVl₂Denoising(noisy,α+1e-6)
	c = 0.5*norm₂²(u-img)
	c_ = 0.5*norm₂²(u_-img)
	g = (c_-c)/1e-6
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

# ╔═╡ f883545c-2052-11eb-3ab3-27e0e9627e58
function gradient(α,u,ū;maxiter=10) #Uzawa for saddle point problems
	Ku = zeros(2,size(u)...)
	∇₂!(Ku,u)
	nKu = sqrt.(Ku[1,:,:].^2 + Ku[2,:,:].^2)
	Inact = nKu .> 1e-7
	Inact2 = zeros(2,size(u)...)
	Inact2[1,:,:] = Inact
	Inact2[2,:,:] = Inact
	Act = 1 .- Inact
	invKu = 1 ./(Inact .*nKu .+ Act)
	Ku[1,:,:] .*= Inact .*invKu
	Ku[2,:,:] .*= Inact .*invKu
	KKu = zeros(size(u))
	∇₂ᵀ!(KKu,Ku)
	
	y = zeros(2,size(u)...)
	x = zeros(size(u))
	Δy = zeros(2,size(u)...)
	Δx = zeros(size(u))
	for i = 1:maxiter
		∇₂ᵀ!(Δx,y)	
		x = ū-u-α*Δx
		∇₂!(Δy,x)
		y = y + (1/(5*α))*(Δy- Inact2 .*y)
	end
	
	return x[:]'*(Inact .*KKu)[:]
end

# ╔═╡ 19af755e-2053-11eb-0fe8-9f10affcdbb2
function os_tr_function(α,img,noisy)
	u = TVl₂Denoising(noisy,α)
	c = 0.5*norm₂²(u-img)
	g = gradient(α,u,img;maxiter=100)
	return (u=u,c=c,g=g)
end

# ╔═╡ 4da59ad0-2053-11eb-3cf9-c74b460c7a9d
αtest = 10.5

# ╔═╡ 682c9bba-2053-11eb-1031-a59210db816b
fd_approx = tr_function(αtest,img,noisy)

# ╔═╡ b70ef3c2-2053-11eb-3a6e-0fc254c1b5cd
os_approx = os_tr_function(αtest,img,noisy)

# ╔═╡ 4a26c068-2077-11eb-2946-21d3654fcca5
opt_os,fx_os,res_os,iters_os = solver(x->os_tr_function(x,img,noisy),x₀)

# ╔═╡ 86f6b782-2077-11eb-1c8b-2ffbd1660c03
#Gray.([img noisy adjust_histogram(fx_os.u,LinearStretching())])

# ╔═╡ a7f9929c-2077-11eb-30df-5de48fd58ace
#[assess_ssim(img,noisy) assess_ssim(img,adjust_histogram(fx_os.u,LinearStretching()))]

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
# ╠═f883545c-2052-11eb-3ab3-27e0e9627e58
# ╠═19af755e-2053-11eb-0fe8-9f10affcdbb2
# ╠═4da59ad0-2053-11eb-3cf9-c74b460c7a9d
# ╠═682c9bba-2053-11eb-1031-a59210db816b
# ╠═b70ef3c2-2053-11eb-3a6e-0fc254c1b5cd
# ╠═4a26c068-2077-11eb-2946-21d3654fcca5
# ╠═86f6b782-2077-11eb-1c8b-2ffbd1660c03
# ╠═a7f9929c-2077-11eb-30df-5de48fd58ace
