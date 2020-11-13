### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ b2f82766-25be-11eb-2514-599e3c813adf
using Revise

# ╔═╡ b6344920-25be-11eb-2a7c-5f51f03376d1
using AlgTools.Util, JLD2, FileIO, Plots, SparseArrays

# ╔═╡ c3ff759e-25be-11eb-119d-0331abe0194e
using ImageQualityIndexes, ImageContrastAdjustment, ImageShow

# ╔═╡ c6e50d3e-25be-11eb-06e8-2b9f3ef51fbb
using VariationalImaging

# ╔═╡ 5feb9ab2-25be-11eb-2fd7-8fa9c3c2b70d
md"""
# Patch Dependent TV-l₂ Parameter Learning
In this notebook we will find the optimal regularizarion parameters for image denoising using a scale dependent parameter ``\alpha\in\mathbb{R}^2``
"""

# ╔═╡ d084b088-25be-11eb-1316-8f10a06655f0
md"Let us first load a previously calculated cost function for a ``\mathbb{R}^2``parameter on a noisy lena image"

# ╔═╡ cc442f6c-25be-11eb-2a02-374df9b02dcc
begin
	data = FileIO.load("pd_lena_cost_function.jld2")
	img = data["img"]
	noisy = data["noisy"]
	costs= data["costs"]
	α₁ = data["α₁"]
	α₂ = data["α₂"]
end

# ╔═╡ 5f558a80-25bf-11eb-22ac-83485b692a75
contour(α₁,α₂,costs)

# ╔═╡ 8a629dac-25c0-11eb-2a72-bfaee59c769b
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
	
	Adj = [spdiagm(0=>ones(n^2)) spdiagm(0=>α[:])*G';
			Act*G+Inact*(prodKuKu-Den)*G Inact+sqrt(eps())*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	@time mult = Adj\Track
	p = @view mult[1:n^2]
	return -spdiagm(0=>p)*(G'*Inact*Den*Gu)
end

# ╔═╡ 1bda8746-25c5-11eb-26d3-0d141e697b53
function learning_function(α,img,noisy)
	noisy = Float64.(Gray{Float64}.(noisy))
	img = Float64.(Gray{Float64}.(img))
	p = PatchOperator(α,img)
	pα = patch(p,α)
	u = TVl₂Denoising(noisy,pα)
	cost = 0.5*norm₂²(u-img)
	grad = gradient(pα,u,img)
	g = reverse_patch(p,reshape(grad,size(img)...))
	return (u=u,c=cost,g=reshape(g,size(α)))
end

# ╔═╡ 9f0930e0-25c5-11eb-164a-992efc455559
αtest = [0.05 0.9]

# ╔═╡ 9754baa4-25c5-11eb-2b71-09c6253a4e30
test = learning_function(αtest,img,noisy)

# ╔═╡ 6c973c50-25c6-11eb-0b56-37af2dbff23c
Gray.([img noisy test.u])

# ╔═╡ dfcdcea6-25e8-11eb-06c2-2dade7cd090d
md"""
## Bilevel Parameter Learning
"""

# ╔═╡ f3c33a18-25e8-11eb-153c-2f60fac8dd20
M,N = size(img)

# ╔═╡ e9f69f70-25e8-11eb-1b04-9bbaf0796635
x₀ = [1/(M*N) 1/(M*N)]

# ╔═╡ 0b3edabc-25e9-11eb-031a-813b088008a8
solver = NSTR(;verbose=true,freq=1,maxit=100)

# ╔═╡ 241c1dce-25e9-11eb-254e-f1a9abe9896c
opt,fx,res,iters = solver(x->learning_function(x,img,noisy),x₀)

# ╔═╡ eb59e050-25ea-11eb-108a-17dc067c273c
Gray.([img noisy fx.u])

# ╔═╡ 9a3b96de-25fa-11eb-0d93-ad193a663207
[assess_ssim(img,noisy) assess_ssim(img,fx.u)]

# ╔═╡ 2965f16a-25f3-11eb-2350-93bf0638968d
begin
	contour(α₁,α₂,costs;levels=10, fill=true)
	scatter!(opt[2]:opt[2],opt[1]:opt[1], label="optimal parameter")
end

# ╔═╡ Cell order:
# ╟─5feb9ab2-25be-11eb-2fd7-8fa9c3c2b70d
# ╠═b2f82766-25be-11eb-2514-599e3c813adf
# ╠═b6344920-25be-11eb-2a7c-5f51f03376d1
# ╠═c3ff759e-25be-11eb-119d-0331abe0194e
# ╠═c6e50d3e-25be-11eb-06e8-2b9f3ef51fbb
# ╟─d084b088-25be-11eb-1316-8f10a06655f0
# ╠═cc442f6c-25be-11eb-2a02-374df9b02dcc
# ╠═5f558a80-25bf-11eb-22ac-83485b692a75
# ╠═8a629dac-25c0-11eb-2a72-bfaee59c769b
# ╠═1bda8746-25c5-11eb-26d3-0d141e697b53
# ╠═9f0930e0-25c5-11eb-164a-992efc455559
# ╠═9754baa4-25c5-11eb-2b71-09c6253a4e30
# ╠═6c973c50-25c6-11eb-0b56-37af2dbff23c
# ╟─dfcdcea6-25e8-11eb-06c2-2dade7cd090d
# ╠═f3c33a18-25e8-11eb-153c-2f60fac8dd20
# ╠═e9f69f70-25e8-11eb-1b04-9bbaf0796635
# ╠═0b3edabc-25e9-11eb-031a-813b088008a8
# ╠═241c1dce-25e9-11eb-254e-f1a9abe9896c
# ╠═eb59e050-25ea-11eb-108a-17dc067c273c
# ╠═9a3b96de-25fa-11eb-0d93-ad193a663207
# ╠═2965f16a-25f3-11eb-2350-93bf0638968d
