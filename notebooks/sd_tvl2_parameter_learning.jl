### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 502d5262-25f7-11eb-1e30-4343658abf14
using Revise

# ╔═╡ 5406a1d6-25f7-11eb-222f-793e1be08760
using TestImages, Noise, ImageShow, Images, ImageContrastAdjustment, ImageQualityIndexes, SparseArrays

# ╔═╡ 27ef4d9a-25f8-11eb-0228-d70b9ea4370b
using VariationalImaging, AlgTools.Util

# ╔═╡ 823153fe-25f6-11eb-0b34-25e3d9d21145
md"""
# Scale Dependent TV-l₂ Parameter Learning
In this notebook we will find the optimal scale dependent parameter for a single image dataset based on the lena image.
"""

# ╔═╡ 67d21eac-25f7-11eb-3335-e556beefa7af
begin
	img = adjust_histogram(testimage("lena_gray"),LinearStretching())
	noisy = adjust_histogram(add_gauss(img,0.1),LinearStretching())
	M,N = size(img)
	Gray.([img noisy])
end

# ╔═╡ 9b51b5aa-25f7-11eb-1a1e-31f1c2d3498b
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

# ╔═╡ a2dea8da-25f7-11eb-174b-29f0a1dee501
function learning_function(α,img,noisy)
	noisy = Float64.(Gray{Float64}.(noisy))
	img = Float64.(Gray{Float64}.(img))
	u = TVl₂Denoising(noisy,α)
	cost = 0.5*norm₂²(u-img)
	grad = gradient(α,u,img)
	return (u=u,c=cost,g=grad)
end

# ╔═╡ bbc63988-25f7-11eb-2de0-9d1fcaa6af76
x₀ =  0.01*ones(size(img))#1/(M*N)*ones(size(img))

# ╔═╡ 171577ec-25f8-11eb-3210-6baf45fe06d3
solver = NSTR(;verbose=true,freq=1,maxit=100)

# ╔═╡ 3c4268ae-25f8-11eb-3940-b73f7da769a9
opt,fx,res,iters = solver(x->learning_function(x,img,noisy),x₀;Δ₀=1.5)

# ╔═╡ a91a6f34-25f9-11eb-17d7-696c6fdf88e4
Gray.([img noisy fx.u opt./maximum(opt)])

# ╔═╡ 80487280-25fa-11eb-3de7-45f7226b6df3
[assess_ssim(img,noisy) assess_ssim(img,fx.u)]

# ╔═╡ c786abd0-25fa-11eb-26fb-b33929594220
sum(opt .< 0)

# ╔═╡ Cell order:
# ╟─823153fe-25f6-11eb-0b34-25e3d9d21145
# ╠═502d5262-25f7-11eb-1e30-4343658abf14
# ╠═5406a1d6-25f7-11eb-222f-793e1be08760
# ╠═27ef4d9a-25f8-11eb-0228-d70b9ea4370b
# ╠═67d21eac-25f7-11eb-3335-e556beefa7af
# ╠═9b51b5aa-25f7-11eb-1a1e-31f1c2d3498b
# ╠═a2dea8da-25f7-11eb-174b-29f0a1dee501
# ╠═bbc63988-25f7-11eb-2de0-9d1fcaa6af76
# ╠═171577ec-25f8-11eb-3210-6baf45fe06d3
# ╠═3c4268ae-25f8-11eb-3940-b73f7da769a9
# ╠═a91a6f34-25f9-11eb-17d7-696c6fdf88e4
# ╠═80487280-25fa-11eb-3de7-45f7226b6df3
# ╠═c786abd0-25fa-11eb-26fb-b33929594220
