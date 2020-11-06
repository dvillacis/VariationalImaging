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
using ImageQualityIndexes, ImageContrastAdjustment

# ╔═╡ 32b60ce0-1fa0-11eb-29d3-1d3df6525a7d
md"""
# TV-l₂ Scalar Parameter Learning
In this notebook we will make use of a trust-region algorithm for finding the optimal scalar regularization parameter for the TV-l₂ image denoising model. Fisrt let us recall the form of the cost function.
"""

# ╔═╡ b35dd92e-1fa0-11eb-15bb-9723015e8379
begin
	data = FileIO.load("tvl2_small_experiment.jld2")
	img = data["img"]
	noisy = data["noisy"]
	costs= data["costs"]
	αrange = data["αrange"]
end

# ╔═╡ 0d9c3818-1fa1-11eb-2d69-bd2d53d4bf65
plot(αrange[1:80], costs[1:80])

# ╔═╡ 656d23c2-1fa1-11eb-3b50-6d6eed3b00dc
md" To perform this optimal parameter search we will make use a nonsmooth trust region algorithm provided in VariationalImaging. This parameter requires a function that return a tuple containing the parameter value, the cost function and the gradient at a particular value."

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
solver = NSTR(;verbose=true,freq=10,maxit=1000)

# ╔═╡ f3cfd7c0-1fa1-11eb-30b5-9faaa4154f18
opt,fx,res,iters = solver(x->tr_function(x,img,noisy),0.00001)

# ╔═╡ c53edac0-1fb3-11eb-2a88-276a0d7eb702
Gray.([img noisy fx.u])

# ╔═╡ a2946062-1fb3-11eb-1ead-3bac6978cde2
[assess_ssim(img,noisy) assess_ssim(img,fx.u)]

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
# ╠═f3cfd7c0-1fa1-11eb-30b5-9faaa4154f18
# ╠═e16f734c-1fab-11eb-3182-55a956044637
# ╠═c53edac0-1fb3-11eb-2a88-276a0d7eb702
# ╠═a2946062-1fb3-11eb-1ead-3bac6978cde2
