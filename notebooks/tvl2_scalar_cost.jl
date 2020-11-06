### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 60a602d2-1f9a-11eb-3e76-3bfddb67250d
using Revise

# ╔═╡ 644da598-1f9a-11eb-0d7d-4935daf962e6
using Images, TestImages, Noise, ImageContrastAdjustment, ImageQualityIndexes, LinearAlgebra

# ╔═╡ 799dd4fc-1f9a-11eb-160e-d59a65db78af
using VariationalImaging

# ╔═╡ f752ec8a-1f9b-11eb-3c1c-010c13ac1514
using PlutoUI

# ╔═╡ ca8886b4-1f9c-11eb-2f95-51638405d027
using AlgTools.Util

# ╔═╡ 4b6892e2-1f9d-11eb-1fd1-3d4f72709409
using Plots

# ╔═╡ 8bacf894-1f9f-11eb-06c0-bb6bd0109cdb
using FileIO, JLD2

# ╔═╡ 1efbc8b2-1f9a-11eb-1299-f18a30fec5b5
md"# TV-l₂ Image Denoising Cost Function
In this notebook we will explore a squared l₂ cost function with a single image dataset." 

# ╔═╡ acbf7ad6-1f9a-11eb-35b4-e1504d3c1677
M,N = (64,64);

# ╔═╡ f4102e3a-1f9a-11eb-09e6-1d9cdaaa4701
σ = 0.2;

# ╔═╡ 9f78b8e2-1f9a-11eb-3868-8b5ac4b3336a
begin
	img = adjust_histogram(0.1*LowerTriangular(ones(M,N)) + 0.8*UpperTriangular(ones(M,N)),LinearStretching());
	noisy = adjust_histogram(add_gauss(img,σ),LinearStretching());
end

# ╔═╡ 95545014-1f9b-11eb-10c7-2d18a39c1928
Gray.([img noisy])

# ╔═╡ 98db3514-1f9a-11eb-23f8-27dcba909560
md"""
$(@bind α Slider(0.001:0.01:10.0))
"""

# ╔═╡ b9f0f758-1f9e-11eb-3ba0-c92c370f802d
α

# ╔═╡ c91da468-1f9b-11eb-09ce-9d37a6642b19
x = TVl₂Denoising(noisy,α);

# ╔═╡ dafac24c-1f9b-11eb-05d4-eb60467caa58
Gray.([img noisy x])

# ╔═╡ 9abb34de-1f9c-11eb-2c73-83b841dc4d5f
md" Let us now calculate the l₂ squared cost with respect to the original image"

# ╔═╡ adcc9e5c-1f9c-11eb-27e5-dd5f7fa29478
cost = 0.5*norm₂²(x-img)

# ╔═╡ 03e07320-1f9d-11eb-2429-afda9086066b
md" Evenmore, plotting this cost function..."

# ╔═╡ 306a0908-1f9d-11eb-0690-bbd8434a3930
αrange = 0.00001:0.01:9.0;

# ╔═╡ 1448249e-1f9d-11eb-367e-89155bae18df
costs = [0.5*norm₂²(TVl₂Denoising(noisy,i)-img) for i ∈ αrange]

# ╔═╡ 4ec289ca-1f9d-11eb-2076-3522224ac094
plot(αrange,costs)

# ╔═╡ 9e7becc0-1f9f-11eb-3f54-3d8e79a2e831
FileIO.save("tvl2_small_experiment.jld2","img",img,"noisy",noisy,"costs",costs,"αrange",αrange)

# ╔═╡ Cell order:
# ╟─1efbc8b2-1f9a-11eb-1299-f18a30fec5b5
# ╠═60a602d2-1f9a-11eb-3e76-3bfddb67250d
# ╠═644da598-1f9a-11eb-0d7d-4935daf962e6
# ╠═799dd4fc-1f9a-11eb-160e-d59a65db78af
# ╠═acbf7ad6-1f9a-11eb-35b4-e1504d3c1677
# ╠═f4102e3a-1f9a-11eb-09e6-1d9cdaaa4701
# ╠═9f78b8e2-1f9a-11eb-3868-8b5ac4b3336a
# ╠═95545014-1f9b-11eb-10c7-2d18a39c1928
# ╠═f752ec8a-1f9b-11eb-3c1c-010c13ac1514
# ╠═98db3514-1f9a-11eb-23f8-27dcba909560
# ╠═b9f0f758-1f9e-11eb-3ba0-c92c370f802d
# ╠═c91da468-1f9b-11eb-09ce-9d37a6642b19
# ╠═dafac24c-1f9b-11eb-05d4-eb60467caa58
# ╟─9abb34de-1f9c-11eb-2c73-83b841dc4d5f
# ╠═ca8886b4-1f9c-11eb-2f95-51638405d027
# ╠═adcc9e5c-1f9c-11eb-27e5-dd5f7fa29478
# ╟─03e07320-1f9d-11eb-2429-afda9086066b
# ╠═306a0908-1f9d-11eb-0690-bbd8434a3930
# ╠═1448249e-1f9d-11eb-367e-89155bae18df
# ╠═4b6892e2-1f9d-11eb-1fd1-3d4f72709409
# ╠═4ec289ca-1f9d-11eb-2076-3522224ac094
# ╠═8bacf894-1f9f-11eb-06c0-bb6bd0109cdb
# ╠═9e7becc0-1f9f-11eb-3f54-3d8e79a2e831
