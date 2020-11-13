### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 56f036c2-2526-11eb-2217-95d49872240a
using Revise

# ╔═╡ 5b482d2c-2526-11eb-23cb-2396ff4a8d47
using Images, TestImages, Noise, ImageContrastAdjustment, ImageQualityIndexes, LinearAlgebra, ImageShow, Plots

# ╔═╡ df8256be-254e-11eb-1adc-49d537d8f76a
using VariationalImaging

# ╔═╡ 27ad334e-2551-11eb-1208-b7f9c4b074bc
using AlgTools.Util

# ╔═╡ 31b2863a-255a-11eb-3ad7-25c11e8ffd6b
using FileIO, JLD2

# ╔═╡ c109c2a6-2525-11eb-2425-b784e711b843
md"""
# Patch Dependent TV-l₂ Cost Function
In this notebook we will explore the cost function for a patch dependen bilevel problem as described in De Los Reyes, Villacís [X]. For this experiment we will create a denoising for a patch of 2x1, e.g., a different parameter for each half of an image.
"""

# ╔═╡ cd8d6b66-252a-11eb-36d3-390d679114db
σ = 0.1

# ╔═╡ d5ec0fec-252a-11eb-1d0a-31738e321ec7
begin 
	img = adjust_histogram(testimage("lena_gray"),LinearStretching())
	noisy = adjust_histogram(add_gauss(img,σ),LinearStretching())
	Gray.([img noisy])
end

# ╔═╡ 173fafa6-252b-11eb-3a5a-b5dc166cd913
md"""
## The Patch Operator
We are now interested on creating an operator that maps a low dimensional vector of size n and outputs a matriz with the same dimension as img, but with the pattern described in the lower dimension matrix.
"""

# ╔═╡ 7b5e24e2-252b-11eb-1b66-3b2eea269a7a
function patch(A,img)
	M,N = size(img)
	R,S = size(A)
	t = ones(Int(M/R),Int(N/S))
	return kron(A,t)
end

# ╔═╡ cd222dd2-2530-11eb-2916-cb3a570868ca
A = [0.06 0.4;0.9 0.08]

# ╔═╡ d63a4904-2530-11eb-3d15-7fd7fe90c827
p = patch(A,img)

# ╔═╡ edeb72bc-2530-11eb-244e-d7bc958ab6f3
Gray.(p)

# ╔═╡ cc4be010-254e-11eb-0c39-fde08b2b8ce9
u = TVl₂Denoising(noisy,p)

# ╔═╡ cbf207a6-254f-11eb-1195-ad6faad002e9
Gray.([img noisy u])

# ╔═╡ 7c3d7d0c-2550-11eb-34e1-f581e964cf1a
md"""
## Plotting the cost function
Now, we restrict ourselves to a two dimensional parameter on the splitted image, let us see how the cost surface looks like.
"""

# ╔═╡ 0d809664-2551-11eb-12b0-d7a1753c1670
αrange = 0.001:0.01:0.5

# ╔═╡ 21867d8e-2553-11eb-1347-611f9a3a5c81
x= αrange; y= αrange;

# ╔═╡ 8747702c-2555-11eb-0e1a-31ed49854486
z = [0.5*norm₂²(img-TVl₂Denoising(noisy,patch([i j],img))) for i ∈ x,j ∈ y]

# ╔═╡ 26f18cfa-2553-11eb-39ab-13388faaba67
contour(x,y,z)

# ╔═╡ aceff5bc-2553-11eb-30d8-452eee5bcce7
contourf(x,y,z)

# ╔═╡ 3f3c2084-255a-11eb-39aa-8b28ea411db0
save("pd_lena_cost_function.jld2","α₁",x,"α₂",y,"costs",z)

# ╔═╡ Cell order:
# ╟─c109c2a6-2525-11eb-2425-b784e711b843
# ╠═56f036c2-2526-11eb-2217-95d49872240a
# ╠═5b482d2c-2526-11eb-23cb-2396ff4a8d47
# ╠═df8256be-254e-11eb-1adc-49d537d8f76a
# ╠═cd8d6b66-252a-11eb-36d3-390d679114db
# ╠═d5ec0fec-252a-11eb-1d0a-31738e321ec7
# ╟─173fafa6-252b-11eb-3a5a-b5dc166cd913
# ╠═7b5e24e2-252b-11eb-1b66-3b2eea269a7a
# ╠═cd222dd2-2530-11eb-2916-cb3a570868ca
# ╠═d63a4904-2530-11eb-3d15-7fd7fe90c827
# ╠═edeb72bc-2530-11eb-244e-d7bc958ab6f3
# ╠═cc4be010-254e-11eb-0c39-fde08b2b8ce9
# ╠═cbf207a6-254f-11eb-1195-ad6faad002e9
# ╟─7c3d7d0c-2550-11eb-34e1-f581e964cf1a
# ╠═27ad334e-2551-11eb-1208-b7f9c4b074bc
# ╠═0d809664-2551-11eb-12b0-d7a1753c1670
# ╠═21867d8e-2553-11eb-1347-611f9a3a5c81
# ╠═8747702c-2555-11eb-0e1a-31ed49854486
# ╠═26f18cfa-2553-11eb-39ab-13388faaba67
# ╠═aceff5bc-2553-11eb-30d8-452eee5bcce7
# ╠═31b2863a-255a-11eb-3ad7-25c11e8ffd6b
# ╠═3f3c2084-255a-11eb-39aa-8b28ea411db0
