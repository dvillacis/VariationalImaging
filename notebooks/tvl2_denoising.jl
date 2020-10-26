### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ e1f185d2-157a-11eb-1a63-2d8bd9c8aeae
using TestImages, Noise, VariationalImaging, Images

# ╔═╡ 84c3db12-157c-11eb-003d-5303c1e4e911
using LinearAlgebra

# ╔═╡ 89ad56e2-157e-11eb-020e-d96fa1908eb7
using Plots

# ╔═╡ 498f1c08-17a9-11eb-3787-b36a3634b39b
using Revise

# ╔═╡ 1e094fc0-157b-11eb-0383-178ff2d2c838
img = Float64.(testimage("lena_gray"))

# ╔═╡ 2b5e7db2-157b-11eb-3454-b5be783ec30f
noisy = add_gauss(img,0.1)

# ╔═╡ 35cf033e-157b-11eb-0064-a51fd7da1229
Gray.([img noisy])

# ╔═╡ 80590364-157b-11eb-0e9b-ade576c9c378
α = 0.1

# ╔═╡ 5b07fffc-157b-11eb-1166-3b0cab9cb1a0
tvl2 = TVDenoising(noisy,α;maxit=100);

# ╔═╡ 735bfd60-157b-11eb-1f2c-a779a7e3d583
Gray.([img noisy tvl2])

# ╔═╡ 8ce43d32-17ac-11eb-246e-a9ee9da94990
α_range = 0.01:0.01:0.5

# ╔═╡ e7b6d0c2-157b-11eb-1d49-bf7a5c09b960
costs = [0.5*norm(img[:]-TVDenoising(noisy,α;maxit=50,verbose=true)[:])^2 for α ∈ α_range]

# ╔═╡ 8de369e0-157e-11eb-1cb0-5b945868dbfc
plot(α_range,costs)

# ╔═╡ eb21b230-17ac-11eb-156d-296610a6b1ae
costs2 = [0.5*norm(img[:]-TVDenoising(noisy,α;maxit=200,verbose=true)[:])^2 for α ∈ α_range]

# ╔═╡ f54f8f52-17ac-11eb-36b6-4386b80a4d5b
plot!(α_range,costs2)

# ╔═╡ bd281558-17ad-11eb-0461-43b9e12b8b6a
costs3 = [0.5*norm(img[:]-TVDenoising(noisy,α;maxit=500,verbose=true)[:])^2 for α ∈ α_range]

# ╔═╡ c4ae4782-17ad-11eb-2beb-f9c644e9db8f
plot!(α_range,costs3)

# ╔═╡ Cell order:
# ╠═e1f185d2-157a-11eb-1a63-2d8bd9c8aeae
# ╠═1e094fc0-157b-11eb-0383-178ff2d2c838
# ╠═2b5e7db2-157b-11eb-3454-b5be783ec30f
# ╠═35cf033e-157b-11eb-0064-a51fd7da1229
# ╠═80590364-157b-11eb-0e9b-ade576c9c378
# ╠═5b07fffc-157b-11eb-1166-3b0cab9cb1a0
# ╠═735bfd60-157b-11eb-1f2c-a779a7e3d583
# ╠═84c3db12-157c-11eb-003d-5303c1e4e911
# ╠═8ce43d32-17ac-11eb-246e-a9ee9da94990
# ╠═e7b6d0c2-157b-11eb-1d49-bf7a5c09b960
# ╠═89ad56e2-157e-11eb-020e-d96fa1908eb7
# ╠═8de369e0-157e-11eb-1cb0-5b945868dbfc
# ╠═eb21b230-17ac-11eb-156d-296610a6b1ae
# ╠═f54f8f52-17ac-11eb-36b6-4386b80a4d5b
# ╠═bd281558-17ad-11eb-0461-43b9e12b8b6a
# ╠═c4ae4782-17ad-11eb-2beb-f9c644e9db8f
# ╠═498f1c08-17a9-11eb-3787-b36a3634b39b
