### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 44cd2470-1ac5-11eb-15f5-5f9c04ac4940
using Revise

# ╔═╡ 4c79f950-1ac5-11eb-3286-1969e5c97f4c
using VariationalImaging, Noise, LinearAlgebra, Images, Plots, ImageQualityIndexes

# ╔═╡ ae801326-1acc-11eb-3f53-3f001565dda7
using Krylov, LinearOperators

# ╔═╡ 652813b0-1ac5-11eb-0bf4-43c772cf0159
M,N = (50,50)

# ╔═╡ 58c574d2-1ac5-11eb-35fa-f7a4de2eb3c7
begin
	img = 0.2*LowerTriangular(ones(M,N))+0.6*UpperTriangular(ones(M,N));
	noisy = add_gauss(img,0.1);
end

# ╔═╡ 8b41ec24-1ac5-11eb-3248-29ce3a3e47de
Gray.([img noisy])

# ╔═╡ a4625de2-1ac5-11eb-1986-99702e22c626
α_range = 0.001:0.02:1.0

# ╔═╡ db354eba-1ac5-11eb-3403-ddebdb3df675
cost_fun = L2UpperLevelCost(img[:])

# ╔═╡ b0c4cc82-1ac5-11eb-330a-c59308f45ba0
costs = [cost_fun(TVDenoising(noisy,x;maxit=100)[:]) for x ∈ α_range]

# ╔═╡ 5fbb7844-1ac6-11eb-3fdc-e9ccde809c84
plot(α_range,costs)

# ╔═╡ 078a1dee-1ac7-11eb-26d8-d160542d07f8
md"## Nonsmooth Trust Region Algorithm"

# ╔═╡ 1e9b20e8-1ac7-11eb-258f-3f63748f0d1c
function tr_function(x)
	eps = 1e-8
	u = TVDenoising(noisy,x;maxit=100)
	u_ = TVDenoising(noisy,x+eps;maxit=100)
	c = cost_fun(u[:])
	c_ = cost_fun(u_[:])
	g = (c_-c)/eps
	return (u=x,c=c,g=g)
end

# ╔═╡ c80cc8ba-1acf-11eb-160b-5f0f32e2a30f
function gamma(x)
	y = zeros(size(x))
	n = Int(length(x)/2)
	for i = 1:n
		a = (sqrt(x[i]^2+x[1+n]^2))
		if a < 1e-5
			y[i]=0
			y[i+1]=0
		else
			y[i] = x[i]/a
			y[i+n]=x[i+n]/a
		end
	end
	return y
end

# ╔═╡ 89e7bd66-1ad1-11eb-1884-51172172a26c
α_range2 = 0.01:0.02:0.25

# ╔═╡ 4ebd4928-1acb-11eb-2323-e5851351ecb6
function tr_fun(x)
	cost_function = L2UpperLevelCost(img[:])
	m,n = size(img)
    L = Gradient(m,n)
    f = L2DataTerm(noisy[:])
    g = NormL21(x*ones(m,n))
    s = PDHG(;maxit=200, verbose=false)
    u,r,i = s(f.b,L*f.b,f,g,L)
	c = cost_function(u)
	Lu = L*u
	A = [opEye(m*n) x*L';-L opEye(2*m*n)]
	track = [img[:]-u[:];zeros(2*m*n)]
	adj,status = lsmr(A'*A,A'*track)
	if status.solved == false
		@error "$status"
	end
	adj = adj[1:m*n]
	g = adj'*(L'*gamma(Lu))
	return (u=x,c=c,g=g)
end

# ╔═╡ 0c883008-1ad1-11eb-17dd-bfe777a68da7
pairs = [tr_fun(x) for x ∈ α_range2]

# ╔═╡ 249f74fe-1af8-11eb-3469-53fb7e39095a
begin
	costs2 = [pairs[i].c for i=1:length(α_range2)]
	grads2 = [pairs[i].g for i=1:length(α_range2)]
end

# ╔═╡ 1a56ca32-1ad1-11eb-0b09-6b3c1463f06b
begin
	plot(α_range2,costs2)
	#plot!(α_range2,grads2,yaxis=:right)
end

# ╔═╡ 61666324-1ac7-11eb-0c6a-d31634ad3028
solver = NSTR(;verbose=true,freq=1,maxit=100)

# ╔═╡ 6b626a08-1ac7-11eb-02b4-f9be8ba77a4b
x,res,it = solver(tr_fun,0.05)

# ╔═╡ 43e12826-1ac9-11eb-2472-0d3994e11d98
#optimal = TVDenoising(noisy,x;maxit=100)

# ╔═╡ 4c93a5cc-1ac9-11eb-08af-1db439c68cdc
#Gray.([img noisy optimal])

# ╔═╡ 552e0406-1aca-11eb-246e-03e88f2af91e
#[assess_ssim(img,noisy) assess_ssim(img,optimal)]

# ╔═╡ Cell order:
# ╠═44cd2470-1ac5-11eb-15f5-5f9c04ac4940
# ╠═4c79f950-1ac5-11eb-3286-1969e5c97f4c
# ╠═652813b0-1ac5-11eb-0bf4-43c772cf0159
# ╠═58c574d2-1ac5-11eb-35fa-f7a4de2eb3c7
# ╠═8b41ec24-1ac5-11eb-3248-29ce3a3e47de
# ╠═a4625de2-1ac5-11eb-1986-99702e22c626
# ╠═db354eba-1ac5-11eb-3403-ddebdb3df675
# ╠═b0c4cc82-1ac5-11eb-330a-c59308f45ba0
# ╠═5fbb7844-1ac6-11eb-3fdc-e9ccde809c84
# ╟─078a1dee-1ac7-11eb-26d8-d160542d07f8
# ╠═1e9b20e8-1ac7-11eb-258f-3f63748f0d1c
# ╠═c80cc8ba-1acf-11eb-160b-5f0f32e2a30f
# ╠═ae801326-1acc-11eb-3f53-3f001565dda7
# ╠═89e7bd66-1ad1-11eb-1884-51172172a26c
# ╠═4ebd4928-1acb-11eb-2323-e5851351ecb6
# ╠═0c883008-1ad1-11eb-17dd-bfe777a68da7
# ╠═249f74fe-1af8-11eb-3469-53fb7e39095a
# ╠═1a56ca32-1ad1-11eb-0b09-6b3c1463f06b
# ╠═61666324-1ac7-11eb-0c6a-d31634ad3028
# ╠═6b626a08-1ac7-11eb-02b4-f9be8ba77a4b
# ╠═43e12826-1ac9-11eb-2472-0d3994e11d98
# ╠═4c93a5cc-1ac9-11eb-08af-1db439c68cdc
# ╠═552e0406-1aca-11eb-246e-03e88f2af91e
