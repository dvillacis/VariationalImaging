### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 44cd2470-1ac5-11eb-15f5-5f9c04ac4940
using Revise

# ╔═╡ 4c79f950-1ac5-11eb-3286-1969e5c97f4c
using VariationalImaging, Noise, LinearAlgebra, Images, Plots, ImageQualityIndexes, ImageContrastAdjustment

# ╔═╡ ae801326-1acc-11eb-3f53-3f001565dda7
using Krylov, LinearOperators

# ╔═╡ 652813b0-1ac5-11eb-0bf4-43c772cf0159
M,N = (40,40)

# ╔═╡ 58c574d2-1ac5-11eb-35fa-f7a4de2eb3c7
begin
	img = adjust_histogram!(0.2*LowerTriangular(ones(M,N))+0.6*UpperTriangular(ones(M,N)),LinearStretching());
	noisy = adjust_histogram!(add_gauss(img,0.1),LinearStretching());
end

# ╔═╡ 8b41ec24-1ac5-11eb-3248-29ce3a3e47de
Gray.([img noisy])

# ╔═╡ a4625de2-1ac5-11eb-1986-99702e22c626
α_range = 0.001:0.01:0.5

# ╔═╡ db354eba-1ac5-11eb-3403-ddebdb3df675
cost_fun = L2UpperLevelCost(img[:])

# ╔═╡ 677c4550-1edf-11eb-3bbc-51347acc567e
recon = [TVDenoising(noisy,x;maxit=2000) for x ∈ α_range];

# ╔═╡ a810e828-1edf-11eb-32cc-5d14abd126c4
[Gray.([recon[x] img]) for x ∈ 1:1:500]

# ╔═╡ a614f23c-1eec-11eb-24d9-7122174bf3a7
Gray.([recon[497].-img recon[498].-img recon[499].-img])

# ╔═╡ d4000366-1eed-11eb-0d36-b9d3414eeb22
[norm(recon[497].-img)^2 norm(recon[498].-img)^2 norm(recon[499].-img)^2]

# ╔═╡ 84779046-1eec-11eb-0306-cbf8af73bc7c
[0.5*norm(recon[497] .-img)^2 0.5*norm(recon[498] .-img)^2 0.5*norm(recon[499] .-img)^2]

# ╔═╡ b0c4cc82-1ac5-11eb-330a-c59308f45ba0
costs = [0.5*norm(recon[i] .-img)^2 for i ∈ 1:1:500]

# ╔═╡ 5fbb7844-1ac6-11eb-3fdc-e9ccde809c84
plot(costs)#plot(α_range,costs)

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
α_range2 = 0.0001:0.01:1.4

# ╔═╡ bfc2a394-1ed5-11eb-0e37-891d2ad8b5a9
function xi(x)
	n = Int(length(x)/2)
	y = zeros(n)
	for i = 1:n
		y[i] = sqrt(x[i]^2+x[i+n]^2)
	end
	return y
end

# ╔═╡ 4ebd4928-1acb-11eb-2323-e5851351ecb6
function tr_fun(x)
	m,n = size(img)
    L = Gradient(m,n)
    f = L2DataTerm(noisy[:])
    g = NormL21(x*ones(m,n))
    s = PDHG(;maxit=200, verbose=false)
    u,r,i = s(f.b,L*f.b,f,g,L)
	c = 0.5*norm(img .-reshape(u,M,N))^2
	Lu = L*u
	nLu = xi(Lu)
	inact = nLu .> 1e-8
	act = 1 .-inact
	Inact = Diagonal([inact;inact])
	Act = Diagonal([act;act])
	A = [opEye(m*n) x*L';(Act-Inact)*L (Inact+0.001*Act)*opEye(2*m*n) ]
	#println(Matrix(Inact))
	track = [img[:]-u[:];zeros(2*m*n)]
	#adj = Matrix(A)\track
	adj,status = cg(A,track)
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
	costs2 = [pairs[i].c for i=1:length(α_range2)];
	grads2 = [pairs[i].g for i=1:length(α_range2)];
end

# ╔═╡ 1a56ca32-1ad1-11eb-0b09-6b3c1463f06b
begin
	plot(α_range2,costs2./maximum(costs2),label="cost",legend=:bottomright)
	plot!(α_range2,grads2./maximum(abs.(grads2)),label="grad",legend=:bottomright)
end

# ╔═╡ 61666324-1ac7-11eb-0c6a-d31634ad3028
solver = NSTR(;verbose=true,freq=1,maxit=100)

# ╔═╡ 6b626a08-1ac7-11eb-02b4-f9be8ba77a4b
#x,res,it = solver(tr_fun,0.001;Δ₀=0.01)

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
# ╠═677c4550-1edf-11eb-3bbc-51347acc567e
# ╠═a810e828-1edf-11eb-32cc-5d14abd126c4
# ╠═a614f23c-1eec-11eb-24d9-7122174bf3a7
# ╠═d4000366-1eed-11eb-0d36-b9d3414eeb22
# ╠═84779046-1eec-11eb-0306-cbf8af73bc7c
# ╠═b0c4cc82-1ac5-11eb-330a-c59308f45ba0
# ╠═5fbb7844-1ac6-11eb-3fdc-e9ccde809c84
# ╟─078a1dee-1ac7-11eb-26d8-d160542d07f8
# ╠═1e9b20e8-1ac7-11eb-258f-3f63748f0d1c
# ╠═c80cc8ba-1acf-11eb-160b-5f0f32e2a30f
# ╠═ae801326-1acc-11eb-3f53-3f001565dda7
# ╠═89e7bd66-1ad1-11eb-1884-51172172a26c
# ╠═bfc2a394-1ed5-11eb-0e37-891d2ad8b5a9
# ╠═4ebd4928-1acb-11eb-2323-e5851351ecb6
# ╠═0c883008-1ad1-11eb-17dd-bfe777a68da7
# ╠═249f74fe-1af8-11eb-3469-53fb7e39095a
# ╠═1a56ca32-1ad1-11eb-0b09-6b3c1463f06b
# ╠═61666324-1ac7-11eb-0c6a-d31634ad3028
# ╠═6b626a08-1ac7-11eb-02b4-f9be8ba77a4b
# ╠═43e12826-1ac9-11eb-2472-0d3994e11d98
# ╠═4c93a5cc-1ac9-11eb-08af-1db439c68cdc
# ╠═552e0406-1aca-11eb-246e-03e88f2af91e
