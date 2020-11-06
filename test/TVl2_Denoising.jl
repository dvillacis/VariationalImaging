using VariationalImaging
using Test

using Printf
using FileIO
using ColorTypes: Gray
import ColorVectorSpace
import TestImages
using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise
using LinearAlgebra
using ImageView

# Parameters
const save_prefix="denoise_result_"

visualise = false

const default_params = (
    α = 0.1,
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    # FISTA
    #τ₀ = 0.9,
    ρ = 0,
    accel = true,
    noise_level = 0.1,
    verbose_iter = 10,
    maxiter = 1000,
    save_results = false,
    image_name = "lena_gray",
    save_iterations = false
)

# Parameters for this experiment
params = default_params
params = params ⬿ (save_prefix = save_prefix * params.image_name,)

# Load image and add noise
b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
println(size(b))
b_noisy = b .+ params.noise_level.*randn(size(b)...)

# Launch (background) visualiser
st, iterate = initialise_visualisation(visualise)

# Run algorithm
x, y, st = denoise_pdps(b_noisy; iterate=iterate, params=params)

if params.save_results
    perffile = params.save_prefix * ".txt"
    println("Saving " * perffile)
    write_log(perffile, st.log, "# params = $(params)\n")
    fn = (t, ext) -> "$(params.save_prefix)_$(t).$(ext)"
    save(File(format"PNG", fn("true", "png")), grayimg(b))
    save(File(format"PNG", fn("data", "png")), grayimg(b_noisy))
    save(File(format"PNG", fn("reco", "png")), grayimg(x))
end

# Exit background visualiser
finalise_visualisation(st)

@testset "TVl2_Denoising.jl" begin
    # Write your tests here.
    @test norm(x-b) < 1e-3
end