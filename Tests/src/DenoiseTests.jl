
module DenoiseTests

# Our exports
export test_op_denoise, test_sd_op_denoise
export test_sumregs_denoise, test_sd_sumregs_denoise

# Dependencies
using Printf
using FileIO
using ColorTypes: Gray
# ColorVectorSpace is only needed to ensure that conversions
# between different ColorTypes are defined.
import ColorVectorSpace
import TestImages

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Visualise

using VariationalImaging.GradientOps
using VariationalImaging.OpDenoise
using VariationalImaging.SumRegsDenoise


# Parameters
const default_save_prefix="result_"

const default_params = (
    ρ = 0,
    noise_level = 0.1,
    verbose_iter = 50,
    maxiter = 1000,
    save_results = false,
    image_name = "lighthouse",
    save_iterations = false
)

const denoise_params = (
    α = 0.1,
    op = FwdGradientOp(),
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    accel = true,
)

const sd_denoise_params = (
    α = [0.09*ones(512,384) 0.8*ones(512,384)],
    op = FwdGradientOp(),
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    accel = true,
)

const sumregs_denoise_params = (
    α₁ = 0.02,
    α₂ = 0.03,
    α₃ = 0.05,
    op₁ = FwdGradientOp(),
    op₂ = BwdGradientOp(),
    op₃ = CenteredGradientOp(),
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    accel = true,
)

const sd_sumregs_denoise_params = (
    α₁ = [0.1*ones(512,256) 1e-10*ones(512,256) 1e-10*ones(512,256)],
    α₂ = [1e-10*ones(512,256) 0.1*ones(512,256) 1e-10*ones(512,256)],
    α₃ = [1e-10*ones(512,256) 1e-10*ones(512,256) 0.2*ones(512,256)],
    op₁ = FwdGradientOp(),
    op₂ = BwdGradientOp(),
    op₃ = CenteredGradientOp(),
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    accel = true,
)

function save_results(params, b, b_data, x, st)
    if params.save_results
        perffile = params.save_prefix * ".txt"
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params)\n")
        fn = (t, ext) -> "$(params.save_prefix)_$(t).$(ext)"
        save(File(format"PNG", fn("true", "png")), grayimg(b))
        save(File(format"PNG", fn("data", "png")), grayimg(b_data))
        save(File(format"PNG", fn("reco", "png")), grayimg(x))
    end
end

###############
# Denoise test
###############

function test_op_denoise(;
                      visualise=true,
                      save_prefix=default_save_prefix,
                      kwargs...)

    # Parameters for this experiment
    params = default_params ⬿ denoise_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * "denoise_" * params.image_name,)

    # Load image and add noise
    b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
    b_noisy = b .+ params.noise_level.*randn(size(b)...)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y, st = op_denoise_pdps(b_noisy; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

###############
# Scale Dependent Denoise test
###############

function test_sd_op_denoise(;
    visualise=true,
    save_prefix=default_save_prefix,
    kwargs...)

    # Parameters for this experiment
    params = default_params ⬿ sd_denoise_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * "denoise_sd_" * params.image_name,)

    # Load image and add noise
    b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
    b_noisy = b .+ params.noise_level.*randn(size(b)...)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y, st = op_denoise_pdps(b_noisy; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

###############
# Denoise test
###############

function test_sumregs_denoise(;
    visualise=true,
    save_prefix=default_save_prefix,
    kwargs...)

    # Parameters for this experiment
    params = default_params ⬿ sumregs_denoise_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * "denoise_sumregs_" * params.image_name,)

    # Load image and add noise
    b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
    b_noisy = b .+ params.noise_level.*randn(size(b)...)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y₁, y₂, y₃, st = sumregs_denoise_pdps(b_noisy; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

function test_sd_sumregs_denoise(;
    visualise=true,
    save_prefix=default_save_prefix,
    kwargs...)

    # Parameters for this experiment
    params = default_params ⬿ sd_sumregs_denoise_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * "denoise_sd_sumregs_" * params.image_name,)

    # Load image and add noise
    b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
    b_noisy = b .+ params.noise_level.*randn(size(b)...)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y₁, y₂, y₃, st = sumregs_denoise_pdps(b_noisy; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

end # Module