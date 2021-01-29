##################
# Denoise testing
##################

module Tests

# Our exports
export test_op_denoise, test_sd_op_denoise, test_sumregs_denoise

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
    verbose_iter = 10,
    maxiter = 1000,
    save_results = true,
    image_name = "lighthouse",
    save_iterations = false
)

const denoise_params = (
    α = 0.1,
    # PDPS
    τ₀ = 5,
    σ₀ = 0.99/5,
    accel = true,
)

# const sd_denoise_params = (
#     α = 0.1*LowerTriangular(ones(512,512))+1e-5*UpperTriangular(ones(512,512)),
#     # PDPS
#     τ₀ = 5,
#     σ₀ = 0.99/5,
#     accel = true,
# )

const sumregs_denoise_params = (
    α₁ = 0.02,
    α₂ = 0.03,
    α₃ = 0.05,
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

    # Define Linear operator
    #op = CenteredGradientOp()
    #op = BwdGradientOp()
    op = FwdGradientOp()

    # Run algorithm
    x, y, st = op_denoise_pdps(b_noisy,op; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

###############
# Scale Dependent Denoise test
###############

# function test_sd_op_denoise(;
#     visualise=true,
#     save_prefix=default_save_prefix,
#     kwargs...)

#     # Parameters for this experiment
#     params = default_params ⬿ sd_denoise_params ⬿ kwargs
#     params = params ⬿ (save_prefix = save_prefix * "denoise_sd_" * params.image_name,)

#     # Load image and add noise
#     b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
#     b_noisy = b .+ params.noise_level.*randn(size(b)...)

#     # Launch (background) visualiser
#     st, iterate = initialise_visualisation(visualise)

#     # Define Linear operator
#     #op = CenteredGradientOp()
#     #op = BwdGradientOp()
#     op = FwdGradientOp()

#     # Run algorithm
#     x, y, st = op_denoise_pdps(b_noisy,op; iterate=iterate, params=params)

#     save_results(params, b, b_noisy, x, st)

#     # Exit background visualiser
#     finalise_visualisation(st)
# end

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

    # Define Linear operator
    op₁ = FwdGradientOp()
    op₂ = BwdGradientOp()
    op₃ = CenteredGradientOp()

    # Run algorithm
    x, y₁, y₂, y₃, st = sumregs_denoise_pdps(b_noisy,op₁,op₂,op₃; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end

end # Module
