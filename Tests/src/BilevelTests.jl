

module BilevelTests

# Parameters
const default_save_prefix="bilevel_result_"

const default_params = (
    verbose_iter = 50,
    maxiter = 1500,
    save_results = true,
    dataset_name = "lighthouse",
    save_iterations = false
)

const bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 2.0,
    Δ₀ = 1.0,
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
# Bilevel learn test
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
    op = BwdGradientOp()
    #op = FwdGradientOp()

    # Run algorithm
    x, y, st = op_denoise_pdps(b_noisy,op; iterate=iterate, params=params)

    save_results(params, b, b_noisy, x, st)

    # Exit background visualiser
    finalise_visualisation(st)
end


end