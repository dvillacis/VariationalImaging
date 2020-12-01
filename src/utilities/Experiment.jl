using Images, ImageQualityIndexes, ImageContrastAdjustment

export Dataset, Experiment, run_experiment, save_experiment

struct Dataset
    name::String
    size::Tuple
    imsize::Tuple
    original::AbstractArray
    noisy::AbstractArray
end

struct Experiment
    name::String
    path::String
    learning_function::Function
end

function run_experiment(ex::Experiment,ds::Dataset;verbose=false,freq=10,maxit=100,tol=1e-4)
    M,N = ds.imsize
    x₀ = (1/(M*N)) * ones()
    solver = NSTR(;verbose=verbose,freq=freq,maxit=maxit,tol=tol)
    opt,fx,res,iters = solver(x->ex.learning_function(x,ds),x₀;Δ₀=0.1)
    save_experiment(ex,ds,opt,fx,res,iters)
end

function save_experiment(ex::Experiment,ds::Dataset,opt::Real,fx,res,iters)
    ex_path = ex.path*"/"*ex.name 
    if !isdir(ex_path)
        mkdir(ex_path)
    end
    # Saving optimally denoised images
    opt_denoised_img = fx.u
    opt_denoised_path = ex_path*"/optimal_denoised.png"
    save(opt_denoised_path,opt_denoised_img)
    # Saving experiment report
    report_path = ex_path*"/"*ex.name*"_report.txt"
    open(report_path,"w") do io
        write(io,"Optimal Parameter: $opt\n")
        write(io,"Residual: $res\n")
        write(io,"Number of Iterations: $iters\n")
        write(io,"Quality Measures\n")
        write(io,"num \t orig_ssim \t orig_psnr \t opt_ssim \t opt_psnr\n")
        write(io,"1 \t $(assess_ssim(ds.original,ds.noisy)) \t $(assess_psnr(ds.original,ds.noisy)) \t $(assess_ssim(ds.original,fx.u)) \t $(assess_psnr(ds.original,fx.u))\n")
    end
end

function save_experiment(ex::Experiment,ds::Dataset,opt::AbstractArray,fx,res,iters)
    ex_path = ex.path*"/"*ex.name 
    if !isdir(ex_path)
        mkdir(ex_path)
    end
    # Saving optimal parameter
    parameter_img_path = ex_path*"/optimal_parameter.png"
    save(parameter_img_path,opt)
    # Saving optimally denoised images
    opt_denoised_img = fx.u
    opt_denoised_path = ex_path*"/optimal_denoised.png"
    save(opt_denoised_path,opt_denoised_img)
    # Saving experiment report
    orig = ds.original
    noisy = ds.noisy
    report_path = ex_path*"/"*ex.name*"_report.txt"
    open(report_path,"w") do io
        write(io,"Residual: $res\n")
        write(io,"Number of Iterations: $iters\n")
        write(io,"Quality Measures\n")
        write(io,"num \t orig_ssim \t orig_psnr \t opt_ssim \t opt_psnr\n")
        write(io,"1 \t $(assess_ssim(orig,noisy) \t assess_psnr(orig,noisy) \t $(assess_ssim(orig,fx.u) \t assess_psnr(orig,fx.u)\n")
    end

end

