using Images, ImageQualityIndexes, ImageContrastAdjustment

export Dataset, Experiment, run_experiment, save_experiment

struct Dataset
    name::String
    size::Int
    imsize::Tuple
    original::AbstractArray
    noisy::AbstractArray
end

function Dataset(name,filelist)
    image_pairs = readlines(filelist)
    dir = dirname(filelist)*"/"
    M,N = size(load(dir*split(image_pairs[1],",")[1]))
    original = zeros(M,N,length(image_pairs))
    noisy = zeros(M,N,length(image_pairs))
    for i = 1:length(image_pairs)
        pair = split(image_pairs[i],",")
        original[:,:,i] = load(dir*pair[1])
        noisy[:,:,i] = load(dir*pair[2])
    end
    Dataset(name,length(image_pairs),(M,N),original,noisy)
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
        write(io,"1 \t $(assess_ssim(orig,noisy)) \t $(assess_psnr(orig,noisy)) \t $(assess_ssim(orig,fx.u)) \t $(assess_psnr(orig,fx.u))\n")
    end

end

