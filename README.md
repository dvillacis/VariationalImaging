# VariationalImaging

Numerical solvers for variational imaging problems. In particular, the solvers for variational image denoising are provided:

1. L2-TV (ROF) Denoising using Chambolle-Pock algorithm
2. Abstract Operator Denoising
3. Abstract Sum of Regularizers Denoising

## Installation

This package requires the modules AlgToos (https://tuomov.iki.fi/software/AlgTools/) and ImageTools (https://tuomov.iki.fi/software/ImageTools/) developed by Tuomo Valkonen. To install those it is necessary to clone said repositories

```sh
$ hg clone https://tuomov.iki.fi/repos/AlgTools/
$ hg clone https://tuomov.iki.fi/repos/ImageTools/
```

Once cloned the repositories, we need to upload those to the julia package manager

```julia
pkg> develop AlgTools
pkg> develop ImageTools
pkg> add https://github.com/dvillacis/VariationalImaging.git
```

Finally, to use it just include it on your code header

```julia
using VariationalImaging
```

## Running examples

This package include a Tests module to illustrate the use of the solvers. To access them go to the VariationalImaging folder add paste the following command

```sh
$ julia --project=Tests
```

once Julia's REPL starts load the testing examples

```julia
julia> using Tests.DenoiseTests
julia> test_sd_sumregs_denoise()
```