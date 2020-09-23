# RegularizationTools.jl

*A Julia package to perform Tikhonov regularization for small to moderate size problems.*

RegularizationTools.jl bundles a set routines to compute the [regularized  Tikhonov inverse](https://en.wikipedia.org/wiki/Tikhonov_regularization) using standard linear algebra techniques.  

## Package Features
- Computes the Tikhonov inverse solution.
- Computes optimal regularization parameter using generalized cross validation or the L-curve.
- Solves problems with up to a few 1000s equations.
- Supports zero, first, and second order regularization out of the box.
- Supports specifying an *a-priori* estimate of the solution.
- Supports user specified smoothing matrices.
- User friendly interface.
- Extensive documentation.

## About
Tikhonv regularization is also known as Phillips-Twomey-Tikhonov regularization or ridge regression (see Hansen, 2000 for a review). The Web-of-Sciences database lists more than 4500 peer-reviewed publications mentioning "Tikhonov regularization" in the title or abstract, with a current publication rate of ≈350 new papers/year. 

 The first draft of this code was part of my [DifferentialMobilityAnalyzers](https://mdpetters.github.io/DifferentialMobilityAnalyzers.jl/stable/) package. Unfortunately, the initial set of algorithms were too limiting and too slow. I needed a better set of regularization tools to work with, which is how this package came into existence. Consequently, the scope of the package is defined by my need to support data inversions for the DifferentialMobilityAnalyzers project. My research area is not on inverse methods and I currently do not intend to grow this package into something that goes much beyond the implemented algorithms. However, the code is a generic implementation of the Tikhonov method and might be useful to applied scientists who need to solve standard ill-posed inverse problems that arise in many disciplines. 

# Quick Start

The package computes the regularized Tikhonov inverse ``{\rm x_{\lambda}}`` by solving the minimization problem 

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}\right\} 
```

where ``{\rm x_{\lambda}}`` is the regularized estimate of ``{\rm x}``,
``\left\lVert \cdot\right\rVert _{2}`` is the Euclidean norm, ``{\rm {\bf L}}`` is the Tikhonov filter matrix, ``\lambda`` is the regularization parameter, and ``{\rm x_{0}}`` is a vector of an *a-priori* guess of the solution. The initial guess can be taken to be ``{\rm x_{0}}=0`` if no *a-priori* information is known. The solve function searches for the optimal ``\lambda`` and returns the inverse. The following script is a minimalist example how to use this package.

```@example
using RegularizationTools, MatrixDepot, Lazy
using Random #hide

# This is a test problem for regularization methods
r = mdopen("shaw", 100, false)       # Load the "shaw" problem from MatrixDepot
A, x  = r.A, r.x                     # A is size(100,100), x is length(100)
Random.seed!(716)  # hide

y = A * x                            # y is the true response 
b = y + 0.2y .* randn(100)           # response with superimposed noise
x₀ = 0.4x                            # some a-priori estimate x₀

# Solve 2nd order Tikhonov inversion (L = uppertridiag(−1, 2, −1)) with intial guess x₀
xλ = @> setupRegularizationProblem(A, 2) solve(b, x₀) getfield(:x)
include("theory/helpers.jl") # hide
standard_plot(y, b, x, xλ, x₀) # hide
```

# Installation

The package can be installed from the Julia package prompt with

```julia
julia> ]add  https://github.com/mdpetters/RegularizationTools.jl.git
```

The closing square bracket switches to the package manager interface and the ```add``` command installs the package and any missing dependencies. To return to the Julia REPL hit the ```delete``` key.

To load the package run

```julia
julia> using RegularizationTools
```

For optimal performance, also install the [Intel MKL linear algebra](https://github.com/JuliaComputing/MKL.jl) library.

## Related work
* [MultivariateStats](https://multivariatestatsjl.readthedocs.io/en/stable/index.html): Implements ridge regression without *a priori* estimate and does not include tools to find the optimal regularization parameter.
* [RegularizedLeastSquares](https://tknopp.github.io/RegularizedLeastSquares.jl/latest/): Implements optimization techniques for large-scale scale linear systems.

## Author and Copyright
Markus Petters, Department of Marine, Earth, and Atmospheric Sciences, NC State University.