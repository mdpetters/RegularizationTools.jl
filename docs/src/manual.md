# Manual

A theoretical background of Tikhonov regularization is provided in the [The Inverse Problem](@ref) section. 

## Solving a Problem

A problem consists of a design matrix ``{\rm {\bf A}}`` and a vector ``{\rm b}`` such that

```math
{\rm b} = {\bf {\rm {\bf A}{\rm x}}} + \epsilon
```

where ``\epsilon`` is noise and the objective is to reconstruct the original parameters ``{\rm x}``. The solution to the problem is to minimize

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}\right\} 
```

where ``{\rm x_{\lambda}}`` is the regularized estimate of ``{\rm x}``,
``\left\lVert \cdot\right\rVert _{2}`` is the Euclidean norm, ``{\rm {\bf L}}`` is the Tikhonov filter matrix, ``\lambda`` is the regularization parameter, and ``{\rm x_{0}}`` is a vector of an *a priori* guess of the solution. The initial guess can be taken to be ``{\rm x_{0}}=0`` if no *a priori* information is known. 

The basic steps to solve the problem are 

```julia
Ψ = setupRegularizationProblem(A, 2)   # Setup the problem 
solution = solve(Ψ, b)                 # Compute the solution 
xλ = solution.x                        # Extract the x
```

The ```solve``` function finds [Optimal Regularization Parameter](@ref), by default using [Generalized Cross Validation](@ref). It applies the optimal ``\lambda`` value to compute ``{\rm x}_\lambda``. The solution is of type [RegularizatedSolution](@ref), which contains the optimal solution (```solution.x```), the optimal ``\lambda`` (```solution.λ```) and output from the otimization routine (```solution.solution```).

An convenient way to find x is using [Lazy](https://github.com/MikeInnes/Lazy.jl) pipes:

```julia
xλ = @> setupRegularizationProblem(A, 2) solve(b) getfield(:x)
```

## Specifying the Order
Common choices for the ``{\bf{\rm{L}}}`` matrix are finite difference approximations of a derivative. There are termed zeroth, first, and second order inversion matrices. 

```math
{\rm {\bf L}_{0}=\left(\begin{array}{ccccc}
1 &  &  &  & 0\\
 & 1\\
 &  & \ddots\\
 &  &  & 1\\
0 &  &  &  & 1
\end{array}\right)}
```

```math
{\rm {\bf L}_{1}=\left(\begin{array}{cccccc}
1 & -1 &  &  &  & 0\\
 & 1 & -1\\
 &  & \ddots & \ddots\\
 &  &  & 1 & -1\\
0 &  &  &  & 1 & -1
\end{array}\right)}
```

```math
{\rm {\bf L}_{2}=\left(\begin{array}{ccccccc}
-1 & 2 & -1 &  &  &  & 0\\
 & -1 & 2 & -1\\
 &  & \ddots & \ddots & \ddots\\
 &  &  & -1 & 2 & -1\\
0 &  &  &  & -1 & 2 & -1
\end{array}\right)}
```

You can specify which of these matrices to use in 
```julia
setupRegularizationProblem(A::AbstractMatrix, order::Int)
```
where order = 0, 1, 2 corresponds to ``{\bf{\rm{L}}_0}``, ``{\bf{\rm{L}}_1}``, and 
``{\bf{\rm{L}}_2}``


### Example : [Phillips Problem](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-phillips)

Example with 100 point discretization and zero initial guess.

```@example
using RegularizationTools, MatrixDepot, Lazy
using Random # hide

r = mdopen("phillips", 100, false)
A, x = r.A, r.x
Random.seed!(850) # hide

y = A * x
b = y + 0.1y .* randn(100)
xλ = @> setupRegularizationProblem(A, 2) solve(b) getfield(:x)
include("theory/helpers.jl") # hide
standard_plot(y, b, x, xλ, 0.0x) # hide
```

!!! note
    The random perturbation ```b = y + 0.1y .* randn(100)```  in each of the examples uses a fixed random seed to ensure reproducibility. The random seed and plot commands are hidden for clarity. 

!!! note
    The example system is a test problem for regularization methods is taken from [MatrixDepot.jl](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-shaw) and is the same system used in Hansen (2000).



### Example 2: [Shaw Problem](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-shaw) 

Zeroth order example with 500 point discretization and moderate initial guess.

```@example
using RegularizationTools, MatrixDepot, Lazy, Random

r = mdopen("shaw", 500, false)
A, x = r.A, r.x
Random.seed!(850) #hide 

y = A * x
b = y + 0.1y .* randn(500)
x₀ = 0.6x

xλ = @> setupRegularizationProblem(A, 0) solve(b, x₀) getfield(:x)
include("theory/helpers.jl") # hide
standard_plot(y, b, x, xλ, x₀)# hide
```

## Using a Custom L Matrix

You can specify custom ``{\rm {\bf L}}`` matrices when setting up problems.

```julia
setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)
```

For example, Huckle and Sedlacek (2010) propose a two-step data based regularization

```math
{\rm {\bf L}} = {\rm {\bf L}}_k {\rm {\bf D}}_{\hat{x}}^{-1} 
```

where ``{\rm {\bf L}}_k`` is one of the finite difference approximations of a derivative, 
``{\rm {\bf D}}_{\hat{x}}=diag(|\hat{x_{1}}|,\ldots|\hat{x_{n}}|)``, ``\hat{x}`` is the reconstruction of ``x`` using ``{\rm {\bf L}}_k``, and ``({\rm {\bf D}}_{\hat{x}})_{ii}=\epsilon\;\forall\;|\hat{x_{i}}|<\epsilon``, with ``\epsilon << 1``. 

### Example 3: [Heat Problem ](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-heat)  

This examples illustrates how to implement the Huckle and Sedlacek (2010) matrix. Note that ```Γ(A, 2)``` returns the [Tikhonov Matrix](@ref) of order 2. 

```@example
using RegularizationTools, MatrixDepot, Lazy, Random, LinearAlgebra

r = mdopen("heat", 100, false)
A, x = r.A, r.x
Random.seed!(150) #hide

y = A * x
b = y + 0.05y .* randn(100)
x₀ = zeros(length(b))

L₂ = Γ(A,2)                  
xλ1 = @> setupRegularizationProblem(A, L₂) solve(b) getfield(:x)
x̂ = deepcopy(abs.(xλ1))
x̂[abs.(x̂) .< 0.1] .= 0.1
L = L₂*Diagonal(x̂)^(-1)
xλ2 = @> setupRegularizationProblem(A, L) solve(b) getfield(:x)
include("theory/helpers.jl") # hide
standard_plot1(y, b, x, xλ1, xλ2) # hide
```

The solution xλ2 is improved over the regular L₂ solution. 

## Customizing the Search Algorithm
The solve function searches for the optimum regularization parameter ``\lambda`` between ``[\lambda_1, \lambda_2]``. The default search range is [0.001, 1000.0] and the interval range can be modified through keyword parameters. The optimality criterion is either the minimum of the [Generalized Cross Validation](@ref) function, or the the maximum curvature of the L-curve (see [L-Curve Algorithm](@ref)). The algorithm can be specified through the alg keyword. Valid algorithms are ```:L_curve```, ```:gcv_svd```, and ```:gcv_tr``` (see [Solve](@ref)).

### Example: [Baart Problem](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-baart)

```@example
using RegularizationTools, MatrixDepot, Lazy, Random

r = mdopen("baart", 100, false)
A, x = r.A, r.x
Random.seed!(150) #hide

y = A  * x
b = y + 0.1y .* randn(100)

xλ1 = @> setupRegularizationProblem(A,1) solve(b, alg=:L_curve, λ₁=100.0, λ₂=1e6) getfield(:x)
xλ2 = @> setupRegularizationProblem(A,1) solve(b, alg=:gcv_svd) getfield(:x)
include("theory/helpers.jl") # hide
x₀ = 0.0*x # hide
standard_plot1(y, b, x, xλ1, xλ2) # hide
```

Note that the output from the L-curve and GCV algorithm are nearly identical. 

!!! note
    The L-curve algorithm is more sensitive to the bounds and slower than the gcv\_svd algorithm. There may, however, be cases where the L-curve approach is preferable. 

## Extracting the Validation Function

The solution is obtained by first transforming the problem to standard form (see [Transformation to Standard Form](@ref)). The following example can be used to extract the [GCV](@ref) function.

```@example
using RegularizationTools, MatrixDepot, Lazy, Random, Underscores, Printf

r = mdopen("shaw", 100, false)
A, x = r.A, r.x
Random.seed!(716) #hide

y = A  * x
b = y + 0.1y .* randn(100)

Ψ = setupRegularizationProblem(A,1)
λopt = @> solve(Ψ, b, alg=:gcv_svd) getfield(:λ)
b̄ = to_standard_form(Ψ, b) 
λs = exp10.(range(log10(1e-1), stop = log10(10), length = 100))
Vλ = @_ map(gcv_svd(Ψ, b̄, _), λs) 
include("theory/helpers.jl") # hide
graph3(λs, Vλ) # hide
```

The calculated λopt from 

```julia
λopt = @> solve(Ψ, b, alg=:gcv_svd) getfield(:λ)
```

is 1.1 and corresponds to the minimum of the GCV curve. 

Alternatively, the L-curve is retrieved through the [L-curve Functions](@ref)
```@example
using RegularizationTools, MatrixDepot, Lazy, Random, Underscores, Printf

r = mdopen("shaw", 100, false)
A, x = r.A, r.x
Random.seed!(716) #hide

y = A  * x
b = y + 0.1y .* randn(100)

Ψ = setupRegularizationProblem(A,1)
λopt = @> solve(Ψ, b, alg=:L_curve, λ₂ = 10.0) getfield(:λ)
b̄ = to_standard_form(Ψ, b) 
λs = exp10.(range(log10(1e-3), stop = log10(100), length = 200))
L1norm, L2norm, κ = Lcurve_functions(Ψ, b̄)
L1, L2 = L1norm.(λs), L2norm.(λs)    
include("theory/helpers.jl") # hide
graph4(L1, L2) # hide
```

The calculated λopt from 

```julia
λopt = @> solve(Ψ, b, alg=:L_curve, λ₂ = 10.0) getfield(:λ)
```

is 0.9 and corresponds to the corner of the L-curve.

## Benchmarks

Systems up to a few 1000 equations are unproblematic. The setup for much larger system slows down due to the ``\approx O(n^2)`` (or worse) time complexity of the SVD and QR factorization of the design matrix. Larger systems require switching to SVD free algorithms, which currently are supported by this package. 

```@example
using RegularizationTools, TimerOutputs, MatrixDepot

to = TimerOutput()

function benchmark(n)
    r = mdopen("shaw", n, false)
    A, x = r.A, r.x
    y = A * x
    b = y + 0.05y .* randn(n)
    Ψ = setupRegularizationProblem(A, 2)
    for i = 1:5
        @timeit to "Setup  (n = $n)" setupRegularizationProblem(A, 2)
        @timeit to "Invert (n = $n)" solve(Ψ, b)
    end
end

map(benchmark, [10, 100, 1000])
show(to)
```
