# The Inverse Problem

Consider the following linear system of equations

```math
{\bf {\rm {\bf A}{\rm x}={\rm y}}}
```

where ``{\bf {\rm {\bf A}}}`` is a square design matrix, ``{\rm x}``
is a vector of input parameters and ``{\rm y}`` is a vector of responses.
To estimate unknown inputs from response, the matrix inverse can be
used

```math
{\rm x={\rm {\bf A}}^{-1}y}
```

However, if a random measurement error ``\epsilon`` is superimposed
on ``{\rm y}``, i.e. ``b_{i}=y_{i}+\epsilon_{i}``, the estimate ``{\rm \hat{x}}`` from the matrix inverse 

```math
{\rm \hat{x}={\rm {\bf A}}^{-1}b}
``` 

becomes dominated by contributions from data error for large systems. 

### Example
!!! note
    The example system is a test problem for regularization methods is taken from [MatrixDepot.jl](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-shaw) and is the same system used in Hansen (2000).

Consider the following example system of 100 equations. The matrix ``\rm{\bf{A}}`` is 100x100.
```@example
using MatrixDepot               # hide
r = mdopen("shaw", 100, false)  # hide
A, x, y = r.A, r.x, r.b         # hide

A
```

The vector ``\rm{x}`` of input variables has 100 elements.
```@example
using MatrixDepot               # hide
using Random                    # hide
r = mdopen("shaw", 100, false)  # hide
A, x, y = r.A, r.x, r.b         # hide
Random.seed!(302)               # hide

x
```

Computing ``\rm{y}`` and ``\rm{b}`` using the pseudo-inverse ```pinv``` shows that the error in ``\rm{b}`` makes the inversion unusable.
```@example
using Cairo               # hide
using Fontconfig          # hide
using RegularizationTools # hide
using MatrixDepot         # hide
using Gadfly              # hide
using Random              # hide
using DataFrames          # hide
using LinearAlgebra       # hide
using Printf              # hide
using Underscores         # hide
using Colors #hide

include("helpers.jl")     # hide
r = mdopen("shaw", 100, false)  # hide
A, x, y = r.A, r.x, r.b         # hide
Random.seed!(302)               # hide

y = A * x
b = y + 0.1y .* randn(100)
x = pinv(A) * y
x̂ = pinv(A) * b
# hide
n = length(x) # hide
d = 1:1:n # hide
df1 = DataFrame(x = d, y = y, Color = ["y" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = b, Color = ["b" for i = 1:n]) # hide
df = [df1; df2] # hide
p1 = graph(df) # hide
df1 = DataFrame(x = d, y = x, Color = ["x" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = x̂, Color = ["x̂" for i = 1:n]) # hide
p2 = graph(df1) # hide
p3 = graph(df2, colors = ["darkred"]) # hide
set_default_plot_size(22cm, 7cm) # hide
hstack(p1, p2, p3) # hide
```
## Tikhonov Regularization

Tikhonov regularization is a means to filter this noise by solving the minimization problem 

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}\right\} 
```

where ``{\rm x_{\lambda}}`` is the regularized estimate of ``{\rm x}``,
``\left\lVert \cdot\right\rVert _{2}`` is the Euclidean norm, ``{\rm {\bf L}}`` is the Tikhonov filter matrix, ``\lambda`` is the regularization parameter, and ``{\rm x_{0}}`` is a vector of an *a priori* guess of the solution. The initial guess can be taken to be ``{\rm x_{0}}=0`` if no *a priori* information is known. The matrix ``{\rm {\bf A}}`` does not need to be square. 

For ``\lambda=0`` the Tikhonov problem reverts to the ordinary least
squares solution. If ``{\rm {\bf A}}`` is square and ``\lambda=0``,
the least-squares solution is ``{\rm \hat{x}={\rm {\bf A}}^{-1}b}``. For large ``\lambda`` the solution reverts to the initial guess.,
i.e. ``\lim_{\lambda\rightarrow\infty}{\rm x_{\lambda}}={\rm x_{0}}``.
Therefore, the regularization parameter ``\lambda`` interpolates between
the initial guess and the noisy ordinary least squares solution. The
filter matrix ``{\rm {\bf L}}`` provides additional smoothness constraints on the solution. The simplest form is to use the identity matrix, ``{\rm {\bf L}}={\rm {\bf I}}``.

The formal solution to the Tikhonov problem is given by

```math
{\rm x_{\lambda}}=\left({\rm {\bf A}^{T}}{\rm {\bf A}}+\lambda^{2}{\rm {\bf L}^{T}{\rm {\bf L}}}\right)^{-1}\left({\rm {\bf A}^{T}}{\rm b}+\lambda^{2}{\rm {\bf L}^{T}{\rm {\bf L}}{\rm x_{0}}}\right)
```

The equation is readily derived by writing ``f=\left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}``,
take ``\frac{df}{d{\rm {\rm x}}}=0``, and solve for ``{\rm x}``. Use
[http://www.matrixcalculus.org/](http://www.matrixcalculus.org/)
to validate symbolic matrix derivatives.

### Example
Here is a simple regularized inversion for the same system using ``{\rm {\bf L}}={\rm {\bf I}}`` and ``{\rm x_{0}}=0``. The regularized solution is 

```math
{\rm x_{\lambda}}=\left({\rm {\bf A}^{T}}{\rm {\bf A}}+\lambda^{2}{\rm {\bf I}}\right)^{-1}{\rm {\bf A}^{T}}{\rm b}
```

The regularized inverse can be trivially computed assuming a value for ``\lambda = 0.11``.
```@example
using RegularizationTools # hide
using MatrixDepot # hide
using Gadfly # hide
using Random # hide
using DataFrames # hide
using Colors # hide
using LinearAlgebra # hide
using Underscores # hide
using Printf #  hide

include("helpers.jl") # hide
r = mdopen("shaw", 100, false) # hide
A, x, y = r.A, r.x, r.b # hide
Random.seed!(716) # hide

y = A * x
b = y + 0.1y .* randn(100)
Iₙ = Matrix{Float64}(I, 100, 100)
λ = 0.11
xλ = inv(A'A + λ^2.0 * Iₙ) * A' * b

n = length(x) # hide
d = 1:1:n # hide
df1 = DataFrame(x = d, y = y, Color = ["y" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = b, Color = ["b" for i = 1:n]) # hide
df = [df1; df2] # hide
p1 = graph(df) # hide
# hide
df1 = DataFrame(x = d, y = x, Color = ["x" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = xλ, Color = ["xλ" for i = 1:n]) # hide
df = [df1; df2] # hide
p2 = graph(df) # hide
set_default_plot_size(15cm, 6cm) # hide
hstack(p1, p2) # hide
```

The solution is not perfect, but it is free of random error and a reasonable approximation of the true ``\rm{x}``.

# Optimal Regularization Parameter
The choice of the optimal regularization parameter is not obvious. If we pick ``\lambda`` too small, the solution is dominated by noise. If we pick ``\lambda`` too large the solution will not approximate the correct solution.

```@example
using RegularizationTools # hide
using MatrixDepot # hide
using Gadfly # hide
using Random # hide
using DataFrames # hide
using Colors # hide
using LinearAlgebra # hide
using Underscores #hide
using Printf #  hide

include("helpers.jl") # hide
r = mdopen("shaw", 100, false) # hide
A, x, y = r.A, r.x, r.b # hide
Random.seed!(716) # hide

y = A * x
b = y + 0.1y .* randn(100)
Iₙ = Matrix{Float64}(I, 100, 100)
f(λ) = inv(A'A + λ^2.0 * Iₙ) * A' * b
xλ1 = f(0.001)
xλ2 = f(0.1) 
xλ3 = f(10.0)

n = length(x) # hide
d = 1:1:n # hide
df1 = DataFrame(x = d, y = y, Color = ["y" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = b, Color = ["b" for i = 1:n]) # hide
df = [df1; df2] # hide
p1 = graph(df) # hide
# hide
df1 = DataFrame(x = d, y = x, Color = ["x" for i = 1:n]) # hide
df2 = DataFrame(x = d, y = xλ1, Color = ["xλ1" for i = 1:n]) # hide
df3 = DataFrame(x = d, y = xλ2, Color = ["xλ2" for i = 1:n]) # hide
df4 = DataFrame(x = d, y = xλ3, Color = ["xλ3" for i = 1:n]) # hide
df = [df1; df2; df3; df4] # hide
p2 = graph(df; colors = ["black", "steelblue3", "darkred", "darkgoldenrod"]) # hide
set_default_plot_size(15cm, 6cm) # hide
hstack(p1, p2) # hide
```
``\lambda = 0.1`` provides an acceptable solution. ``\lambda = 0.001`` is noisy (under-regularized) and ``\lambda = 10.0`` is incorrect (over-regularized). There are several objective methods to find the optimal regularization parameter. The general procedure to identify the optimal ``\lambda`` is to compute ``{\rm x_{\lambda}}`` for
a range of regularization parameters over the interval [``\lambda_1``, ``\lambda_2``] and then apply some evaluation criterion that objectively evaluates the quality of the solution. This package implements two of these, the L-curve method and generalized cross validation.

## L-Curve Method
The L-curve method evaluates the by balancing the size of the residual
norm ``L_{1}=\left\lVert {\bf {\rm {\bf A}{\rm x_{\lambda}}-{\rm b}}}\right\rVert _{2}`` and the size of the solution norm ``L_{2}=\left\lVert {\rm {\bf L}({\rm x_{\lambda}}-{\rm x_{0}})}\right\rVert _{2}`` for ``{\rm x_{\lambda}}\in[\lambda_{1},\lambda_{2}]``. The L-curve
consists of a plot of ``\log L_{1}`` vs. ``\log L_{1}``. The following example illustrates the L-curve without specifying an *a priori* input.
```@example
using MatrixDepot # hide
using Gadfly  # hide
using Random # hide
using DataFrames # hide
using Colors # hide
using LinearAlgebra # hide
using Underscores # hide
using NumericIO # hide
using Printf # hide

include("helpers.jl") # hide
r = mdopen("shaw", 100, false) # hide
A, x, y = r.A, r.x, r.b # hide
Random.seed!(716) # hide

y = A * x
b = y + 0.1y .* randn(100)
Iₙ = Matrix{Float64}(I, 100, 100)
f(λ) = inv(A'A + λ^2.0 * Iₙ) * A' * b
L1(λ) = norm(A * f(λ) - b)
L2(λ) = norm(Iₙ * f(λ))
λs = exp10.(range(log10(1e-5), stop = log10(10), length = 100))
residual, solution = L1.(λs), L2.(λs)

set_default_plot_size(5inch, 3.5inch) # hide
graph1(residual, solution) # hide
```

The optimal ``\lambda_{opt}`` is the corner of the L-curve. In this example this is ``\lambda_{opt} \approx 0.1``, which yielded the acceptable solution earlier. Finding the corner of the L-curve can be automated by performing an gradient descent search to find the mimum value of the curvature of the L-curve (Hansen, 2000). The implementation is discussed in the [L-Curve Algorithm](@ref) section.


The solve function in RegularizationTools can be used to find λopt through the L-curve algorithm, searching over the predefined interval [``\lambda_1``, ``\lambda_2``].

```@example
using MatrixDepot # hide
using Lazy   # hide
using Random # hide
using RegularizationTools

r = mdopen("shaw", 100, false) #hide
A, x, y = r.A, r.x, r.b   # hide
Random.seed!(716)  # hide
y = A * x
b = y + 0.1y .* randn(100)
solution = @> setupRegularizationProblem(A, 0) solve(b, alg = :L_curve, λ₁ = 0.01, λ₂ = 1.0)
λopt = solution.λ
```

## Generalized Cross Validation
If the general form of the problem 

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}\right\}
```

has the smoothing matrix ``{\rm {\bf L={\rm {\bf I}}}}``, the problem
is considered to be in standard form. The general-form problem can be transformed into standard form (see [Transformation to Standard Form](@ref) for algorithm). If the problem is in standrad form, and if ``{\rm x_{0}}=0``, the GCV estimate of ``\lambda`` is (Golub et al., 1979):

```math
V(\lambda)=\frac{n\left\lVert ({\bf {\rm {\bf {\bf I}-}{\bf A_{\lambda}}){\rm b}}}\right\rVert _{2}^{2}}{tr({\rm {\bf I}-{\rm {\bf A_{\lambda}})}^{2}}}
```

where ``{\bf A_{\lambda}}={\rm {\rm {\bf A}}}\left({\bf AA}^{T}-\lambda^{2}{\rm {\bf I}}\right)^{-1}{\bf A}^{T}`` is the influence matrix, tr is the [matrix trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)), ``n`` is the size of ``\rm{b}``. Note that ``{\rm x_{\lambda}}=\left({\rm {\bf A}^{T}}{\rm {\bf A}}+\lambda^{2}{\rm {\bf I}}\right)^{-1}{\rm {\bf A}^{T}}{\rm b}``. Therefore ``{\rm {\rm {\rm {\bf A}}}x_{\lambda}={\rm {\bf A_{\lambda}}b}}`` and ``\left\lVert ({\bf {\rm {\bf {\bf I}-}{\bf A_{\lambda}}){\rm b}}}\right\rVert _{2}=\left\lVert {\bf {\rm {\bf A}{\rm x_{\lambda}}-{\rm b}}}\right\rVert _{2}``. The optimal ``\lambda_{opt}`` coincides with the global minimum of ``V(\lambda)``. 

The following example evaluates ``V(\lambda)`` over a range of ``\lambda``. 
```@example
using MatrixDepot #hide
using Gadfly #hide
using Random #hide
using Colors #hide
using DataFrames #hide
using Printf#hide
using Lazy#hide
using Underscores #hide
using LinearAlgebra #hide
include("helpers.jl") #hide
r = mdopen("shaw", 100, false) #hide
A, x, y = r.A, r.x, r.b #hide
Random.seed!(716) #hide
y = A * x
b = y + 0.1y .* randn(100)
Iₙ = Matrix{Float64}(I, 100, 100)
Aλ(λ) = A*inv(A'A + λ^2.0*Iₙ)*A'
gcv(λ) = 100*norm((Iₙ - Aλ(λ))*b)^2.0/tr(Iₙ - Aλ(λ))^2.0
λs = exp10.(range(log10(1e-3), stop = log10(1), length = 100))
V = map(gcv, λs)
#hide
set_default_plot_size(5inch, 3.5inch) # hide
graph2(λs, V) #hide
```

The GCV curve has a steep part for large ``\lambda`` and a shallow part for small ``\lambda``. The minimum occurs near ``\lambda = 0.1``.  The ```solve``` function in RegularizationTools can be used to find λopt through the GCV approach, searching over the predefined interval [``\lambda_1``, ``\lambda_2``].

```@example
using MatrixDepot # hide
using Lazy   # hide
using Random # hide
using RegularizationTools

r = mdopen("shaw", 100, false) #hide
A, x, y = r.A, r.x, r.b   # hide
Random.seed!(716)  # hide
y = A * x
b = y + 0.1y .* randn(100)
solution = @> setupRegularizationProblem(A, 0) solve(b, alg = :gcv_svd, λ₁ = 0.01, λ₂ = 1.0)
λopt = solution.λ
```

Note that the objective λopt from the L-curve and GCV criterion are nearly identical. 


## Transformation to Standard Form

The general from couched in terms of ``\{\rm\bf{A}, \rm b, \rm x, \rm x_0\}`` and ``\rm{\bf{L}}``

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf A}{\rm x}-{\rm b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf L}({\rm x}-{\rm x_{0}})}\right\rVert _{2}^{2}\right\}
```

is transformed to standard form couched in terms of ``\{\rm \bf {\bar A}, \rm \bar b, \rm \bar x, \rm \bar{x}_0\}`` and ``\rm{\bf{I}}``

```math
{\rm {\rm x_{\lambda}}}=\arg\min\left\{ \left\lVert {\bf {\rm {\bf \bar A}{\rm x}-{\rm \bar b}}}\right\rVert _{2}^{2}+\lambda^{2}\left\lVert {\rm {\bf I}({\rm \bar x}-{\rm \bar x_{0}})}\right\rVert _{2}^{2}\right\}
```

as introduced by Eldén (1977) using notation from Hansen (1998, Chapter 2.3.1). The algorithm computes the explicit transformation using two [QR factorizations](https://en.wikipedia.org/wiki/QR_decomposition). The matrices needed for the explicit conversion depend on ``\rm \bf A`` and ``\rm \bf L`` and are computed and cached in [setupRegularizationProblem](@ref). 

The λopt search is performed in in standard form, and the solution is computed in standard form. Then the resulting solution is transformed back to the general form using the same matrices.

## Solving the Standard Equations

The solution ``{\rm \bar x_{\lambda}}`` for the transformed standard equation is 

```math
{\rm \bar x_{\lambda}}=\left({\rm {\bf \bar A}^{T}}{\rm {\bf \bar A}}+\lambda^{2}{\rm {\bf I}}\right)^{-1} \left( {\rm {\bf \bar A}^{T}}{\rm \bar b} + λ^2 \rm \bar {x}_0 \right)
```

``{\rm \bar x_{\lambda}}`` is found using the [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition). It is the alorithm used in the [MultivariateStats](https://multivariatestatsjl.readthedocs.io/en/stable/index.html) package and generally faster than the QR approach (Lira et al., 2016).

## L-Curve Algorithm
The curvature ``\kappa(\lambda)`` of the L-curve is computed using Eq. (18) in Hansen (2000). The expression requires calculation of the solution and residual norms, as well as the first derivative of the solution norm. The derivative is calculated using finite differences from the [Calculus](https://github.com/JuliaMath/Calculus.jl) package
The corner of the L-curve occurs when the curvature maximizes. Finally, 
``λ_{opt}`` is found by minimizing the ``-\kappa(\lambda)`` function (see [L-Curve Method](@ref)) on a bounded interval using [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method), as implemented in the [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) package.

## Generalized Cross Validation Algorithm

```math
V(\lambda)=\frac{n\left\lVert ({\bf {\rm {\bf {\bf I}-}{\bf \bar A_{\lambda}}){\rm \bar b}}}\right\rVert _{2}^{2}}{tr({\rm {\bf I}-{\rm {\bf \bar A_{\lambda}})}^{2}}}
```

The slowest part in the GCV calculation is evaluation of the trace. 
The GCV estimate is computed either using the [single value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) algorithm (Golub et al., 1979) (```gcv_svd```) or the explicit calculation using the trace term (```gcv_tr```). The SVD of the design matrix in standard form ``\rm \bf{\bar A}`` is calculated and cached in [setupRegularizationProblem](@ref). When an initial guess is included, the denominator is computed using the SVD estimate and the numerator is computed via ``
 \left\lVert {\bf {\rm {\bf \bar A}{\rm \bar x_{\lambda}}-{\rm \bar b}}}\right\rVert _{2}^2
`` and ``\rm \bar x_\lambda`` is obtained using the Cholesky decomposition algorithm solving for the Tikhonov solution in standard form with an initial guess. Note that this is an approximation because the trace term in the denominator does not account for the intial guess. Comparison with the L-curve method suggest that this approximation does not affect the quality of the regularized solution. Finally, ``λ_{opt}`` is found by minimizing ``V(\lambda)`` on a bounded interval using [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method), as implemented in the [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) package.
