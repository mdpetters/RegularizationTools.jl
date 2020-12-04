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

For example, Huckle and Sedlacek (2012) propose a two-step data based regularization

```math
{\rm {\bf L}} = {\rm {\bf L}}_k {\rm {\bf D}}_{\hat{x}}^{-1} 
```

where ``{\rm {\bf L}}_k`` is one of the finite difference approximations of a derivative, 
``{\rm {\bf D}}_{\hat{x}}=diag(|\hat{x_{1}}|,\ldots|\hat{x_{n}}|)``, ``\hat{x}`` is the reconstruction of ``x`` using ``{\rm {\bf L}}_k``, and ``({\rm {\bf D}}_{\hat{x}})_{ii}=\epsilon\;\forall\;|\hat{x_{i}}|<\epsilon``, with ``\epsilon << 1``. 

### Example 3: [Heat Problem ](https://matrixdepotjl.readthedocs.io/en/latest/regu.html#term-heat)  

This examples illustrates how to implement the Huckle and Sedlacek (2012) matrix. Note that ```Γ(A, 2)``` returns the [Tikhonov Matrix](@ref) of order 2. 

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

## Adding Boundary Constraints
To add boundary constraints (e.g. enforce that all solutions are positive), the following procedure is implemented. Computes the algebraic solution without constraints, truncate the solution at the upper and lower bounds, and use the result as initial condition for solving the minimization problem with a least squares numerical solver [LeastSquaresOptim](https://github.com/matthieugomez/LeastSquaresOptim.jl). The regularization parameter λ obtained from the algebraic solution is used for a single pass optimization. See [Solve](@ref) for a complete list of methods.

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
ψ = @>> L₂*Diagonal(x̂)^(-1) setupRegularizationProblem(A) 
lower = zeros(100)
upper = ones(100)
xλ2 = @> solve(ψ, b, lower, upper) getfield(:x)
include("theory/helpers.jl") # hide
standard_plot1(y, b, x, xλ1, xλ2) # hide
```

This is the same example as above, but imposing a lower and upper bound on the solution. Note that this solve method computes the algebraic solution using ```solve(Ψ, b; kwargs...)``` method to compute the starting point for the least square minimization. You can also call

```julia
xλ2 = @> solve(ψ, b, x₀, lower, upper) getfield(:x)
```

to bound the least-square solver by the output from the ```solve(Ψ, b, x₀; kwargs...)``` method.

## Customizing the Search Algorithm
The solve function searches for the optimum regularization parameter ``\lambda`` between ``[\lambda_1, \lambda_2]``. The default search range is [0.001, 1000.0] and the interval range can be modified through keyword parameters. The optimality criterion is either the minimum of the [Generalized Cross Validation](@ref) function, or the the maximum curvature of the L-curve (see [L-Curve Algorithm](@ref)). The algorithm can be specified through the alg keyword. Valid algorithms are ```:L_curve```, ```:gcv_svd```, and ```:gcv_tr``` (see [Solve](@ref)).

### Example 

```@example
using RegularizationTools, MatrixDepot, Lazy, Random

r = mdopen("shaw", 100, false)
A, x = r.A, r.x
Random.seed!(100) #hide

y = A  * x
b = y + 0.1y .* randn(100)

xλ1 = @> setupRegularizationProblem(A,1) solve(b, alg=:L_curve, λ₁=0.1, λ₂=10.0) getfield(:x)
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

## Creating a Design Matrix

### Standard Approach


This package provides an abstract generic interface to create a design matrix from a forward model of a linear process. To understand this functionality, first consider the standard approach to find the design matrix by discretization of the Fredholm integral equation.

Consider a one-dimensional Fredholm integral equation of the first kind on a finite interval:

$\int_{a}^{b}K(q,s)f(s)ds=g(q)\;\;\;\;\;\;s\in[a,b]\;\mathrm{and}\;q\in[c,d]$

where $K(q,s)$ is the kernel. The inverse problem is to find $f(s)$ for a given $K(q,s)$ and $g(q)$.

The integral equation can be cast as system of linear equations such that

$\mathbf{A}\mathrm{x=b}$

where $\mathrm{x}=[x_{1},\dots,x_{n}]=f(s_{i})$ is a discrete vector representing f, $\mathrm{b=}[b_{1},\dots,b_{n}]=g(q_{j})$ is a vector representing $g$ and $\mathrm{\mathbf{A}}$ is the $n\times n$ design matrix.

Using the quadrature method (Hansen, 2008), the integral is approximated by a weighted sum such that

$\int_{a}^{b}K(q_{i},s)f(s)ds=g(q_{i})\approx\sum_{j=1}^{n}wK(q_{i},s_{j})f(s_{j})$

where $w=\frac{b-a}{n}$, and $s_{j}=(j-\frac{1}{2})w$. The elements comprising the design matrix $\mathrm{\mathbf{A}}$ are $a_{i,j}=wK(q_{i},s_{j})$.

This simple kernel (Baart, 1982) serves as an illustrative example

$K(q,s)=\exp(q\cos(s))$

$f(s)=\sin(s)$

$g(q)=\frac{2\sin s}{s}$

$s\in[0,\pi]\;\mathrm{and}\;q\in[0,\pi/2]$

The following imperative code discretizes the problem with $n=12$ points:

```julia
a, b = 0.0, π
n, m = 12,12
c, d = 0.0, π/2
A = zeros(n,m)
w = (b-a)/n
baart(x,y) = exp(x*cos(y))
q = range(c, stop = d, length = n)
s = [(j-0.5)*(b-a)/n for j = 1:m]

for i = 1:n
    for j = 1:m
        A[i,j] = w*baart(q[i], (j-0.5)*w)
    end
end
```

In this example, the functions $f(s)$ and $g(q)$ are known. Evaluating $\mathrm{b}=\mathrm{\mathbf{A}x}$ with $\mathrm{\mathrm{x}}=sin(s)$ approximately yields $2sin(q)/q$. 

```julia
x = sin.(s)
b1 = A*x 
b2 = 2.0.*sinh.(q)./q
```

Here, b1 and b2 are close. However, as noted by Hansen (2008), the product $\mathbf{A}\mathrm{x}$ is, in general, different from $g(q)$ due to discretization errors.

### Alternative Approach

Discretization of the kernel may be less straight forward for more complex kernels that describe physical processes. Furthermore, in physical processes or engineering systems, the mapping of variables isn't immediately clear. This is especially true when thinking about kernel functions of processes that have not yet been modeled in the literature.

Imagine the following problem studying a system of interest with some instrument. The instrument has a dial and intends to measure a physical property $x$ that is expressible as a numeric value. The dial can be any abstract notion such as "channel number", "input voltage", or even a combination of parameters such as "channel number 1 at high gain", "channel number 1 at low gain", and so forth. At each dial setting the instrument passes the physical property through a filter that produces an observable reading $b$. The domain of the problem consists of a list of valid set points $[s_{i}]$, with a corresponding list of physical properties $[x_{i}]$.

![image](assets/inverse.png)

We can then defined a node element of the domain as a product data type

```julia
struct Domain{T1<:Any,T2<:Number,T3<:Any}
    s::AbstractVector{T1}
    x::AbstractVector{T2}
    q::T3
end
```

A single node in the domain is defined as

```julia
node = Domain([s], [x], q)
```

where $q$ is an abstract query value of the instrument. The forward problem is to defined a function that maps a node of the domain to the observable $b$, i.e.

```julia
b = f(node) 
```

Here $b$ corresponds to a numerical value displayed by the detector at setting $q$.

The higher order function [designmatrix](@ref)

```julia
function designmatrix(s::Any, q::Any, f::Function)::AbstractMatrix
    n = length(s)
    x(i) = @_ map(_ == i ? 1.0 : 0.0, 1:n)
    nodes = [Domain(s, x(i), q[j]) for i ∈ 1:n, j ∈ 1:length(q)]
    return @> map(f, nodes) transpose copy
end
```

maps the list of setpoints s, query points q to the design matrix A for any function ```f(node)``` that can operate and the defined setpoints.

Then the $\mathrm{b}=\mathrm{\mathbf{A}x}$ is the linear transformation from a list of physical properties $[x_{i}]$ at query points $[q_{i}]$.

Below are three examples on how the function designmatrix works. Complete examples are included in the examples folder.

### Example 1

#### Problem Setup

Consider a black and white camera taking a picture in a turbulent atmosphere. The camera has 200x200 pixel resolution. The input values to the camera correspond to the photon count hitting the detector. The output corresponds to a gray scale value between 0 and 1. Let us ignore other effects such as lens distortions, quantum efficiency of the detector, wavelength dependency, or electronic noise. However, the turbulent atmosphere will blur the image by randomly redirecting photons. The [Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) kernel function is

$K(x,y)=\frac{1}{2\pi\sigma}\exp\left(-\frac{x^{2}+y^{2}}{2\sigma^{2}}\right)$

where x, y are the the distances from the pixel coordinate in the horizontal and vertical axis, and $\sigma$ is the standard deviation of the Gaussian distribution. The value of $K(x,y)$ is zero for pixels further than $x>c$ and $y>c$.

Each setpoint is $s_{i}$ is a tuple of physical coordinates (x,y). The domain for setpoints is a vector of tuples covering all pixels. 

```@example
using Lazy

s = @> [(i,j) for i = 0:20, j = 0:20] reshape(:)
```

The domain of query points $q_{i}$ are pixel coordinates. For this problem it is sensible that $[s]=[q]$, but this need not be the case. The function $f(node)\rightarrow b$ is obtained via

```julia
function get_domainfunction(σ, c)
    blur(x,y) = 1.0/(2.0*π*σ^2.0)*exp(-(x^2.0 + y^2.0)/(2.0*σ^2.0))
    
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        x₁, y₁ = q[1], q[2]

        y = mapfoldl(+, 1:length(x)) do i
            x₀, y₀ = s[i][1], s[i][2]   
            Δx, Δy = abs(x₁-x₀), abs(y₁-y₀)
            tmp = (Δx < c) && (Δy < c) ? blur(Δx,Δy) * x[i] : 0.0
        end
    end
end

f = get_domainfunction(2.0, 4)
```

Note that the function f is specialized for a particular σ and c value. The design matrix is then obtained via

```julia
s = @> [(i,j) for i = 0:20, j = 0:20] reshape(:)
q = s
A = designmatrix(s, q, f) 
```

Finally, the blurred image is computed via

```julia
using Lazy

img = @> rand(21,21) reshape(:)
b1 = A*img
```

The design matrix is $n^2 \times n^2$, here 441 elements. The image is flattened to a 1D array and the blurred image is obtained via matrix multiplication. 

Alternatively, this package provides the higher order function [forwardmodel](@ref), which performs the equivalent calculation using the specialized function x. 

```julia
function forwardmodel(
    s::AbstractVector{T1},
    x::AbstractVector{T2},
    q::AbstractVector{T3},
    f::Function,
)::AbstractArray where {T1<:Any,T2<:Number,T3<:Any}
    return @>> (@_ map(Domain(s, x, _), q)) map(f)
end
```

for example

```
b1 = A*img
b2 = forwardmodel(s, img, q, f) 
```

and b1 and b2 are approximately equal.

#### Notes

1.  The returned function $f$ is specialized for a specific value $\sigma$. In general, the forward model may depend on a substantial number of hyper parameters.

2.  The setpoints $[s]$ and query points $[q]$ are interpreted by the function $f$. They therefore can by of any valid type.

3.  The types if $[x]$ and $[b]$ need not be the same, although both must be a subtype of Number.

4.  Unsurprisingly the matrix operation is much faster than the forwardmodel.

5.  The matrix A can be used in principle to blur and deblur images. You can try it with tiny images. Unfortunately, even a small 200x200 pixel image produces a 40000x40000 matrix, which is too large for the algorithms implemented in this package.

6.  The matrix A produced by the function designmatrix is by default a dense matrix. In practice the blur matrix of a well-ordered image is a Toeplitz matrix that can be stored in sparse format. The generic function cannot account for this.

7.  The computation time using designmatrix is slower (or much slower for large images) than the explicit discretization of the kernel function.

8.  Finally, the advantage of the designmatrix approach is to allow for easier conceptualization of the problem using type signatures and a declarative programming interface. This allows for factoring out some of the mathematical details of the discretization problem. It also immediately allows to apply the algorithm to non-square images nor does it rely on a particular sorting of the pixels to construct the matrix.

### Example 2

#### Problem Setup

Consider a population of particles suspended in air. The particle size distribution is represented in 8 size channels between 100 and 1000 nm. The populations light scattering and light absorption properties are measured at 3 wavelength. The properties are the convolution between the particle size distribution and the optical properties determined by Mie theory.

$\beta=\int_{D_{p1}}^{D_{p2}}Q\frac{dN}{dD_{p}}dD_{p}$

where $Q$ is either the scattering or absorption cross section of the particle that can be computed from [Mie theory](https://en.wikipedia.org/wiki/Mie_scattering), represented through a function ```Mie(ref,Dp,λ)```, which is a function of the complex refractive index of the particle ($ref$), particle diameter ($D_{p}$), and wavelength ($\lambda$). Integration is performed over the size distribution $\frac{dN}{dD_{p}}$ and the interval $[D_{p1},D_{p2}]$. In this example, the domain $[s]$ comprises the 8 particle diameters

```@example
include("size_functions.jl") # hide 
Dp, N = lognormal([[100.0, 500.0, 1.4]], d1 = 100.0, d2 = 1000.0, bins = 8) # hide

s = zip(["Dp$i" for i = 1:length(Dp)],Dp) |> collect
```

The setpoints $[s]$ are an array of tuples with labeled diameter and midpoint diameters in units of [nm]. 

The query points are the scattering and absorption cross sections measured at three different wavelength

```@example
λs = [300e-9, 500e-9, 900e-9] # hide
q1 = zip(["βs$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect #hide
q2 = zip(["βa$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect # hide

q = [q1; q2] 
```

Thus the query points $[q]$ are an array of tuples with labels and wavelengths in units of [m]. Note that $\beta_s$ denotes scattering and $\beta_a$ the absorption cross section of the aerosol.

The design matrix is obtained via

```julia
function get_domainfunction(ref)   
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        Dp, λ = (@_ map(_[2], s)), q[2]
       
        Q = @match q[1][1:3] begin 
            "βs" => @_ map(Qsca(π*Dp[_]*1e-9/λ, ref), 1:length(Dp))
            "βa" => @_ map(Qabs(π*Dp[_]*1e-9/λ, ref), 1:length(Dp))
            _ => trow("error")
        end
        mapreduce(+, 1:length(Dp)) do i
            π/4.0 *(x[i]*1e6)*Q[i]*(Dp[i]*1e-6)^2.0*1e6
        end
    end
end

f = get_domainfunction(complex(1.6, 0.01))
```

note that ```Qsca``` and ```Qabs``` are functions that return the scattering and absorption cross section from Mie theory for a given size parameter and refractive index. The domain function f is specialized for the refractive index $n = 1.6 + 0.01i". 

The matrix is obtained via

```
Dp, N = lognormal([[100.0, 500.0, 1.4]], d1 = 100.0, d2 = 1000.0, bins = 8)

A = designmatrix(s, q, g) 
b = A*N 
```

The matrix A is 6x8. N are the number concentrations of at each size (length 8). A*N produces 6 measurements corresponding to three $\beta_s$ and three $\beta_a$ values. 

#### Notes

1.  In this example 8 sizes are mapped to 6 observations. The setpoint $[s]$ and query $[q]$ domains are distinctly different in type.

2.  The input and output tuples annotate data. The annotations are interpreted by the domain function to decide which value to compute (scattering or absorption).

3.  This "toy" example illustrates the advantage of the domainmatrix discretization method. Only a few lines of declarative code are needed to compute converted to the domain matrix, even though the underlying model is reasonably complex. There is need to explicitly write out the equations to discretize the domain.

### Example 3

This example is the implementation of the Baart (1982) kernel shown in the beginning.

$K(q,s)=\exp(q\cos(s))$

In this example, the domain $[s]$ comprises the integration from $[0..\pi]$

```julia
s = range(0, stop = π, length = 120) 
q = range(0, stop = π/2, length = 120) 
```

The forward model domain function is 

```julia
function get_domainfunction(K)   
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        Δt = (maximum(s) - minimum(s))./length(x)
        y = @_ mapfoldl(K(q, (_-0.5)*Δt) * x[_] * Δt, +, 1:length(x)) 
    end
end
```

The design matrix is obtained via

```julia
f = get_domainfunction((x,y) -> exp(x*cos(y)))
A = designmatrix(s, q, f) 
```

Again, in this  the solution is knonw. Evaluating $\mathrm{b}=\mathrm{\mathbf{A}x}$ with $\mathrm{\mathrm{x}}=sin(s)$ approximately yields $2sin(q)/q$. 

```julia
x = sin.(s)
b1 = A*x 
b2 = 2.0.*sinh.(q)./q
```

And again b1 and b2 are close. Also A computed using the designmatrix function is the same than the explicit integration by quadrature. 


## Benchmarks

Systems up to a 1000 equations are unproblematic. The setup for much larger system slows down due to the ``\approx O(n^2)`` (or worse) time complexity of the SVD and generalized SVD factorization of the design matrix. Larger systems require switching to SVD free algorithms, which are currently not supported by this package. 

```@example
using RegularizationTools, TimerOutputs, MatrixDepot

to = TimerOutput()

function benchmark(n)
    r = mdopen("shaw", n, false)
    A, x = r.A, r.x
    y = A * x
    b = y + 0.05y .* randn(n)
    Ψ = setupRegularizationProblem(A, 2)
    for i = 1:1
        @timeit to "Setup  (n = $n)" setupRegularizationProblem(A, 2)
        @timeit to "Invert (n = $n)" solve(Ψ, b)
    end
end

map(benchmark, [10, 100, 1000])
show(to)
```
