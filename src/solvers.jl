@doc raw"""
    Γ(m::Int, order::Int)

Return the smoothing matrix L for Tikhonov regularization of a system of size `m`.  Order
can be 0, 1, ..., n. Code to generate matrix is based on the suggestion posted in Issue #7,
which was inspired by Eilers, P. H. C. (2003). Analytical Chemistry, 75(14),
3631–3636. (Supporting Information).

```julia
L = Γ(m, 1)
```
"""
@memoize function Γ(m::Int, order::Int)
    if order == 0
        return Array{Float64}(LinearAlgebra.I, (m, m))
    end

    return diff(Γ(m, order-1), dims=1)
end

function zot(A::AbstractMatrix, λ::AbstractFloat)
    a = deepcopy(A)
    n = size(A'A, 1)
    for i = 1:n
        @inbounds a[i, i] += λ
    end
    return a
end

@doc raw"""
    to_standard_form(Ψ::RegularizationProblem, b::AbstractVector)

Converts vector b to standard form using (Hansen, 1998)

Example Usage (Regular Syntax)
```julia
b̄ = to_standard_form(Ψ, b)
```

Example Usage (Lazy Syntax)
```julia
b̄ = @>> b to_standard_form(Ψ)
```
"""
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector) = b - Ψ.A*Ψ.K₀T⁻¹H₀ᵀ*b

@doc raw"""
    to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector)

Converts vector b and x₀ to standard form using (Hansen, 1998)

Example Usage (Regular Syntax)
```julia
b̄ = to_standard_form(Ψ, b, x₀)
```
"""
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector) =
    b - Ψ.A*Ψ.K₀T⁻¹H₀ᵀ*b, Ψ.L * x₀

@doc raw"""
    to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector)

Converts solution ``\bar {\rm x}`` computed in standard form back to general form 
``{\rm x}`` using (Hansen, 1998).

```math
{\rm x}={\rm {\bf L^{+}_A}\bar{x} + x\_0}
```

where the matrices and vectors are defined in [RegularizationProblem](@ref)

Example Usage (Regular Syntax)
```julia
x = to_general_form(Ψ, b, x̄) 
```

Example Usage (Lazy Syntax)
```julia
x = @>> x̄ to_general_form(Ψ, b) 
```
"""
function to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector) 
	x = Ψ.L⁺ * x̄  + Ψ.K₀T⁻¹H₀ᵀ*(b - Ψ.A*Ψ.L⁺*x̄ ) 
end

@doc raw"""
    solve(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat)

Compute the Tikhonov solution for problem Ψ in standard form for regularization parameter λ
and using zero as initial guess. Returns a vector ``\rm {\bar x}_\lambda``. 

```math
{\rm x_{\lambda}}=\left({\rm {\bf \bar A}^{T}}{\rm {\bf \bar A}}+\lambda^{2}{\rm {\bf I}}\right)^{-1} 
{\rm {\bf {\bar A}}^{T}}{\rm {\bar b}} 
```
Example Usage (Standard Syntax)
```julia
# A is a Matrix and b is a response vector. 
Ψ = setupRegularizationProblem(A, 1)     # Setup problem
b̄ = to_standard_form(Ψ, b)               # Convert to standard form
x̄ = solve(A, b̄, 0.5)                     # Solve the equation
x = to_general_form(Ψ, b, x̄)             # Convert back to general form
```

Example Usage (Lazy Syntax)
```julia
# A is a Matrix and b is a response vector. 
Ψ = setupRegularizationProblem(A, 1)     # Setup problem
b̄ = @>> b to_standard_form(Ψ)            # Convert to standard form
x̄ = solve(A, b̄, 0.5)                     # Solve the equation
x = @>> x̄ to_general_form(Ψ, b)          # Convert back to general form
```
"""
solve(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat) = try
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄)
catch
     zot(Ψ.ĀĀ, λ^2.0) \ (Ψ.Ā' * b̄)
end

@doc raw"""
    solve(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector, λ::AbstractFloat)

Compute the Tikhonov solution for problem Ψ in standard form for regularization parameter λ
and using x̄₀ as initial guess. 

```math
{\rm x_{\lambda}}=\left({\rm {\bf \bar A}^{T}}{\rm {\bf \bar A}}+\lambda^{2}{\rm {\bf I}}\right)^{-1} 
\left({\rm {\bf {\bar A}}^{T}}{\rm {\bar b}} + \lambda^2 {\rm {\bar x}}_0 \right)
```

Example Usage (Standard Syntax)
```julia
# A is a Matrix and b is a response vector. 
Ψ = setupRegularizationProblem(A, 2)     # Setup problem
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)       # Convert to standard form
x̄ = solve(A, b̄, x̄₀, 0.5)                 # Solve the equation
x = to_general_form(Ψ, b, x̄)             # Convert back to general form
```
"""
solve(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector, λ::AbstractFloat) = try
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄ + λ^2.0 * x̄₀)
catch
    zot(Ψ.ĀĀ, λ^2.0) \ (Ψ.Ā' * b̄ + λ^2.0 * x̄₀)
end

@doc raw"""
    function solve(
        Ψ::RegularizationProblem,
        b::AbstractVector;
        alg = :gcv_svd,
        method = Brent(),
        λ₁ = 0.0001,
        λ₂ = 1000.0,
        λ₀ = 1.0,
        kwargs...
    )

Find the optimum regularization parameter λ using the algorithm `alg` and optimization
`method`. Choices for algorithms are
```    
    :gcv_tr - generalized cross validation using the trace formulation (slow)
    :gcv_svd - generalized cross validation using the SVD decomposition (fast)
    :L_curve - L-curve algorithm 
```
Optimization methods are those available in the Optim package. Methods that optimize on
bounds (`Brent()` and `GoldenSection()`) will use [λ₁, λ₂] for bounds. Other methods will
use λ₀ for the initial guess. The methods `Brent()` (default) and `NelderMead()` are
recommended. Any additional keyword arguments will be passed on to the optimization method.

!!! tip
    The gcv\_svd algorithm is fastest and most stable. The L\_curve algorithn is sensitive to the upper 
    and lower bound. Specify narrow upper and lower bounds to obtain a good solution.

The solve function takes the original data, converts it to standard form, performs the search
within the specified bounds and returns a [RegularizatedSolution](@ref)

Example Usage (Standard Syntax)
```julia
# A is a Matrix and b is a response vector. 
Ψ = setupRegularizationProblem(A, 2)     # Setup problem
sol = solve(Ψ, b)                        # Solve it
```

Example Usage (Lazy Syntax)
```julia
# A is a Matrix and b is a response vector. 
sol = @> setupRegularizationProblem(A, 1) solve(b)
```
"""
function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector;
    alg = :gcv_svd,
    method = Brent(),
    λ₁ = 0.0001,
    λ₂ = 1000.0,
    λ₀ = 1.0,
    kwargs...
)
    b̄ = @>> b to_standard_form(Ψ)
    if alg==:L_curve
        L1, L2, κ = Lcurve_functions(Ψ, b̄)
        κ_log10λ(log10λ::AbstractFloat) = κ(10^log10λ)
    end

    optfunc(x) = @match alg begin
        :gcv_tr => gcv_tr_log10λ(Ψ, b̄, x)
        :gcv_svd => gcv_svd_log10λ(Ψ, b̄, x)
        :L_curve => 1.0 - κ_log10λ(x)
        _ => throw("Unknown algorithm, use :gcv_tr, :gcv_svd, or :L_curve")
    end
    λ, solution = _solve_λ(optfunc, method, λ₁, λ₂, λ₀; kwargs...)
    
    x̄ = solve(Ψ, b̄, λ)
    x = @>> x̄ to_general_form(Ψ, b) 
     
    return RegularizedSolution(x, λ, solution)
end

@doc raw"""
    function solve(
        Ψ::RegularizationProblem,
        b::AbstractVector,
        x₀::AbstractVector;
        alg = :gcv_svd,
        method = Brent(),
        λ₁ = 0.0001,
        λ₂ = 1000.0,
        λ₀ = 1.0,
        kwargs...
    )

Same as above, but includes an initial guess x₀. Example Usage (Lazy Syntax)
```julia
# A is a Matrix and b is a response vector. 
sol = @> setupRegularizationProblem(A, 1) solve(b, x₀, alg = :L_curve, λ₂ = 10.0)
```
"""
function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector,
    x₀::AbstractVector;
    alg = :gcv_svd,
    method = Brent(),
    λ₁ = 0.0001,
    λ₂ = 1000.0,
    λ₀ = 1.0,
    kwargs...
)
    b̄ = @>> b to_standard_form(Ψ)
    x̄₀ = Ψ.L * x₀
    if alg==:L_curve
        L1, L2, κ = Lcurve_functions(Ψ, b̄, x̄₀)
        κ_log10λ(log10λ::AbstractFloat) = κ(10^log10λ)
    end

    optfunc(x) = @match alg begin
        :gcv_tr => gcv_tr_log10λ(Ψ, b̄, x̄₀, x)
        :gcv_svd => gcv_svd_log10λ(Ψ, b̄, x̄₀, x)
        :L_curve => 1.0 - κ_log10λ(x)
        _ => throw("Unknown algorithm, use :gcv_tr, :gcv_svd, or :L_curve")
    end
    λ, solution = _solve_λ(optfunc, method, λ₁, λ₂, λ₀; kwargs...)
    
    x̄ = solve(Ψ, b̄, x̄₀, λ)
    x = @>> x̄ to_general_form(Ψ, b)   

    return RegularizedSolution(x, λ, solution)
end

"""
Internal function that performs the operations of calculating the regularization parameter
that minimizes the specified objective function.
"""
function _solve_λ(optfunc::Function, method::Any, λ₁::AbstractFloat, λ₂::AbstractFloat,
                 λ₀::AbstractFloat; kwargs...)
    if method in [Brent(), GoldenSection()] # univariate methods with bounds
        solution = optimize(optfunc, log10(λ₁), log10(λ₂), method; kwargs...)
        log10λ = @> solution Optim.minimizer
    else # methods that use an intitial guess; gradient-free and gradient methods via
        # default finite-difference work Ok, but `autodiff=:forward` is not working
        solution = optimize(x->optfunc(first(x)), [log10(λ₀)], method,
                            Optim.Options(; kwargs...))
        # alternative method call, but cautioned against in the docs
        #solution = optimize(x->optfunc(first(x)), [log10(λ₀)]; method=method, kwargs...)
        log10λ = @> solution Optim.minimizer first
    end
    λ = 10^log10λ
    return λ, solution
end

@doc raw"""
    function solve(
        Ψ::RegularizationProblem, 
        b::AbstractVector,
        lower::AbstractVector, 
        upper::AbstractVector;
        kwargs...
    )

Constraint minimization of [RegularizationProblem](@ref) Ψ, with observations b and
upper and lower bounds for each xᵢ.

The function computes the algebraic solution using ```solve(Ψ, b; kwargs...)```, truncates the
solution at the upper and lower bounds and uses this solution as initial condition for
the minimization problem using a box-constrained optimizer. The returned solution
is using the regularization parameter λ obtained from the algebraic solution.
"""
function solve(
    Ψ::RegularizationProblem, 
    b::AbstractVector,
    lower::AbstractVector, 
    upper::AbstractVector;
    kwargs...
)
    return solve_numeric(Ψ, b, solve(Ψ, b; kwargs...), lower, upper)
end

@doc raw"""
    function solve(
        Ψ::RegularizationProblem, 
        b::AbstractVector,
        x₀::AbstractVector,
        lower::AbstractVector, 
        upper::AbstractVector;
        kwargs...
    )


Constraint minimization of [RegularizationProblem](@ref) Ψ, with observations b, intial 
guess x₀ and upper and lower bounds for each xᵢ.

The function computes the algebraic solution using ```solve(Ψ, b; kwargs...)```, truncates the
solution at the upper and lower bounds and uses this solution as initial condition for
the minimization problem using a box-constrained optimizer. The returned solution
is using the regularization parameter λ obtained from the algebraic solution.
"""    
function solve(
    Ψ::RegularizationProblem, 
    b::AbstractVector,
    x₀::AbstractVector,
    lower::AbstractVector, 
    upper::AbstractVector;
    kwargs...
)
    return solve_numeric(Ψ, b, solve(Ψ, b, x₀; kwargs...), lower, upper)
end


function solve_numeric(
    Ψ::RegularizationProblem, 
    b::AbstractVector, 
    xλ::RegularizedSolution,
    lower::AbstractVector,
    upper::AbstractVector
)
    λ = xλ.λ  
    xᵢ = clamp.(xλ.x, lower, upper)
    
    A, L = Ψ.A, Ψ.L
    
    function f(x)
        norm(A*x - b)^2.0 + λ^2.0 * norm(L*x)^2.0
    end
    
    function g!(G, x)
        G .= 2.0 .* (A'*(A*x - b) .+ λ^2.0 .* (L'*(L*x)))
    end
    
    r = Optim.optimize(
        f,
        g!,
        lower,
        upper,
        xᵢ,
        Optim.Fminbox(),
        Optim.Options(iterations = 10000),
    )
    return RegularizedSolution(Optim.minimizer(r), λ, r)
end


@doc raw"""
    setupRegularizationProblem(A::AbstractMatrix, order::Int)

Precompute matrices to initialize Reguluarization Problem based on design matrix A and 
zeroth, first, or second order difference operator. See Hanson (1998) and source code
for details.

Example Usage
```julia
Ψ = setupRegularizationProblem(A, 0) # zeroth order problem
Ψ = setupRegularizationProblem(A, 2) # second order problem
```
"""
setupRegularizationProblem(A::AbstractMatrix, order::Int) = 
    setupRegularizationProblem(A, Γ(size(A,2), order))

@doc raw"""
    setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)

Precompute matrices to initialize Reguluarization Problem based on design matrix and 
Tikhonov smoothing matrix. See Hansen (1998, Eq. 2.35)

Example Usage
```julia
Ψ = setupRegularizationProblem(A, L) 
```
"""
@memoize function setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)
    p, n = size(L)
    Iₙ = Matrix{Float64}(I, n, n) 
    Iₚ = Matrix{Float64}(I, p, p)
	Q,R = qr(L')
	K0 = Q[:,p+1:end]
	H,T = qr(A*K0)
	H0 = H[:,1:n-p]
	K₀T⁻¹H₀ᵀ = K0*T^-1*H0'
	L⁺ = pinv(L)
	L⁺ₐ = (Iₙ - K₀T⁻¹H₀ᵀ*A)*L⁺
    Ā = A*L⁺ₐ

    RegularizationProblem(
        Ā,
        A,
        L,
        Ā'Ā,
        Ā',
        svd(Ā),
        Iₙ,
        Iₚ,
        L⁺ₐ,
		L⁺,
		K₀T⁻¹H₀ᵀ	
    )
end
