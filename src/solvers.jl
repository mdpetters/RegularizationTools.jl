@doc raw"""
    Γ(A::AbstractMatrix, order::Int)

Return the smoothing matrix L for zero, first and second order Tikhonov regularization 
based on the size of design matrix A. Order can be 0, 1 or 2.

```julia
L = Γ(A, 1)
```
"""
function Γ(A::AbstractMatrix, order::Int)
    n = size(A, 1)

    @match order begin
        0 => zot(n)
        1 => fot(n)
        2 => sot(n)
        _ => throw("Order not supported, select 0, 1, or 2")
    end
end

zot(n) = Matrix{Float64}(I, n, n)

function zot(A::AbstractMatrix, λ::AbstractFloat)
    a = deepcopy(A)
    n = size(a, 1)
    for i = 1:n
        @inbounds a[i, i] += λ
    end
    return a
end

function fot(n)
    L = zeros(n + 1, n)
    for i = 1:n
        @inbounds L[i, i] = -1
        @inbounds L[i+1, i] = 1
    end
    return L
end

function sot(n)
    L = zeros(n + 2, n)
    for i = 1:n
        @inbounds L[i, i] = 1
        @inbounds L[i+1, i] = -2
        @inbounds L[i+2, i] = 1
    end
    return L
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
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector) = b

@doc raw"""
    to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector)

Converts vector b and x₀ to standard form using (Hansen, 1998)

Example Usage (Regular Syntax)
```julia
b̄ = to_standard_form(Ψ, b, x₀)
```
"""
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector) =
    b, Ψ.L * x₀

@doc raw"""
    to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector)

Converts solution ``\bar {\rm x}`` computed in standard form back to general form 
``{\rm x}`` using (Hansen, 1998). Solution is truncated to regularized space, given
by the matrix L. If L is p × n and p < n, then only the solution 1:p is valid. The remaining 
parameters can be estiamted from the least-squares solution if needed.

```math
{\rm x}={\rm {\bf L^{+}_A}\bar{x}}
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
    n, p = size(Ψ.L⁺ₐ)
    x = Ψ.L⁺ₐ * x̄ 
    if p < n 
        return x[1:p]
    else 
        return x
    end 
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
solve(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat) =
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄)

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
solve(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector, λ::AbstractFloat) =
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄ + λ^2.0 * x̄₀)


@doc raw"""
    function solve(
        Ψ::RegularizationProblem,
        b::AbstractVector;
        alg = :gcv_svd,
        λ₁ = 0.0001,
        λ₂ = 1000.0,
    )

Find the optimum regularization parameter λ between [λ₁, λ₂] using the algorithm alg. Choices
for algorithms are
```    
    :gcv_tr - generalized cross validation using the trace formulation (slow)
    :gcv_svd - generalized cross validation using the SVD decomposition (fast)
    :L_curve - L-curve algorithm 
```

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
    λ₁ = 0.0001,
    λ₂ = 1000.0,
)
    b̄ = @>> b to_standard_form(Ψ)
    L1, L2, κ = Lcurve_functions(Ψ, b̄)

    solution = @match alg begin
        :gcv_tr  => @_ optimize(gcv_tr(Ψ, b̄, _), λ₁, λ₂) 
        :gcv_svd => @_ optimize(gcv_svd(Ψ, b̄, _), λ₁, λ₂) 
        :L_curve => @_ optimize(1.0 - κ(_), λ₁, λ₂) 
        _ => throw("Unknown algorithm, use :gcv_tr, :gcv_svd, or :L_curve")
    end

    λ = @> solution Optim.minimizer
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
        λ₁ = 0.0001,
        λ₂ = 1000.0,
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
    λ₁ = 0.0001,
    λ₂ = 1000.0,
)
    b̄ = @>> b to_standard_form(Ψ)
    x̄₀ = Ψ.L * x₀
    L1, L2, κ = Lcurve_functions(Ψ, b̄, x̄₀)

    solution = @match alg begin
        :gcv_tr  => @_ optimize(gcv_tr(Ψ, b̄, x̄₀, _), λ₁, λ₂) 
        :gcv_svd => @_ optimize(gcv_svd(Ψ, b̄, x̄₀, _), λ₁, λ₂)
        :L_curve => @_ optimize(1.0 - κ(_), λ₁, λ₂) 
        _ => throw("Unknown algorithm, use :gcv_tr, :gcv_svd, or :L_curve")
    end

    λ = @> solution Optim.minimizer
    x̄ = solve(Ψ, b̄, x̄₀, λ)
    x = @>> x̄ to_general_form(Ψ, b)   

    return RegularizedSolution(x, λ, solution)
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
the minimization problem using a Least Squares numerical solver. The returned solution
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
the minimization problem using a Least Squares numerical solver. The returned solution
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
    xᵢ = xλ.x
    xᵢ[xᵢ .< lower] .= lower[xᵢ .< lower]
    xᵢ[xᵢ .> upper] .= upper[xᵢ .> upper]
    
    n = length(b)
    LᵀL = Ψ.L'*Ψ.L
    
    function f!(out, x)
        out[1] = norm(Ψ.A*x - b)^2.0 + λ^2.0*norm(LᵀL*x)^2.0
    end
    
    function g!(out, x)
        ot = Ψ.A'*(Ψ.A*x - b) + λ^2.0*LᵀL*x
        [out[i] = 2.0*ot[i] for i = 1:n]
    end
    
    LLSQ = LeastSquaresProblem(
        x = xᵢ, 
        f! = f!, 
        g! = g!,
        output_length=n
    )
   
    r = optimize!(
        LLSQ, 
        Dogleg(LeastSquaresOptim.QR()), 
        lower = lower, 
        upper = upper,
        x_tol=1e-10
    ) 
    return return RegularizedSolution(r.minimizer, λ, r)
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
    setupRegularizationProblem(A, Γ(A, order))

@doc raw"""
    setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)

Precompute matrices to initialize Reguluarization Problem based on design matrix and 
Tikhonov smoothing matrix. See Hansen (1998, Eq. 2.35)

Example Usage
```julia
Ψ = setupRegularizationProblem(A, L) 
```
"""
function setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)
    p, n = size(L)
    Iₙ = Matrix{Float64}(I, n, n) 
    Iₚ = Matrix{Float64}(I, p, p)
    F = svd(A,L)
    U = F.U
    V = F.V
    Σ = Matrix(F.D1)
    M = Matrix(F.D2)
    X = inv((F.R0*F.Q'))
    L⁺ₐ = X*pinv(M)*V'
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
    )
end
