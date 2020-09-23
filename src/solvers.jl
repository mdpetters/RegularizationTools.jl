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

```math
{\rm \bar{b}={\rm {\bf H}_q^{T}{\rm b}}}
```

where ``\rm {\bf H}_q^T`` is defined in [RegularizationProblem](@ref)

Example Usage (Regular Syntax)
```julia
b̄ = to_standard_form(Ψ, b)
```

Example Usage (Lazy Syntax)
```julia
b̄ = @>> b to_standard_form(Ψ)
```
"""
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector) = Ψ.Hqᵀ * b

@doc raw"""
    to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector)

Converts vector b and x₀ to standard form using (Hansen, 1998)

```math
{\rm \bar{b}={\rm {\bf H}_q^{T}{\rm b}}}
```

```math
{\rm \bar{x}_{0}={\rm {\bf H}_qx_{0}}}
```

where ``\rm {\bf H}_q^T`` and ``\rm {\bf L}`` are defined in [RegularizationProblem](@ref)

Example Usage (Regular Syntax)
```julia
b̄ = to_standard_form(Ψ, b, x₀)
```
"""
to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, x₀::AbstractVector) =
    Ψ.Hqᵀ * b, Ψ.L * x₀

@doc raw"""
    to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector)

Converts solution ``\bar {\rm x}`` computed in standard form back to general form ``{\rm x}`` using (Hansen, 1998)

```math
{\rm x}={\rm {\bf L^{+}}\bar{x}+K_{0}T_{0}^{-1}{\rm {\bf {H}_{\rm 0}^{{\rm T}}}({\rm b-{\rm {\bf A}{\rm {\bf L^{{\rm +}}{\rm \bar{x})}}}}}}}
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
to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector) =
    Ψ.L⁺ * x̄ + Ψ.K0 * Ψ.T0^(-1) * Ψ.H0ᵀ * (b - Ψ.A * Ψ.L⁺ * x̄)

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
Tikhonov smoothing matrix. 

(1) Pseudo-inverse of L (used in standard form transformation)

```math
{\bf L^{+}}=({\rm {\bf L}^{T}}{\rm {\rm {\bf L}}})^{-1}{\rm {\rm {\bf L}}}
```

(2) QR factorization of L' (used in standard form transformation)
```math
{\rm {\bf L}^{T}}={\rm {\bf K}{\bf R}}=\left({\rm {\rm {\bf K}}_{p}},{\rm {\bf K}}_{0}\right)\left(\begin{array}{c}
{\rm R_{p}}\\
0
\end{array}\right)
```

(3) QR factorization of AK (used in standard form transformation)
```math
{\rm {\bf A}{\rm {\bf K}}_{0}}={\rm {\bf H}{\bf T}}=\left({\rm {\rm {\bf H}}_{0}},{\rm {\bf H}}_{{\rm q}}\right)\left(\begin{array}{c}
{\rm T_{0}}\\
0
\end{array}\right)
```

(4) Computation of Ā (used in standard form transformation)
```math
{\bf \bar{A}}={\rm {\bf H}}_{{\rm q}}^{{\rm T}}{\rm {\bf A}{\bf L^{+}}}
```

(5) Computation and storage of Ā'Ā (used in Tikhonov solution in standard form)
```math
{\rm {\bf \bar{A}}}{\rm {\bf \bar{A}}}={\rm {\bf \bar{A}}}^{{\rm T}}{\rm {\bf \bar{A}}}
```

Example Usage
```julia
Ψ = setupRegularizationProblem(A, L) 
```
"""
function setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)
    n, p = size(L')
    Iₙ = Matrix{Float64}(I, n, n) 
    Iₚ = Matrix{Float64}(I, p, p)
    L⁺ = convert(Matrix, (L' * L)^(-1) * L')
    K, R = qr(L')
    K = convert(Matrix,K)
    Kp = convert(Matrix,(view(K,:,1:n)))
    K0 = convert(Vector, (view(K,:,n)))
    H, T = qr(hcat(A * K0))
    H = H*Iₙ
    H0ᵀ = convert(Matrix,(view(H,:,1))')
    Hqᵀ = convert(Matrix,(view(H, :, 1:n))')
    T0 = T
    Ā = Hqᵀ[:,:] * A * L⁺

    RegularizationProblem(
        Ā,
        A,
        L,
        Ā'Ā,
        Ā',
        svd(Ā),
        Iₙ,
        Iₚ,
        L⁺,
        Hqᵀ,
        H0ᵀ,
        T0,
        K0,
    )
end
