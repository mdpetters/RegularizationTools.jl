@doc raw"""
    gcv_tr(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat)

Compute the [Generalized Cross Validation](@ref) using the trace term. Requires that the 
vector b̄ is in standard form.

```math
V(\lambda)=\frac{n\left\lVert ({\bf {\rm {\bf {\bf I}-}{\bf \bar {A}_{\lambda}})
{\bar{\rm b}}}}\right\rVert _{2}^{2}}{tr({\rm {\bf I}-{\rm {\bar {\bf A}_{\lambda}})}^{2}}}
```

Example Usage
```julia
using Underscores

Ψ = setupRegularizationProblem(A, 1)           # Setup problem
b̄ = to_standard_form(Ψ, b)                     # Convert to standard form
Vλ = gcv_tr(Ψ, b̄, 0.1)                         # V(λ) single λ value
Vλ = @_ map(gcv_tr(Ψ, b̄, _), [0.1, 1.0, 10.0]) # V(λ) for array of λ
```
"""
function gcv_tr(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat)
    n = size(Ψ.Ā, 1)
    Aλ = Ψ.Ā * inv(Ψ.ĀĀ + λ^2.0 * Ψ.Iₚ) * Ψ.Ā'
    return n * norm((Ψ.Iₙ - Aλ) * b̄)^2.0 / tr(Ψ.Iₙ - Aλ)^2.0
end


@doc raw"""
    gcv_tr(
        Ψ::RegularizationProblem,
        b̄::AbstractVector,
        x̄₀::AbstractVector,
        λ::AbstractFloat,
    )

Compute the [Generalized Cross Validation](@ref) using the trace term and intial guess. 
Requires that the vectors b̄ and x̄₀ are in standard form.

```math
V(\lambda)=\frac{n\left\lVert {\bf {\rm {\bf \bar{A}}{\rm \bar{x}{}_{\lambda}}-
{\rm \bar{b}}}}\right\rVert _{2}^{2}}{tr({\rm {\bf I}-{\rm {\bar {\bf A}_{\lambda}})}^{2}}}
```
Example Usage
```julia
using Underscores

Ψ = setupRegularizationProblem(A, 1)               # Setup problem
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)                 # Convert to standard form
Vλ = gcv_tr(Ψ, b̄, x̄₀, 0.1)                         # V(λ) single λ value
Vλ = @_ map(gcv_tr(Ψ, b̄, x̄₀, _), [0.1, 1.0, 10.0]) # V(λ) for array of λ
```
"""
function gcv_tr(
    Ψ::RegularizationProblem,
    b̄::AbstractVector,
    x̄₀::AbstractVector,
    λ::AbstractFloat,
)
    n = size(Ψ.Ā, 1)
    x̄(λ::AbstractFloat) = solve(Ψ, b̄, x̄₀, λ)
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    Aλ = Ψ.Ā * inv(Ψ.ĀĀ + λ^2.0 * Ψ.Iₚ) * Ψ.Ā'
    return n * L1(λ)^2.0 / tr(Ψ.Iₙ - Aλ)^2.0
end

@doc raw"""
    gcv_svd(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat)
   
Compute the [Generalized Cross Validation](@ref) using the trace term using the SVD 
algorithm. Requires that the vector b̄ is in standard form.

Example Usage
```julia
using Underscores

Ψ = setupRegularizationProblem(A, 1)            # Setup problem
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)              # Convert to standard form
Vλ = gcv_svd(Ψ, b̄, x̄₀, 0.1)                     # V(λ) single λ value
Vλ = @_ map(gcv_svd(Ψ, b̄, _), [0.1, 1.0, 10.0]) # V(λ) for array of λ
```
"""
function gcv_svd(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat)
    F = Ψ.F̄
    n, p = size(F.U)
    z = F.U' * b̄
    nλ = λ^2.0

    s1 = 0.0
    for j = 1:p
        s1 += (nλ / (F.S[j] * F.S[j] + nλ))^2.0 * z[j] * z[j]
    end

    s2 = 0.0
    for j = 1:p
        s2 += nλ / (F.S[j] * F.S[j] + nλ)
    end

    return n * (norm(b̄)^2.0 - norm(z)^2.0 + s1) / (n - p + s2)^2.0
end

@doc raw"""
    gcv_svd(
        Ψ::RegularizationProblem,
        b̄::AbstractVector,
        x̄₀::AbstractVector,
        λ::AbstractFloat,
    )

Compute the [Generalized Cross Validation](@ref) using the SVD algorithm and intial guess. 
Requires that the vectors b̄ and x̄₀ are in standard form.

Example Usage
```julia
using Underscores

Ψ = setupRegularizationProblem(A, 1)               # Setup problem
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)                 # Convert to standard form
Vλ = gcv_tr(Ψ, b̄, x̄₀, 0.1)                         # V(λ) single λ value
Vλ = @_ map(gcv_tr(Ψ, b̄, x̄₀, _), [0.1, 1.0, 10.0]) # V(λ) for array of λ
```
"""
function gcv_svd(
    Ψ::RegularizationProblem,
    b̄::AbstractVector,
    x̄₀::AbstractVector,
    λ::AbstractFloat,
)
    x̄(λ::AbstractFloat) = solve(Ψ, b̄, x̄₀, λ)
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)

    F = Ψ.F̄
    n, p = size(F.U)
    nλ = λ^2.0

    s2 = 0.0
    for j = 1:p
        s2 += nλ / (F.S[j] * F.S[j] + nλ)
    end

    return n * L1(λ)^2.0 / (n - p + s2)^2.0
end

function curvature_functions(x̄::Function, L1::Function, L2::Function)
    η⁰(λ::AbstractFloat) = (log.(L2.(λ) .^ 2.0))[1]
    ρ⁰(λ::AbstractFloat) = (log.(L1.(λ) .^ 2.0))[1]
    ηᵖ(λ::AbstractFloat) = (derivative(η⁰, λ))[1]

    function κ(λ::AbstractFloat)
        nᵖ = ηᵖ(λ)
        η = η⁰(λ)
        ρ = ρ⁰(λ)
        return 1.0 -
            2.0 * η * ρ * (λ^2.0 * nᵖ * ρ + 2.0 * λ * η * ρ + λ^4.0 * η * nᵖ) /
            (nᵖ * (λ^2.0 * η^2.0 + ρ^2.0)^1.5)
    end

    return L1, L2, κ
end
@doc raw"""
    Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector) 

Compute the L-curve functions to evaluate the norms L1, L2, and the curvature κ. 
Requires that the vectors b̄ is in standard form.

Example Usage

```julia
Ψ = setupRegularizationProblem(A, 1)
b̄ = to_standard_form(Ψ, b)
L1norm, L2norm, κ = Lcurve_functions(Ψ, b̄)

L1norm.([0.1, 1.0, 10.0])    # L1 norm for λ's
L2norm.([0.1, 1.0, 10.0])    # L2 norm for λ's
κ.([0.1, 1.0, 10.0])         # L-curve curvature for λ's
```
"""
function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector) 
    x̄(λ::AbstractFloat)  = solve(Ψ, b̄, λ) 
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * x̄(λ))
    return curvature_functions(x̄, L1, L2)
end

@doc raw"""
    Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector)

Compute the L-curve functions to evaluate the norms L1, L2, and the curvature κ. 
Requires that the vectors b̄ and x̄₀ are in standard form.

Example Usage
```julia
Ψ = setupRegularizationProblem(A, 1)
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)                 
L1norm, L2norm, κ = Lcurve_functions(Ψ, b̄, x̄₀)

L1norm.([0.1, 1.0, 10.0])    # L1 norm for λ's
L2norm.([0.1, 1.0, 10.0])    # L2 norm for λ's
κ.([0.1, 1.0, 10.0])         # L-curve curvature for λ's
```
"""
function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector)
    x̄(λ::AbstractFloat)  = solve(Ψ, b̄, x̄₀, λ) 
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * (x̄(λ) - x̄₀))
    return curvature_functions(x̄, L1, L2) 
end
