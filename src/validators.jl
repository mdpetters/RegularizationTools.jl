# Algorithms to compute the optimal regularization parameter λ


function gcv_tr(Ψ::RegularizationProblem, b::AbstractVector, λ::AbstractFloat)
    n = size(Ψ.Ā, 1)
    Aλ = Ψ.Ā * inv(Ψ.ĀĀ + λ^2.0 * Ψ.Iₚ) * Ψ.Ā'
    return n * norm((Ψ.Iₙ - Aλ) * b)^2.0 / tr(Ψ.Iₙ - Aλ)^2.0
end

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

# SVD based algorithm to find gcv estimate
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

function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector)
    x̄(λ::AbstractFloat)  = solve(Ψ, b̄, x̄₀, λ) 
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * (x̄(λ) - x̄₀))
    return curvature_functions(x̄, L1, L2) 
end


function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector) 
    x̄(λ::AbstractFloat)  = solve(Ψ, b̄, λ) 
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * x̄(λ))
    return curvature_functions(x̄, L1, L2)
end