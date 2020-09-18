# Algorithms to compute the optimal regularization parameter λ


function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector;
    alg = :gcv_tr,
    λ₁ = 0.0001,
    λ₂ = 1000.0,
)
    b̄ = @pipe b |> to_standard_form(Ψ, _)
    L1, L2, κ = Lcurve_functions(Ψ, b̄)

    λ = @match alg begin
        :gcv_tr => optimize(λ -> gcv_tr(Ψ, b̄, λ), λ₁, λ₂) |> Optim.minimizer
        :gcv_svd => optimize(λ -> gcv_svd(Ψ, b̄, λ), λ₁, λ₂) |> Optim.minimizer
        :L_curve => optimize(λ -> 1.0 - κ(λ), λ₁, λ₂) |> Optim.minimizer
        _ => throw("Unknown Algorithm")
    end

    x̄ = solve(Ψ, b̄, λ)
    x = @pipe x̄ |> to_general_form(Ψ, b, _)

    return clean(x)
end


function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector,
    x₀::AbstractVector;
    alg = :L_curve,
    λ₁ = 0.0001,
    λ₂ = 1000.0,
)
    b̄ = @pipe b |> to_standard_form(Ψ, _)
    x̄₀ = Ψ.L * x₀
    L1, L2, κ = Lcurve_functions(Ψ, b̄, x̄₀)

    λ = @match alg begin
        :gcv_tr => optimize(λ -> gcv_tr(Ψ, b̄, x̄₀, λ), λ₁, λ₂) |> Optim.minimizer
        :gcv_svd => optimize(λ -> gcv_svd(Ψ, b̄, x̄₀, λ), λ₁, λ₂) |> Optim.minimizer
        :L_curve => optimize(λ -> 1.0 - κ(λ), λ₁, λ₂) |> Optim.minimizer
        _ => throw("Unknown Algorithm")
    end

    x̄ = solve(Ψ, b̄, Ψ.L * x₀, λ)
    x = @pipe x̄ |> to_general_form(Ψ, b, _)

    return clean(x)
end

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


function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector)
    x̄(λ::AbstractFloat) = solve(Ψ, b̄, x̄₀, λ)
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * (x̄(λ) - x̄₀))
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

function Lcurve_functions(Ψ::RegularizationProblem, b̄::AbstractVector)
    x̄(λ::AbstractFloat) = solve(Ψ, b̄, λ)
    L1(λ::AbstractFloat) = norm(Ψ.Ā * x̄(λ) - b̄)
    L2(λ::AbstractFloat) = norm(Ψ.Iₚ * x̄(λ))
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
