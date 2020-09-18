@doc raw"""
    Γ(A::AbstractMatrix, order::Int)

Return the smoothing matrix L for zero, first and second order Tikhonov regularization 
based on the size of design matrix A.
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

to_standard_form(Ψ::RegularizationProblem, b::AbstractVector) = Ψ.Hqᵀ * b

to_standard_form(Ψ::RegularizationProblem, b::AbstractVector, bᵢ::AbstractVector) = 
    Ψ.Hqᵀ * b, L * bᵢ

to_general_form(Ψ::RegularizationProblem, b::AbstractVector, x̄::AbstractVector) = 
    Ψ.L⁺ * x̄ + Ψ.K0 * Ψ.T0^(-1) * Ψ.H0ᵀ * (b - Ψ.A * Ψ.L⁺ * x̄)

solve(Ψ::RegularizationProblem, b̄::AbstractVector, λ::AbstractFloat) = 
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄)

solve(Ψ::RegularizationProblem, b̄::AbstractVector, x̄₀::AbstractVector, λ::AbstractFloat) = 
    cholesky!(Hermitian(zot(Ψ.ĀĀ, λ^2.0))) \ (Ψ.Ā' * b̄ + λ^2.0 * x̄₀)

setupRegularizationProblem(A::AbstractMatrix, b::AbstractVector, order::Int) =
    setupRegularizationProblem(A, b, zeros(length(b)), order)

function setupRegularizationProblem(
    A::AbstractMatrix,
    b::AbstractVector,
    x₀::AbstractVector,
    order::Int,
)
    L = Γ(A, order)             
    n, p = size(L')
    L⁺ = (L' * L)^(-1) * L'
    K, R = qr(L')
    Kp = K[:, 1:n]
    K0 = K[:, n]
    H, T = qr(A * K0)
    H0ᵀ = (H[:, 1])'
    Hqᵀ = (H[:, 1:end])'
    T0 = T
    Ā = Hqᵀ * A * L⁺
    Iₙ = Matrix{Float64}(I, n, n) # Make sparse?
    Iₚ = Matrix{Float64}(I, p, p)

    RegularizationProblem(Ā, A, L, Ā'Ā, Ā', svd(Ā), Iₙ, Iₚ, L⁺, Hqᵀ, H0ᵀ, T0, K0, order)
end
