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

function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector;
    alg = :gcv_tr,
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
     
    return RegularizationSolution(x, λ, solution)
end

function solve(
    Ψ::RegularizationProblem,
    b::AbstractVector,
    x₀::AbstractVector;
    alg = :L_curve,
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
    x̄ = solve(Ψ, b̄, λ)
    x = @>> x̄ to_general_form(Ψ, b)   

    return RegularizationSolution(x, λ, solution)
end

setupRegularizationProblem(A::AbstractMatrix, order::Int) = 
    setupRegularizationProblem(A, Γ(A, order))
#LinearAlgebra.QRCompactWYQ{Float64,Array{Float64,2}}:
function setupRegularizationProblem(A::AbstractMatrix, L::AbstractMatrix)
    n, p = size(L')
    Iₙ = Matrix{Float64}(I, n, n) 
    Iₚ = Matrix{Float64}(I, p, p)
    L⁺ = convert(Matrix, (L' * L)^(-1) * L')
    K, R = qr(L')
    K = convert(Matrix,K)
    Kp = convert(Matrix,(view(K,:,1:n)))
    K0 = convert(Vector, (view(K,:,n)))
    tmp=fill(0.0,n);
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
