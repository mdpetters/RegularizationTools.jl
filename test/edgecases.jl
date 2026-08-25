# Edge-case and regression tests for input validation, struct field correctness,
# bounded-solve correctness, and coverage gaps.
using Random
using Optim

@testset "Γ input validation" begin
    @test_throws ArgumentError Γ(10, -1)
    @test_throws ArgumentError Γ(5, 5)
    @test Γ(8, 0) == Matrix{Float64}(I, 8, 8)
    @test size(Γ(8, 1)) == (7, 8)
    @test size(Γ(8, 2)) == (6, 8)
end

@testset "setupRegularizationProblem input validation" begin
    A = rand(10, 8)
    @test_throws DimensionMismatch setupRegularizationProblem(A, Γ(7, 2))     # wrong column count
    @test_throws DimensionMismatch setupRegularizationProblem(A, rand(9, 8))   # L has more rows than columns
end

@testset "RegularizationProblem field correctness" begin
    A = rand(12, 8)
    L = Γ(8, 2)
    Ψ = setupRegularizationProblem(A, L)
    @test Ψ.L⁺ ≈ pinv(L)
    @test Ψ.L⁺ₐ ≈ (Matrix{Float64}(I, 8, 8) - Ψ.K₀T⁻¹H₀ᵀ * A) * pinv(L)
end

@testset "bounded solve correctness" begin
    Random.seed!(42)
    A = rand(15, 6)
    xtrue = abs.(rand(6))
    b = A * xtrue
    Ψ = setupRegularizationProblem(A, 2)

    # Loose bounds: bounded solution must match the algebraic (unconstrained) solution
    sol = solve(Ψ, b; alg = :gcv_svd)
    xb = solve(Ψ, b, fill(-1e8, 6), fill(1e8, 6); alg = :gcv_svd).x
    @test xb ≈ sol.x rtol = 1e-4

    # Tight bounds are respected
    lb, ub = zeros(6), fill(0.5, 6)
    xb2 = solve(Ψ, b, lb, ub; alg = :gcv_svd).x
    @test all(lb .<= xb2 .<= ub .+ 1e-10)
end

@testset "designmatrix shortcut" begin
    s = range(0, stop = π, length = 8)
    f(node::Domain) = sum(node.x)
    @test designmatrix(s, f) == designmatrix(s, s, f)
end

@testset "λ search with non-bounds method" begin
    Random.seed!(7)
    A = rand(12, 5)
    b = A * abs.(rand(5))
    Ψ = setupRegularizationProblem(A, 1)
    sol = solve(Ψ, b; method = NelderMead())
    @test sol.λ > 0.0
    @test isfinite(sol.λ)
    @test length(sol.x) == 5
end

@testset "type stability" begin
    @test_nowarn @inferred Γ(8, 2)
    @test_nowarn @inferred setupRegularizationProblem(rand(10, 8), 2)
    s = range(0, stop = π, length = 8)
    q = range(0, stop = π / 2, length = 8)
    f(node::Domain) = sum(node.x)
    @test_nowarn @inferred designmatrix(s, q, f)
    @test_nowarn @inferred forwardmodel(s, collect(1.0:8.0), q, f)

    # solve-path primitives
    A = rand(10, 6)
    b = A * rand(6)
    Ψ = setupRegularizationProblem(A, 2)
    b̄ = to_standard_form(Ψ, b)
    x̄ = solve(Ψ, b̄, 1.0)
    @test_nowarn @inferred to_standard_form(Ψ, b)
    @test_nowarn @inferred to_general_form(Ψ, b, x̄)
    @test_nowarn @inferred solve(Ψ, b̄, 1.0)
    @test_nowarn @inferred gcv_svd(Ψ, b̄, 1.0)
    @test_nowarn @inferred gcv_tr(Ψ, b̄, 1.0)
end
