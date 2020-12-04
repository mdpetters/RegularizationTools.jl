r = mdopen("shaw", 100, false)
A, x = r.A, r.x
Random.seed!(100)
y = A  * x
b = y + 0.1y .* randn(100)
x₀ = 0.5x

Ψ = setupRegularizationProblem(A,1)
λopt = @> solve(Ψ, b, alg=:gcv_svd) getfield(:λ)
@test round(λopt, digits = 3) ≈ 1.165
λopt = @> solve(Ψ, b, alg=:gcv_tr) getfield(:λ)
@test round(λopt, digits = 3) ≈ 1.165
λopt = @> solve(Ψ, b, alg =:L_curve) getfield(:λ)
@test round(λopt, digits = 3) ≈ 1.194
λopt = @> solve(Ψ, b, x₀, alg =:L_curve) getfield(:λ)
@test round(λopt, digits = 3) ≈ 1.176

