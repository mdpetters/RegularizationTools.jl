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

r = mdopen("baart", 100, false)
A, x = r.A, r.x

y = A  * x
b = y + 0.1y .* randn(100)

xλ1 = @> setupRegularizationProblem(A,0) solve(b, alg=:L_curve, λ₁=10.0, λ₂=1000) getfield(:x)
xλ2 = @> setupRegularizationProblem(A,0) solve(b, alg=:gcv_svd) getfield(:x)
@test round(sum(xλ1), digits = 0) == round(sum(xλ2), digits = 0)

xλ1 = @> setupRegularizationProblem(A,1) solve(b, x₀, alg=:L_curve, λ₁=100.0, λ₂=1e6) getfield(:x)
xλ2 = @> setupRegularizationProblem(A,1) solve(b, x₀, alg=:gcv_svd) getfield(:x)
@test round(sum(xλ1), digits = 0) == round(sum(xλ2), digits = 0)

xλ1 = @> setupRegularizationProblem(A,2) solve(b, alg=:L_curve, λ₁=100.0, λ₂=1e6) getfield(:x)
xλ2 = @> setupRegularizationProblem(A,2) solve(b, alg=:gcv_svd) getfield(:x)
@test round(sum(xλ1), digits = 0) == round(sum(xλ2), digits = 0)

