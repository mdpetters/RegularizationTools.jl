r = mdopen("shaw", 100, false)
A, x = r.A, r.x
Random.seed!(100)
y = A  * x
b = y + 0.1y .* randn(100)
x₀ = 0.5x

Ψ = setupRegularizationProblem(A, 1)               
b̄, x̄₀ = to_standard_form(Ψ, b, x₀)                
Vλ1 = gcv_svd(Ψ, b̄, x̄₀, 11.1)                     
Vλ2 = gcv_tr(Ψ, b̄, x̄₀, 11.1)                     
@test round(Vλ1, digits = 1) == round(Vλ2, digits = 1)

Vλ1 = gcv_svd(Ψ, b̄, 2.1)                     
Vλ2 = gcv_tr(Ψ, b̄, 2.1)                     
@test round(Vλ1, digits = 1) == round(Vλ2, digits = 1)

b̄ = to_standard_form(Ψ, b)                
Vλ = @_ map(gcv_svd(Ψ, b̄, _), [0.001, 1.0, 1000.0])
@test round.(Vλ, digits = 2) == [0.06, 0.05, 5.42]

L1norm, L2norm, κ = Lcurve_functions(Ψ, b̄)
L1 = L1norm.([0.1, 1.0, 10.0])
L2 = L2norm.([0.1, 1.0, 10.0])
cv = κ.([0.1, 1.0, 10.0])

@test round.(L1, digits = 1) == [2.2, 2.2, 3.1]
@test round.(L2, digits = 1) == [0.8, 0.5, 0.3]
@test round.(cv, digits = 1) == [1.0, 16.3, -18.3]