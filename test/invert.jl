include("mie_functions.jl")
include("size_functions.jl")

function get_domainfunction(ref)   
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        Dp, λ = (@_ map(_[2], s)), q[2]
       
        Q = @match q[1][1:3] begin 
            "βs" => @_ map(Qsca(π*Dp[_]*1e-9/λ, ref), 1:length(Dp))
            "βa" => @_ map(Qabs(π*Dp[_]*1e-9/λ, ref), 1:length(Dp))
            _ => trow("error")
        end
        mapreduce(+, 1:length(Dp)) do i
            π/4.0 *(x[i]*1e6)*Q[i]*(Dp[i]*1e-6)^2.0*1e6
        end
    end
end

Dp, N = lognormal([[100.0, 500.0, 1.4]], d1 = 100.0, d2 = 1000.0, bins = 8)
λs = [300e-9, 500e-9, 900e-9, 1200e-9, 1500e-9]

s = zip(["Dp$i" for i = 1:length(Dp)],Dp) |> collect
q1 = zip(["βs$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect
q2 = zip(["βa$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect
q = [q1; q2] 

ref = complex(1.8, 0.01)
f = get_domainfunction(ref)
A = designmatrix(s, q, f) 

e = [-0.4, 0.12, 1.26, -1.42, 0.3, 0.8, 0.15, 0.04, -0.68, -0.61]
b = A*N .+ 0.01*(A*N) .* e

lb, ub, ε, x₀ = zeros(8), zeros(8) .+ 50.0, 0.02, 0.5*N

xλ1 = @> setupRegularizationProblem(A,2) solve(b, alg=:gcv_svd) getfield(:x)
xλ2 = invert(A, b, Lₖ(2); alg = :gcv_svd)
@test xλ1 ≈ xλ2

xλ = invert(A, b, Lₖ(2); alg = :gcv_svd, λ₁ = 0.1)
@test round(sum(xλ), digits = 0) == 176.0
xλ = invert(A, b, LₖB(2,lb,ub); alg = :L_curve, λ₁ = 0.1)
@test round(sum(xλ), digits = 0) == 143.0
xλ = invert(A, b, Lₖx₀(2, x₀); alg = :L_curve, λ₁ = 0.1)
@test round(sum(xλ), digits = 0) == -105.0
xλ = invert(A, b, Lₖx₀(2, x₀); alg = :gcv_tr)
@test round(sum(xλ), digits = 0) == 177
xλ = invert(A, b, Lₖx₀(2, x₀); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 177
xλ = invert(A, b, Lₖx₀B(2,x₀,lb,ub); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 179
xλ = invert(A, b, LₖDₓ(2, ε); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 133.0
xλ = invert(A, b, LₖDₓB(2, ε, lb, ub); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 139.0
xλ = invert(A, b, Lₖx₀Dₓ(2,x₀, ε); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 133
xλ = invert(A, b, Lₖx₀DₓB(2, x₀, ε, lb, ub); alg = :gcv_svd)
@test round(sum(xλ), digits = 0) == 139