# Example 2 from manual to build the design matrix 
using RegularizationTools
using Underscores
using MLStyle
using Memoize
using LinearAlgebra
using Gadfly

import Lazy.@>, Lazy.@>>

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
λs = [300e-9, 500e-9, 900e-9]

s = zip(["Dp$i" for i = 1:length(Dp)],Dp) |> collect
q1 = zip(["βs$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect
q2 = zip(["βa$i" for i in Int.(round.(λs*1e9, digits=0))], λs) |> collect
q = [q1; q2] 

ref = complex(1.8, 0.01)
f = get_domainfunction(ref)
A = designmatrix(s, q, f) 
b = forwardmodel(s, N, q, f) 

y = A*N 
b = A*N .+ 0.01*y .* randn(6)

xλ = @> setupRegularizationProblem(A,2) solve(b, alg=:gcv_svd) getfield(:x)
plot(x = Dp, y = xλ, Geom.line, layer(x = Dp, y = N))