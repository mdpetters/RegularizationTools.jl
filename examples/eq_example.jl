# Example 3 from manual to build the design matrix 
using RegularizationTools
using MatrixDepot
using Underscores
using Memoize
import Lazy.@>, Lazy.@>>

function get_domainfunction(K)   
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        Δt = (maximum(s) - minimum(s))./length(x)
        y = @_ mapfoldl(K(q, (_-0.5)*Δt) * x[_] * Δt, +, 1:length(x)) 
    end
end

f = get_domainfunction((x,y) -> exp(x*cos(y)))

s = range(0, stop = π, length = 12) 
q = range(0, stop = π/2, length = 12) 
A1 = designmatrix(s, q, f) 
x = sin.(s)
A1*x .- 2.0.*sinh.(q)./q
b1 = A1*x
b2 = forwardmodel(s, x, q, f)
b1 ≈ b2

a, b = 0.0, π
n, m = 12,12
c, d = 0.0, π/2
A = zeros(n,m)
w = (b-a)/n
q = range(c, stop = d, length = n)
s = [(j-0.5)*(b-a)/n for j = 1:m]
baart(x,y) = exp(x*cos(y))

for i = 1:n
    for j = 1:m
        A[i,j] = w*baart(q[i], (j-0.5)*w)
    end
end
A ≈ A1
