# Example 1 from manual to build the design matrix 

using RegularizationTools
using MatrixDepot
using Random
import Lazy.@>, Lazy.@>>

function get_domainfunction(σ, c)
    blur(x,y) = 1.0/(2.0*π*σ^2.0)*exp(-(x^2.0 + y^2.0)/(2.0*σ^2.0))
    
    function f(node::Domain)
        s, x, q = node.s, node.x, node.q
        x₁, y₁ = q[1], q[2]

        y = mapfoldl(+, 1:length(x)) do i
            x₀, y₀ = s[i][1], s[i][2]   
            Δx, Δy = abs(x₁-x₀), abs(y₁-y₀)
            tmp = (Δx < c) && (Δy < c) ? blur(Δx,Δy) * x[i] : 0.0
        end
    end
end

img = @> rand(21,21) reshape(:)
f = get_domainfunction(2.0, 4)
s = @> [(i,j) for i = 0:20, j = 0:20] reshape(:)
q = s
A = designmatrix(s, q, f) 
B = MatrixDepot.blur(Float64, 21, 4, 2.0, true) |> Matrix

A ≈ B
forwardmodel(s,img,q,f) ≈ A*img
forwardmodel(s,img,q,f)
img = @> rand(11,21) reshape(:)
ii = randperm(11*21)
s = @> [(i,j) for i = 0:10, j = 0:20] reshape(:)
q = s
A = designmatrix(s, q, f) 
B = designmatrix(s[ii], q[ii], f) 
(A*img)[ii] ≈ B*img[ii]
