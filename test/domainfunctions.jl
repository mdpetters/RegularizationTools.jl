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
@test b1 ≈ b2

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
@test A ≈ A1


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

@test A ≈ B
forwardmodel(s,img,q,f) ≈ A*img
forwardmodel(s,img,q,f)
img = @> rand(11,21) reshape(:)
ii = randperm(11*21)
s = @> [(i,j) for i = 0:10, j = 0:20] reshape(:)
q = s
A = designmatrix(s, q, f) 
B = designmatrix(s[ii], q[ii], f) 
@test (A*img)[ii] ≈ B*img[ii]