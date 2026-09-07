# Example from Jonathan Stickel, with code posted in issue #7
# The raised issue has been addressed in version 0.5.0 
import RegularizationTools
const RT = RegularizationTools
import LinearAlgebra
const LA = LinearAlgebra
using CairoMakie

# create simulated data
npts = 100
xmin = 0.0
xspan = 3/2*π
x = range(xmin, xmin+xspan, length=npts)
yt = sin.(x)
stdev = 1e-1*maximum(yt)
y = yt + stdev*randn(npts)

# RegularizationTools
I = Array{Float64}(LA.I, (npts, npts)) # `A` matrix
Ψ = RT.setupRegularizationProblem(I, 2) 
lmbd = 10.0
ybar = RT.solve(Ψ, y, lmbd) # direct solve in so-called "standard" form
yhat = RT.to_general_form(Ψ, y, ybar)

# direct smoothing by regularization -- given λ and equally spaced x 
function deriv_matrix(d::Int64, N::Int64)
    if d == 0
        return Array{Int64}(LA.I, (N, N))
    end
    return diff(deriv_matrix(d-1, N), dims=1)
end
d = 2
λ = 1e-3 # "scaled" lambda
D = deriv_matrix(d, npts)
δ = LA.tr(D'D)/npts^(d+2)
yhatd = (LA.I + λ/δ*D'D)\y


fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
scatter!(ax, x, y, label="data", color=:white, strokecolor=:black, strokewidth=1)
lines!(ax, x, yhat, label="Regularization tools")
lines!(ax, x, yhatd, label="Direct method")
axislegend(ax)
fig
