md(A, x) = @. A[1] / (√(2π) * log(A[3])) * exp(-(log(x / A[2]))^2 / (2log(A[3])^2))
logn(A, x) = mapreduce((A) -> md(A, x), +, A)

function lognormal(A; d1 = 8.0, d2 = 2000.0, bins = 256)
    De = 10.0 .^ range(log10(d1), stop = log10(d2), length = bins + 1)
    Dp = sqrt.(De[2:end] .* De[1:end-1])
    ΔlnD = log.(De[2:end] ./ De[1:end-1])
    S = logn(A, Dp)
    N = S .* ΔlnD
    return Dp, N
end
