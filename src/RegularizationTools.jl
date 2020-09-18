module RegularizationTools

using LinearAlgebra
using MLStyle
using Pipe
using Optim
using Calculus

export setupRegularizationProblem,
    solve,
    to_standard_form,
    to_general_form,
    gcv_tr,
    gcv_svd


# b̄::Vector{Float64}     # Standard form of measured quantities
# x̄₀::Vector{Float64}    # Standard form a-priori estimate of x (p)
# b::Vector{Float64}     # General form of measured quantitites (p)
# x₀::Vector{Float64}    # General form of a-priori estimate of x (p)
# intital_guess::Bool    # Initial guess provides (true/false)

struct RegularizationProblem
    Ā::Matrix{Float64}     # Standard form of design matrix
    A::Matrix{Float64}     # General form of the design matrix (n×p)
    L::Matrix{Float64}     # Smoothing matrix (n×p)
    ĀĀ::Matrix{Float64}    # Cached value of Ā'Ā for performance
    Āᵀ::Matrix{Float64}    # Cached value of Ā' for performance
    F̄::SVD                 # Cached SVD decomposition of Ā 
    Iₙ::Matrix{Float64}    # Cached identity matrix n×n
    Iₚ::Matrix{Float64}    # Cached identity matrix p×p
    L⁺::Matrix{Float64}    # Cached left Moore–Penrose inverse (standard-form conversion)    
    Hqᵀ::Matrix{Float64}   # Cached Hq' (standard-form conversion)
    H0ᵀ::Matrix{Float64}   # Cached H0' (standard-form conversion)
    T0::Matrix{Float64}    # Cached T0  (standard-form conversion)
    K0::Vector{Float64}    # Cached K0  (standard-form conversion)
    order::Int             # Order of Tikhonov smoothing matrix (0, 1, or 2)
end

clean(x) = map(x -> x < 0.0 ? 0.0 : x, x)

include("tikhonov.jl")
include("optimization.jl")

end
