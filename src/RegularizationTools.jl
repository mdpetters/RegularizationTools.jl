module RegularizationTools

using LinearAlgebra
using MLStyle
using Lazy
using Underscores
using Optim
using Calculus

export setupRegularizationProblem,
    solve,
    to_standard_form,
    to_general_form,
    gcv_tr,
    gcv_svd,
    Lcurve_functions

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
end

struct RegularizationSolution
    x::AbstractVector
    λ::AbstractFloat
    solution::Optim.UnivariateOptimizationResults
end

include("solvers.jl")
include("validators.jl")

end
