module RegularizationTools

using LinearAlgebra
using MLStyle
using Lazy
using Underscores
using Optim
using Calculus

export setupRegularizationProblem,
    RegularizationProblem,
    RegularizedSolution,
    solve,
    to_standard_form,
    to_general_form,
    gcv_tr,
    gcv_svd,
    Lcurve_functions,
    Γ

@doc raw"""
    RegularizationProblem

This data type contains the cached matrices used in the inversion. The problem is 
initialized using the constructor [setupRegularizationProblem](@ref) with the design matrix 
A and the the Tikhonv matrix L as inputs. The hat quantities, e.g. Ā, is the calculated
design matrix in standard form. ĀĀ, Āᵀ, F̄ are precomputed to speed up repeating inversions
with different data. Hqᵀ, Hqᵀ, T0, K0 are cached to speed up the repeated conversion of 
data [to\_standard\_form](@ref) and [to\_general\_form](@ref)

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
    Hqᵀ::Matrix{Float64}   # Cached H0' (standard-form conversion)
    T0::Matrix{Float64}    # Cached T0  (standard-form conversion)
    K0::Vector{Float64}    # Cached K0  (standard-form conversion)

"""
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

@doc raw"""
    RegularizatedSolution

Data tpye to store the optimal solution x of the inversion. λ is the optimal λ used 
solution is the raw output from the Optim search.

    x::AbstractVector
    λ::AbstractFloat
    solution::Optim.UnivariateOptimizationResults
"""
struct RegularizedSolution
    x::AbstractVector
    λ::AbstractFloat
    solution::Optim.UnivariateOptimizationResults
end

include("solvers.jl")
include("validators.jl")

end
