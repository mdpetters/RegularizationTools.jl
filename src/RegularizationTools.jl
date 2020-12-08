module RegularizationTools

using MLStyle
using Memoize
using Underscores
using LinearAlgebra
using Calculus
using Optim
using LeastSquaresOptim
import Lazy.@>, Lazy.@>>, Lazy.@as

export setupRegularizationProblem,
    RegularizationProblem,
    RegularizedSolution,
    Domain,
    solve,
    to_standard_form,
    to_general_form,
    gcv_tr,
    gcv_svd,
    Lcurve_functions,
    Γ,
    forwardmodel,
    designmatrix,
    invert,
    Lₖ,
    Lₖx₀,
    LₖB, 
    Lₖx₀B, 
    LₖDₓ,
    Lₖx₀Dₓ, 
    LₖDₓB, 
    Lₖx₀DₓB

@doc raw"""
    RegularizationProblem

This data type contains the cached matrices used in the inversion. The problem is 
initialized using the constructor [setupRegularizationProblem](@ref) with the design matrix 
A and the the Tikhonv matrix L as inputs. The hat quantities, e.g. Ā, is the calculated
design matrix in standard form. ĀĀ, Āᵀ, F̄ are precomputed to speed up repeating inversions
with different data. L⁺ₐ is cached to speed up the repeated conversion of 
data [to\_standard\_form](@ref) and [to\_general\_form](@ref)

    Ā::Matrix{Float64}     # Standard form of design matrix
    A::Matrix{Float64}     # General form of the design matrix (n×p)
    L::Matrix{Float64}     # Smoothing matrix (n×p)
    ĀĀ::Matrix{Float64}    # Cached value of Ā'Ā for performance
    Āᵀ::Matrix{Float64}    # Cached value of Ā' for performance
    F̄::SVD                 # Cached SVD decomposition of Ā 
    Iₙ::Matrix{Float64}    # Cached identity matrix n×n
    Iₚ::Matrix{Float64}    # Cached identity matrix p×p
    L⁺ₐ::Matrix{Float64}   # Cached A-weighted generalized inverse of L(standard-form conversion)    
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
    L⁺ₐ::Matrix{Float64}   # Cached A-weighted generalized inverse of L
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
    solution::Any
end

@doc raw"""
    Domain{T1<:Any,T2<:Number,T3<:Any}

Functor to map from a domain characterized by a list of setpoints [s], each 
associated with a list of numerical values [x] to a query value q. 

    s::AbstractVector{T1}
    x::AbstractVector{T2}
    q::T3
"""
struct Domain{T1<:Any,T2<:Number,T3<:Any}
    s::AbstractVector{T1}
    x::AbstractVector{T2}
    q::T3
end

@data InverseMethod begin 
    Lₖ(Int)                            # Pure Tikhonov
    Lₖx₀(Int,Vector)                   # with initial guess
    LₖB(Int,Vector,Vector)             # with bounds
    Lₖx₀B(Int,Vector,Vector,Vector)    # with initial guess + bounds
    LₖDₓ(Int,Float64)                  # with filter 
    Lₖx₀Dₓ(Int,Vector,Float64)         # with initial guess + filter 
    LₖDₓB(Int,Float64,Vector,Vector)   # with filter + bound
    Lₖx₀DₓB(Int,Vector,Float64,Vector,Vector) # with initial guess + filter + bound
end

BLAS.set_num_threads(1)
include("solvers.jl")
include("validators.jl")
include("genericfunctions.jl")
include("invert.jl")

end
