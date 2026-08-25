using RegularizationTools
using MatrixDepot
using Memoize
using MLStyle
using Random
using Test
using Underscores
using LinearAlgebra
using Lazy
using CSV
using DataFrames

@testset "RegularizationTools.jl" begin
    include("standard_form.jl")  # Test standard form transformation
    include("solvers.jl")        # Test optimization
    include("validators.jl")     # Test validators
    include("domainfunctions.jl") 
    include("invert.jl")
    include("edgecases.jl")      # Input validation, field correctness, edge cases
end
