using RegularizationTools
using MatrixDepot
using Random
using Test
using Underscores
using LinearAlgebra
using Lazy

@testset "RegularizationTools.jl" begin
    include("standard_form.jl")  # Test standard form transformation
    include("solvers.jl")        # Test optimization
    include("validators.jl")     # Test validators
    include("domainfunctions.jl")
end
