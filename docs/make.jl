push!(LOAD_PATH, "../src/")
using Documenter, LinearAlgebra

makedocs(
  sitename = "RegularizationTools.jl",
  authors = "Markus Petters",
  pages = Any[
    "Home" => "index.md",
  ]
)

