push!(LOAD_PATH, "../src/")
using Documenter, LinearAlgebra, RegularizationTools

makedocs(
  sitename = "RegularizationTools.jl",
  authors = "Markus Petters",
  pages = Any[
    "Home" => "index.md",
    "Manual" => "manual.md",
    "Theory" => "theory/theory.md",
    "Library" => "library.md",
    "References" => "references.md"
  ]
)

