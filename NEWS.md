# RegularizationTools.jl NEWS

#### Notes on release changes and ongoing development
---
## current master

## v0.3.1
- Add memoization
- Merge CompatHelper: add new compat entry for "Memoize" at version "0.4"
- Fix various bugs related to non-square matrices

## v0.3.0
- Add abstract interface to generate design matrix from a forward model.
- Add documentation and examples how to use the interface
- Add fallback in solve function in case cholesky factorization fails.
- Fix bug when creating L matrix for non-square design matrices.

## v0.2.0
- Add constraint minimization solver to enfore upper and lower bound for some problems.
- Add documentation.

## v0.1.1
- Fix standard form conversion error. Solution is now based on generalized SVD. 
- Touch up documentation.
- RegularizationTools is now part of the general registry

## v0.1.0
- Initial Release
