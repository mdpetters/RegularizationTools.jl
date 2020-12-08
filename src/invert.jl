@doc raw"""
    function invert(A::Matrix, b::Vector, method::InverseMethod; kwargs...)

High-level API function to perform Tikhonov inversion. The function used 
the algebraic data type InverseMethod

```julia
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
```

to define the problem and then dispatches to the correct method. The kwargs... are passed to the solve function
For example the standard way to perform second order regularization is 

```julia
xλ = @> setupRegularizationProblem(A, 2) solve(b) getfield(:x)
```

this can alternativel written as 

```julia
invert(A, b, Lₖ(2))
```

where ```Lₖ(2)``` denotes the second order method. The method nomenclature is
```Lₖ``` for regularization order, ```B``` for bounded search, ```x₀``` for 
initial condition, and ```Dₓ``` for the Huckle and Sedlacek (2012) two-step data based 
regularization. The method data type takes hyper parameters to initialize the search. 
Examples of method initializations are

```julia
# Hyper parameters 
k: order, lb: low bound, ub: upper bound, ε: noise level, x₀: initial guess
k, lb, ub, ε, x₀ = 2, zeros(8), zeros(8) .+ 50.0, 0.02, 0.5*N

xλ = invert(A, b, Lₖ(k); alg = :gcv_tr, λ₁ = 0.1)
xλ = invert(A, b, Lₖ(k); alg = :gcv_svd, λ₁ = 0.1)
xλ = invert(A, b, LₖB(k, lb, ub); alg = :L_curve, λ₁ = 0.1)
xλ = invert(A, b, Lₖx₀(k, x₀); alg = :L_curve, λ₁ = 0.1)
xλ = invert(A, b, Lₖx₀(k, x₀); alg = :gcv_tr)
xλ = invert(A, b, Lₖx₀(k, x₀); alg = :gcv_svd)
xλ = invert(A, b, Lₖx₀B(k, x₀, lb, ub); alg = :gcv_svd)
xλ = invert(A, b, LₖDₓ(k, ε); alg = :gcv_svd)
xλ = invert(A, b, LₖDₓB(k, ε, lb, ub); alg = :gcv_svd)
xλ = invert(A, b, Lₖx₀Dₓ(k, x₀, ε); alg = :gcv_svd)
xλ = invert(A, b, Lₖx₀DₓB(k, x₀, ε, lb, ub); alg = :gcv_svd)
```
"""    
function invert(A::Matrix, b::Vector, method::InverseMethod; kwargs...)
    n = length(b)

    res = @match method begin
        Lₖ(k) => @as p begin
            setupRegularizationProblem(A, k)
            solve(p, b; kwargs...)    
            getfield(p, :x)
        end
        Lₖx₀(k, x₀) => @as p begin
            setupRegularizationProblem(A, k)
            solve(p, b, x₀; kwargs...)      
            getfield(p, :x)
        end
        LₖB(k, lb, ub)  => @as p begin
            setupRegularizationProblem(A, k)
            solve(p, b, lb, ub; kwargs...)     
            getfield(p, :x)
        end
        Lₖx₀B(k, x₀, lb, ub) => @as p begin
            setupRegularizationProblem(A, k)
            solve(p, b, x₀, lb, ub; kwargs...) 
            getfield(p, :x)
        end
        LₖDₓ(k, ε) => begin
            x̂ = @as p begin
                setupRegularizationProblem(A, k)
                solve(p, b; kwargs...) 
                getfield(p, :x)
            end 
            x̂[x̂ .< ε] .= ε 
            ψ = @>> Γ(A, k)*Diagonal(x̂)^(-1) setupRegularizationProblem(A)
            @as p begin
                solve(ψ, b; kwargs...) 
                getfield(p, :x)
            end
        end
        LₖDₓB(k, ε, lb, ub) => begin
            x̂ = @as p begin
                setupRegularizationProblem(A, k)
                solve(p, b; kwargs...) 
                getfield(p, :x)
            end 
            x̂[x̂ .< ε] .= ε
            ψ = @>> Γ(A, k)*Diagonal(x̂)^(-1) setupRegularizationProblem(A)
            @as p begin
                solve(ψ, b, x̂, lb, ub; kwargs...) 
                getfield(p, :x)
            end 
        end
        Lₖx₀Dₓ(k, x₀, ε) => begin
            x̂ = @as p begin
                setupRegularizationProblem(A, k)
                solve(p, b, x₀; kwargs...) 
                getfield(p, :x)
            end 
            x̂[x̂ .< ε] .= ε
            ψ = @>> Γ(A, k)*Diagonal(x̂)^(-1) setupRegularizationProblem(A)
            @as p begin
                solve(ψ, b; kwargs...) 
                getfield(p, :x)
            end 
        end
        Lₖx₀DₓB(k, x₀, ε, lb, ub) => begin
            x̂ = @as p begin
                setupRegularizationProblem(A, k)
                solve(p, b, x₀; kwargs...) 
                getfield(p, :x)
            end 
            x̂[x̂ .< ε] .= ε
            ψ = @>> Γ(A, k)*Diagonal(x̂)^(-1) setupRegularizationProblem(A)
            @as p begin
                solve(ψ, b, x̂, lb, ub; kwargs...) 
                getfield(p, :x)
            end 
        end
        _ => throw("Unknown Method")
    end
end
