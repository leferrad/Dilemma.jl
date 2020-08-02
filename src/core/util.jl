using Distributions
using LinearAlgebra
using Random

export
    ones_in_zeros,
    mvrandnorm,
    sherman_morrison,
    draw_index_from_probs


"""
    ones_in_zeros(n::Integer, idx::Union{Integer, Vector}) -> Vector{Int}

Get a vector of zeros and ones.

# Arguments
- `n::Integer`: number of elements in resulting vector
- `idx::Union{Integer, Vector}`: positions for ones

# Returns
- `Vector{Int}`: vector of `n` elements, having all zeros 
    except by ones in `idx` positions.
"""
function ones_in_zeros(n::Integer, idx::Union{Integer, Vector})
    x = zeros(n, 1)

    if idx isa Integer
        x[idx] = 1
    else
        for i in idx
            x[i] = 1
        end
    end
    return x
end


"""
    sample_mvnorm(args) -> Matrix{Float64}

Produces one or more samples from the specified multivariate normal distribution.

Used by LinTSPolicy for arm selection.

# Arguments
- `n::Integer`: the number of samples required.
- `μ::Vector{Float64}: having the means of the variables
- `σ::Matrix{Float64}: a positive-definite symmetric matrix specifying the covariance matrix of the variables
- `rng=MersenneTwister(123): random number generator

# Returns
- `Matrix{Float64}`: `n` x `μ` matrix with one sample in each row.
"""
function sample_mvnorm(
    n::Integer,
    μ::Vector{Float64},
    σ::Matrix{Float64},
    rng=MersenneTwister(123)
)
    n_cols = size(σ, 2)
    μs = repeat(μ, inner=n)'  # transposed result
    A = rand(rng, Normal(), n, n_cols)
    b = cholesky(Hermitian(σ)).U
    x = A * b
    return μs + x
end


"""
    sherman_morrison_inv(A_inv, X) -> Matrix{Float64}

Sherman-Morrisson inverse.

Used by LinTSPolicy for learning phase.     

# Arguments
- `A_inv::Array{Float64}`: typically, a matrix of policy parameters
- `X::Vector{Float64}`: typically, a context vector

# Returns
- `Matrix{Float64}`: inverted matrix
"""
function sherman_morrison_inv(A_inv, X)
    # TODO: outer product using BLAS (more efficient)
    num = A_inv * (X * X') * A_inv
    den = 1.0 + X' * A_inv * X  # TODO: X' * A_inv as crossproduct??
    return A_inv - num / den
end


"""
    draw_index_from_probs(probs::Vector{Float64}, rng::AbstractRNG) -> Int

Util function to draw an arm or index from a Vector of probability values,
    using a RNG to achieve reproducible results. 

Used by policies like SoftmaxPolicy and Exp3Policy 
    to draw arms based on their parameters.        

# Arguments
- `probs::Vector{Float64}`: vector of probability values
- `rng::AbstractRNG`: random numbers generator for random operations

# Returns
- `Int`: drawn arm
"""
function draw_index_from_probs(
    probs::Vector{Float64},
    rng::AbstractRNG=MersenneTwister(123)
)
    if ~isapprox(sum(probs), 1.0, atol=3)
        throw(ArgumentError("Argument 'probs' must sum up to 1"))
    end
    
    n = length(probs)
    cum_probability = 0.0
    z = rand(rng, Uniform(0, 1))
    for i in 1:n
      cum_probability += probs[i]
      if cum_probability > z  return i end
    end

    return n
end
