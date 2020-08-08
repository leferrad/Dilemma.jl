using Dilemma
using Distributions
using Random

export
    UniformBandit


"""
    UniformBandit(k::Int, a=1, b=1; seed=123) -> StochasticBandit

`StochasticBandit` that has `k` arms following a *Uniform* reward distribution
    with interval [`a`, `b`]. This could be used to model random
    reward values to test policies.

    UniformBandit(D::Vector{T}; seed=123) where {T<:Uniform}

In case the user provides the vector of `Distribution.Uniform` instances
    for the bandit's arms.

    UniformBandit(a::Vector{T}, b::Vector{T}; seed=123) where {T<:Real}

The vectors `a` and `b` have the parameter values
    for each of the bandit's arms that follow a `Distribution.Uniform` function.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance.

# Examples
```julia
using Dilemma

# Create Policy for a
# Bandit with k=3 arms
policy = EpsilonGreedyPolicy(0.1)
k, a, b = 3, -3, 3
bandit = UniformBandit(k, a, b)
agent = Agent(policy, bandit, "ϵ-greedy Uniform")
```
"""
UniformBandit(k::Int, a=-1, b=1; seed=123) = StochasticBandit([Uniform(a, b) for _ in 1:k], seed=seed)
UniformBandit(D::Vector{T}; seed=123) where {T<:Uniform} = StochasticBandit(D, seed=seed)
UniformBandit(a::Vector{T}, b::Vector{T}; seed=123) where {T<:Real} = (
    length(a) != length(b) ?
        throw(DimensionMismatch("Arguments 'a' and 'b' must have same length")) :
        StochasticBandit([Uniform(a_, b_) for (a_, b_) in zip(a, b)], seed=seed)
)
