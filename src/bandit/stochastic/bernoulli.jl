using Dilemma
using Distributions
using Random

export
    BernoulliBandit

"""
    BernoulliBandit(k::Int, p=0.5; seed=123)

`Bandit` that has `k` arms following a *Bernoulli* reward distribution
    with probability `p`. In a bandit scenario, this can be used to simulate
    binary events, such as user clicks or sales conversions from recommendations.

    BernoulliBandit(D::Vector{T}; seed=123) where {T<:Bernoulli}

In case the user provides the vector of `Distribution.Bernoulli` instances
    for the bandit's arms.

    BernoulliBandit(p::Vector{T}; seed=123) where {T<:Real}

The vector `p` has the weights probability of binary reward values
    for each of the bandit's arms.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance.

# Examples
```julia
using Dilemma

# Create Policy for a
# Bandit with k=3 arms
policy = EpsilonGreedyPolicy(0.1)
bandit = BernoulliBandit([0.2, 0.5, 0.1])
agent = Agent(policy, bandit, "ϵ-greedy Bernoulli")
```
"""
BernoulliBandit(k::Int, p=0.5; seed=123) = StochasticBandit([Bernoulli(p) for _ in 1:k], seed=seed)
BernoulliBandit(D::Vector{T}; seed=123) where {T<:Bernoulli} = StochasticBandit(D, seed=seed)
BernoulliBandit(p::Vector{T}; seed=123) where {T<:Real} = StochasticBandit([Bernoulli(p_) for p_ in p], seed=seed)
