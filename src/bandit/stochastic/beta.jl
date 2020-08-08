using Dilemma
using Distributions
using Random

export
    BetaBandit

"""
    BetaBandit(k::Int, α=1, β=1; seed=123) -> StochasticBandit

`StochasticBandit` that has `k` arms following a *Beta* reward distribution
    with `α` and `β` parameters. In a bandit scenario, this can be a suitable model
    for the random behavior of percentages and proportions.

    BetaBandit(D::Vector{T}; seed=123) where {T<:Beta}

In case the user provides the vector of `Distribution.Beta` instances
    for the bandit's arms.

    BetaBandit(α::Vector{T}, β::Vector{T}; seed=123) where {T<:Real}

The vectors `α` and `β` have the parameter values
    for each of the bandit's arms that follow a `Distribution.Beta` function.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance.

# Examples
```julia
using Dilemma

# Create Policy for a
# Bandit with k=3 arms
policy = EpsilonGreedyPolicy(0.1)
k, α, β = 3, 1, 2
bandit = BetaBandit(k, α, β)
agent = Agent(policy, bandit, "ϵ-greedy Beta")
```
"""
BetaBandit(k::Int, α=1, β=1; seed=123) = StochasticBandit([Beta(α, β) for _ in 1:k], seed=seed)
BetaBandit(D::Vector{T}; seed=123) where {T<:Beta} = StochasticBandit(D, seed=seed)
BetaBandit(α::Vector{T}, β::Vector{T}; seed=123) where {T<:Real} = (
    length(α) != length(β) ?
        throw(DimensionMismatch("Arguments 'α' and 'β' must have same length")) :
        StochasticBandit([Beta(α_, β_) for (α_, β_) in zip(α, β)], seed=seed)
)
