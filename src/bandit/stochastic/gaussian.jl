using Dilemma
using Distributions
using Random

export
    GaussianBandit

"""
    GaussianBandit(k::Int, p::Float64=0.5; seed=123) -> StochasticBandit

`StochasticBandit` that has `k` arms following a *Gaussian* or *Normal* reward distribution
    with provided mean `μ` and standard deviation `σ`.

    GaussianBandit(D::Vector{Normal}; seed=123)

In case the user provides de vector of `Distribution.Normal` instances
    for the bandit's arms.

    GaussianBandit(μ::Vector{Float64}, σ::Vector{Float64}; seed=123)

The vectors `μ` and σ have the mean and standard deviation values respectively
    for the distribution of each of the bandit's arms.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance.

# Examples
```julia
using Dilemma

# Create Policy for a
# Bandit with k=3 arms
# Gaussians with equal mean and stdev
policy = EpsilonGreedyPolicy(0.1)
k, μ, σ = 3, 0.1, 0.3
bandit = GaussianBandit(k, μ, σ)
agent = Agent(policy, bandit, "ϵ-greedy Gaussian")
```
"""
GaussianBandit(k, μ=0, σ=1; seed=123) = StochasticBandit([Normal(μ, σ) for _ in 1:k], seed=seed)
GaussianBandit(D::Vector{T}; seed=123) where {T<:Normal} = StochasticBandit(D, seed=seed)
GaussianBandit(μ::Vector{T}, σ::Vector{T}; seed=123) where {T<:Real} = (
    length(μ) != length(σ) ?
        throw(DimensionMismatch("Arguments 'μ' and 'σ' must have same length")) :
        StochasticBandit([Normal(μ_, σ_) for (μ_, σ_) in zip(μ, σ)], seed=seed)
)
