using Dilemma
using Distributions
using Random

export
    StochasticBandit,
    observe,
    pull


@doc raw"""
    StochasticBandit(D::Vector{T}, seed::Integer=123) where {T<:Distribution}

`Bandit` that handles stationary reward functions as **Univariate** Distributions to obtain
    random values. You can use it for any Distribution that return numerical values
    after a rand() call. For more details about supported Distributions, 
    see [the docs](https://juliastats.org/Distributions.jl/latest/univariate/).

This is a synthetic bandit, intended for quick tests. 

Base struct for implementations like `BernoulliBandit`,
    `GaussianBandit`, `UniformBandit`, etc.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance
```
"""

mutable struct StochasticBandit{T<:UnivariateDistribution} <: Bandit
    k::Int
    d::Nothing
    D::Vector{T}
    offset::Float64
    arms::Tuple{Vararg{Symbol}}
    rng::MersenneTwister

    function StochasticBandit(
        D::Vector{T};
        offset::Real=0.0,
        seed::Integer=123
    ) where {T<:UnivariateDistribution}
        arms = Tuple(Symbol("arm_$(i)_$(typeof(d).name)") for (i,d) in enumerate(D))
        new{T}(length(D),
               nothing,
               D,
               float(offset),
               arms,
               MersenneTwister(seed))
    end
end

# To have a shorter display name
Base.show(io::IO, b::StochasticBandit) = print(io,
    "$(string(typeof(b))) with $(length(b.D)) arms: "*
    "$([string(typeof(d)) for d in b.D])");

"""
    observe(args...) -> Context

Get a context vector (if available) from a `Bandit` for a given time step `t`.

# Arguments
- `bandit::BernoulliBandit`: bandit to get context from
- `t::Integer`: time step in simulation

# Returns
- `Context`: context only having number of arms `k` defined.
"""
function observe(bandit::StochasticBandit, t::Integer)
    """No context here"""
    return Context()
end


"""
    pull(args...) -> Reward

Get reward information about selecting the arm from a chosen `Action`,
    applied to a given `Bandit` for a given time step `t`.

# Arguments
- `bandit::BernoulliBandit`: bandit to get reward from
- `t::Integer`: time step in simulation
- `action::Action`: with arm selected by some `Policy`

# Returns
- `Reward`: reward for the arm selected, and containing
    optimal arm and the reward associated.
"""
function pull(
    bandit::StochasticBandit,
    t::Integer,
    action::Action
)
    if action.choice > bandit.k
        throw(ArgumentError("Argument 'action' must be in [1,$(bandit.k)]. Got $(action.choice)"))
    end

    # TODO: check distribution returning numerical value?
    rewards = [float(rand(bandit.rng, D)) + bandit.offset
               for D in bandit.D]
    optimal_arm = argmax(rewards)
    arm = action.choice

    reward = Reward(rewards[arm],
                    optimal_arm,
                    rewards[optimal_arm])

    return reward
end
