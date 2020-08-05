using Random

export
    Reward

"""
    Reward(value, optimal_arm, optimal_value, regret)

Represents the reward obtained from pulling a `Bandit`'s arm.

This is handled directly by a `Policy`, so it should be transparent for the user in the program.

# Fields
- `value::T where {T <: Real}`: reward value obtained 
- `optimal_arm::Union{Integer, Nothing}`: index of arm with optimal reward for that moment
- `optimal_value::Union{T, Nothing} where {T <: Real}`: value of optimal reward for that moment
"""
mutable struct Reward
    value::Float64
    optimal_arm::Union{Int, Nothing}
    optimal_value::Union{Float64, Nothing}

    function Reward(
        value::T,
        optimal_arm::Union{Integer, Nothing}=nothing,
        optimal_value::Union{T, Nothing}=nothing
    ) where {T<:Real}
        new(float(value), optimal_arm, optimal_value)
    end
end


abstract type RewardFunction <: Function end

# (r::RewardFunction)(t::Integer, context::Context, arm::Integer; kwargs...) = throw(ErrorException("Not implemented!"))

# NOTE: RandomRewardFunction not needed, see StochasticBandit
