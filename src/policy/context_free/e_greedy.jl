using Dilemma
using Random: AbstractRNG, MersenneTwister

export
    EpsilonGreedyTheta,
    EpsilonGreedyPolicy,
    choose,
    initialize!,
    learn!

@doc raw"""
    EpsilonGreedyTheta <: Theta

Parameters "θ" for a `EpsilonGreedyPolicy`.
The policy assigns a θ for each of the available arms.

Parameters are used to track the reward information obtained for a given arm,
    and use it in exploration mode.

!!! note
    This type should be only used by an `EpsilonGreedyPolicy` instance.

# Fields
- `μ::Float64`: average regret of choosing a given arm
- `n::Int`: amount of times the arm was chosen
"""
mutable struct EpsilonGreedyTheta <: Theta
    μ::Float64
    n::Int
end

@doc raw"""
    EpsilonGreedyPolicy(ϵ::Float64, seed::Integer=123)

Policy for Context-free Bandits, following ϵ-greedy exploration - exploitation.

A ϵ-greedy strategy will be used to select an arm as:

- **Explore** with probability ``p = \epsilon →`` choose an action at random
- **Exploit** with probability ``p = 1 - \epsilon →`` get action with max reward

Using a `seed` for reproducibility, normally handled by a `Simulator` instance

!!! note
    This method should be only used for Context-free Bandits (e.g. `BernoulliBandit`).

# Examples
```julia
using Dilemma

# Create Policy with ϵ=0.1 for a
# Bandit with k=3 arms
policy = EpsilonGreedyPolicy(0.1)
bandit = BernoulliBandit([0.2, 0.5, 0.1])
agent = Agent(policy, bandit, "ϵ-greedy")
```
"""
mutable struct EpsilonGreedyPolicy <: Policy
    ϵ::Float64
    θ::Union{Vector{EpsilonGreedyTheta}, Nothing}
    is_oracle::Bool
    rng::AbstractRNG

    function EpsilonGreedyPolicy(ϵ::Float64; seed::Integer=123)
        rng = MersenneTwister(seed)
        new(ϵ, nothing, false, rng)
    end
end

@doc raw"""
    choose(args...) -> Action

Use a `EpsilonGreedyPolicy` instance to choose an arm,
    based on the current values of parameters.

# Arguments
- `policy::EpsilonGreedyPolicy`: used for arm selection
- `t::Integer`: time step in simulation
- `bandit::Bandit`: used to get number of arms `k`

# Returns
- `Action`: containing selected arm

# Throws
- `DimensionMismatch`: When the parameters of the policy don't match with the given `Bandit`
"""
function choose(policy::EpsilonGreedyPolicy, t::Integer, bandit::Bandit)
    if policy.θ === nothing
        # Set parameters of policy if they were not set yet
        initialize!(policy, bandit)
    end

    # Check compatibility between policy and bandit
    if length(policy.θ) != bandit.k
        throw(DimensionMismatch(
            "Policy parameters θ with dimension k=$(length(policy.θ)) does not match "*
             "with bandit dimension k=$(bandit.k)"))
    end

    # Get probability uniformly
    p = rand(policy.rng)

    if p <= policy.ϵ
        # Explore
        choice = rand(policy.rng, 1:bandit.k)
    else
        # Exploit
        choice = argmax([policy.θ[arm].μ for arm in 1:bandit.k])[1]
    end

    return Action(choice, k=bandit.k)
end

"""
    initialize!(policy::EpsilonGreedyPolicy, bandit::Bandit)

Set initial parameters of `EpsilonGreedyPolicy` based on the given `Bandit`.

# Arguments
- `policy::EpsilonGreedyPolicy`: policy to be modified
- `bandit::Bandit`: used to get number of arms `k`
"""
function initialize!(policy::EpsilonGreedyPolicy, bandit::Bandit)
    # parameters are repeated for each of the k arms
    policy.θ = [EpsilonGreedyTheta(0.0, 0) for i in 1:bandit.k];
end

"""
    learn!(args...) -> Vector{EpsilonGreedyTheta}

Update parameters of `EpsilonGreedyPolicy` based on some action taken,
    with a corresponding reward.

# Arguments
- `policy::EpsilonGreedyPolicy`: policy to be updated
- `t::Integer`: time step in simulation
- `context::Context`: not used in this implementation
- `action::Action`: used to get arm selected
- `reward::Reward`: to get numerical reward obtained with action taken

# Returns
- `Vector{EpsilonGreedyTheta}`: vector of `k` parameters θ updated
"""
function learn!(
    policy::EpsilonGreedyPolicy,
    t::Integer,
    context::Context,
    action::Action,
    reward::Reward
)
    arm = action.choice
    reward = reward.value

    # Update parameters
    policy.θ[arm].n += 1
    policy.θ[arm].μ += (reward - policy.θ[arm].μ) / policy.θ[arm].n

    return policy.θ
end
