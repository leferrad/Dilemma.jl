using Dilemma
using Random: AbstractRNG, MersenneTwister

export
    RandomTheta,
    RandomPolicy,
    choose,
    initialize!,
    learn!

@doc raw"""
    RandomTheta <: Theta

Parameters "θ" for a `RandomPolicy`.
The policy assigns a θ for each of the available arms.

Here "θ"" is only used to track the number of times the arm was selected.

# Fields
- `n::Int`: amount of times the arm was chosen
"""
mutable struct RandomTheta <: Theta
    n::Int
end

@doc raw"""
    RandomPolicy(seed::Integer=123)

`Policy` that always explores, choosing arms uniformly at random.
    This is normally used as a baseline to compare policies.

Using a `seed` for reproducibility, normally handled by a `Simulator` instance

# Examples
```julia
using Dilemma

# Create Policy for a
# Bandit with k=3 arms
policy = RandomPolicy()
bandit = BernouilleBandit([0.2, 0.5, 0.1])
agent = Agent(policy, bandit, "Random")
```
"""
mutable struct RandomPolicy <: Policy
    θ::Union{Vector{RandomTheta}, Nothing}
    is_oracle::Bool
    rng::AbstractRNG

    function RandomPolicy(;seed::Integer=123)
        rng = MersenneTwister(seed)
        new(nothing, false, rng)
    end
end

@doc raw"""
    choose(args...) -> Action

Use a `RandomPolicy` instance to choose an arm,
    based on the current values of parameters.

# Arguments
- `policy::RandomPolicy`: used for arm selection
- `t::Integer`: time step in simulation
- `bandit::Bandit`: used to get number of arms `k`

# Returns
- `Action`: containing selected arm

"""
function choose(policy::RandomPolicy, t::Integer, bandit::Bandit)
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

    # Always explore
    choice = rand(policy.rng, 1:bandit.k)

    return Action(choice, k=bandit.k)
end

"""
    initialize!(policy::RandomPolicy, bandit::Bandit)

Set initial parameters of `RandomPolicy`.

# Arguments
- `policy::RandomPolicy`: policy to be modified
- `bandit::Bandit`: used to get number of arms `k`
"""
function initialize!(policy::RandomPolicy, bandit::Bandit)
    # parameters are repeated for each of the k arms
    policy.θ = [RandomTheta(0) for i in 1:bandit.k];
end

"""
    learn!(args...) -> Vector{RandomTheta}

Update parameters of `RandomPolicy` based on some action taken.

# Arguments
- `policy::RandomPolicy`: policy to be updated
- `t::Integer`: time step in simulation
- `context::Context`: not used in this implementation
- `action::Action`: used to get arm selected
- `reward::Reward`: not used in this implementation

# Returns
- `Vector{RandomTheta}`: vector of `k` parameters θ updated
"""
function learn!(
    policy::RandomPolicy,
    t::Integer,
    context::Context,
    action::Action,
    reward::Reward
)
    arm = action.choice
    reward = reward.value

    # Update parameters
    policy.θ[arm].n += 1

    return policy.θ
end
