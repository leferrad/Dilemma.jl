using Dilemma
using LinearAlgebra
using Random

export
    ContextualEpsilonGreedyTheta,
    ContextualEpsilonGreedyPolicy,
    choose,
    initialize!,
    learn!

@doc raw"""
    ContextualEpsilonGreedyTheta <: Theta

Parameters "θ" for a `ContextualEpsilonGreedyPolicy`.
The policy assigns a θ for each of the available arms.

Parameters are used to compute the expected reward
through a linear operation over a `Context` matrix ``X ∈ \real^{d}``:

``r = X  \inv{A} b``

!!! note
    This type should be only used by an `ContextualEpsilonGreedyPolicy` instance.

# Fields
- `A::Array{T} where {T <: Real}`: Matrix of d x d dimensions
- `b::Array{T} where {T <: Real}`: Vector of d x 1 dimensions
"""
mutable struct ContextualEpsilonGreedyTheta <: Theta
    A::Array{T} where {T <: Real}
    b::Array{T} where {T <: Real}
end # mutable struct


@doc raw"""
    ContextualEpsilonGreedyPolicy(ϵ::Float64, seed::Integer=123)

Policy for Contextual Bandits, following ϵ-greedy exploration - exploitation,
    with unique linear models.

Parameters given by `ContextualEpsilonGreedyTheta` are used to compute the expected reward
    through a linear operation over a `Context` vector ``x_a ∈ \real^{d}``
    for a selected arm *a*:

``f(X,θ_{A,b},a) = (A^{-1} b) X_a``

A ϵ-greedy strategy will be used to select an arm as:

- **Explore** with probability ``p = \epsilon →`` choose an action at random
- **Exploit** with probability ``p = 1 - \epsilon →`` get action with max reward, as

    ``a = \arg\max_a f(X,θ,a)``

Using a `seed` for reproducibility, normally handled by a `Simulator` instance

!!! note
    This method should be only used for Contextual Bandits (e.g. `ContextualLogitBandit`).

# Examples
```julia
using Dilemma

# Create Policy with ϵ=0.1 for a
# Contextual Bandit with k=10 arms and d=5 dimensions
policy = ContextualEpsilonGreedyPolicy(0.1)
bandit = ContextualLogitBandit(10, 5)
agent = Agent(policy, bandit, "ctx ϵ-greedy")
```
"""
mutable struct ContextualEpsilonGreedyPolicy <: Policy
    ϵ::Float64
    θ::Union{Vector{ContextualEpsilonGreedyTheta}, Nothing}
    is_oracle::Bool
    rng::AbstractRNG

    function ContextualEpsilonGreedyPolicy(ϵ::Float64, seed::Integer=123)
        rng = MersenneTwister(seed)
        new(ϵ, nothing, false, rng)
    end
end


@doc raw"""
    choose(args...) -> Action

Use a `ContextualEpsilonGreedyPolicy` instance to choose an arm,
based on the current values of parameters and a given `Bandit` instance.

# Arguments
- `policy::ContextualEpsilonGreedyPolicy`: used for arm selection
- `t::Integer`: time step in simulation
- `bandit::Bandit`: used to get number of arms `k` and observe a `Context`

# Returns
- `Action`: containing selected arm

# Throws
- `DimensionMismatch`: When the parameters of the policy don't match with the given `Bandit`

# Examples
```julia
using Dilemma
using Random

# TODO: modify this example!!

# Create Policy
policy = ContextualEpsilonGreedyPolicy(0.1)
# Set a dummy Context randomly
d = 5  # d dimensions for x
Random.seed!(123)
x = rand(1, d)
context = Context(x)
# Get action for some t and the given context
t = 1
get_action(policy, t, context)

# output
Action(1)
```
"""
function choose(policy::ContextualEpsilonGreedyPolicy, t::Integer, bandit::Bandit)
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
        expected_rew = zeros(Float64, bandit.k, 1)
        # Observe context information
        context = observe(bandit, t)
        # get expected reward in all arms
        for arm in 1:bandit.k
            if size(policy.θ[arm].A) != (context.d, context.d)
                throw(DimensionMismatch(
                    "Policy parameters θ for arm $arm with dimension $(size(policy.θ[arm].A)) "*
                    "does not match context dimension $((context.d, context.d))"
                ))
            end
            X = collect(context)
            A = policy.θ[arm].A
            b = policy.θ[arm].b
            A_inv = inv(A)
            θ̂  = A_inv * b
            expected_rew[arm] = (X' * θ̂ )[1]
        end
        # get arm with max reward
        choice = argmax(expected_rew)[1]
    end

    return Action(choice, k=bandit.k)
end

"""
    initialize!(policy::ContextualEpsilonGreedyPolicy, bandit::Bandit)

Set initial parameters of `ContextualEpsilonGreedyPolicy` based on the attributes
    of a given `Bandit`.

!!! note
    This method should be only called by an `Agent` instance.

# Arguments
- `policy::ContextualEpsilonGreedyPolicy`: policy to be modified
- `bandit::Bandit`: used to get number of arms `k`

# Examples
```julia
using Dilemma
using Distributions
using Random

# TODO: update example!

# Create Policy
policy = ContextualEpsilonGreedyPolicy(0.1)
# Set a dummy Context randomly
d = 5  # d dimensions for x
Random.seed!(123)
x = rand(1, d)
context = Context(x)
# Set initial parameters of Policy
initialize!(policy, context)

policy.θ[1].A
# output
5×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0
```
"""
function initialize!(policy::ContextualEpsilonGreedyPolicy, bandit::Bandit)
    if ! hasproperty(bandit, :d)
        context = observe(bandit, 1)
        d = context.d
    else
        d = bandit.d
    end

    if d === nothing
        throw(ErrorException("Bandit has no value defined for dimension parameter 'd'"))
    end

    A = convert(Matrix, Diagonal(ones(d, d)))  # diagonal matrix of d x d
    b = zeros(d, 1)  # zeros vector of d x 1
    # parameters are repeated for each of the k arms
    policy.θ = [ContextualEpsilonGreedyTheta(A, b) for i in 1:bandit.k];
end

"""
    learn!(args...) -> Vector{ContextualEpsilonGreedyTheta}

Update parameters of `ContextualEpsilonGreedyPolicy` based on some action taken,
with a corresponding reward and a given context.

!!! note
    This method should be only called by an `Agent` instance.

# Arguments
- `policy::ContextualEpsilonGreedyPolicy`: policy to be updated
- `t::Integer`: time step in simulation
- `context::Context`: used to get context matrix `X`
- `action::Action`: used to get arm selected
- `reward::Reward`: to get numerical reward obtained with action taken

# Returns
- `Vector{ContextualEpsilonGreedyTheta}`: vector of `k` parameters θ updated
"""
function learn!(
    policy::ContextualEpsilonGreedyPolicy,
    t::Integer,
    context::Context,
    action::Action,
    reward::Reward
)

    if policy.θ === nothing
        throw(ErrorException("Policy parameters must be initialized with a Bandit before calling learn!()"))
    end

    arm = action.choice
    reward = reward.value
    Xa = collect(context)

    # Update linear models
    Xao = Xa .* Xa'  # outer product
    policy.θ[arm].A .+= Xao
    policy.θ[arm].b .+= Xa * reward
end
