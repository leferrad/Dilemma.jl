using Dates
using Dilemma
using Distributions
using Random

export
    Agent,
    Step,
    reset_agent!,
    do_step!,
    set_seed!


"""
    Step(t, context, action, reward)

Represents the result of an step taken by an `Agent` in a simulation.

# Fields
- `t::Int`: time step in the program
- `context::Union{Context, Nothing}`: context obtained from `Bandit` in the given time `t`.
- `action::Union{Action, Nothing}`: action chosen by `Policy`, applied to `Bandit`.
- `reward::Union{Reward, Nothing}`: reward obtained after applying action in `Bandit`.
```
"""
mutable struct Step
    t::Int
    context::Union{Context, Nothing}
    action::Union{Action, Nothing}
    reward::Union{Reward, Nothing}

    function Step(
        t::Integer, 
        context::Union{Context, Nothing}, 
        action::Union{Action, Nothing},
        reward::Union{Reward, Nothing}
    )
        if t < 0
            throw(ArgumentError("Argument 't' must be >= 0"))
        end    

        new(t, context, action, reward)
    end
end


"""
    Agent(policy, bandit; name, sparse)

Represents an actor in a bandits program which handles a `Policy` that interacts
    with a `Bandit` by selecting arms to pull to increase the rewards obtained 
    along the steps.

# Fields
- `policy::Policy`: used by agent to select arms to pull
- `bandit::Bandit`: containing arms to pull
- `name::Union{String, Nothing}=nothing`: if nothing, a default name is set
- `sparse::Float64=0.0`: probability for learning a given step
- `seed::Integer=123`: used for reproducibility
```
"""
mutable struct Agent
    policy::Policy 
    bandit::Bandit
    name::String
    sparse::Float64
    t::Integer
    cum_regret::Float64
    cum_reward::Float64
    seed::Int
    rng::MersenneTwister

    function Agent(
        policy::Policy, 
        bandit::Bandit
        ;
        name::Union{String, Nothing}=nothing, 
        sparse::Float64=0.0,
        seed::Integer=123,
    )

        if name === nothing
            name = "Agent: Policy=$(string(typeof(policy))), "*
                   "Bandit=$(string(typeof(bandit)))"
        end

        agent = new(policy, bandit, name, sparse, 
                    1, 0.0, 0.0, seed, MersenneTwister(seed))

        return agent
    end
end


"""
    reset_agent!(agent::Agent)

Reset meta data of agent, such as the policy and bandit,
    the time step `t`, the `rng`, and cumulated statistics.   

# Arguments
- `agent::Agent`: agent to reset
"""
function reset_agent!(agent::Agent)
    initialize!(agent.policy, agent.bandit);
    
    agent.cum_reward = 0.0;
    agent.cum_regret = 0.0;
    agent.t = 1;
    agent.rng = MersenneTwister(agent.seed);
end


"""
    do_step!(agent::Agent)

Use an `Agent` to do a step in a bandits program:
- Choose an arm to pull based on the agent's `Policy`
- Pull the arm selected from the agent's `Bandit`
- Compute results obtained (context, reward) to let the `Policy` learn from that 
- Return and `Step` with the results

# Arguments
- `agent::Agent`: agent to use
"""
function do_step!(agent::Agent)
    # Pull an arm based on some selected action
    action = choose(agent.policy, agent.t, agent.bandit)
    reward = pull(agent.bandit, agent.t, action)
    context = observe(agent.bandit, agent.t)

    # Compute regret 
    if reward.optimal_value !== nothing
        regret = reward.optimal_value - reward.value
    else
        regret = 0.0  # no regret value to be added
    end

    # Cumulate values
    agent.cum_regret += regret
    agent.cum_reward += reward.value

    # Learn with some defined sparsity
    if (rand(agent.rng) > agent.sparse)
        learn!(agent.policy, agent.t, context, action, reward)
    end

    # Record step
    step = Step(agent.t, context, action, reward) 

    # Increase time counter
    agent.t += 1

    return step
end


"""
    set_seed!(agent::Agent, seed::Integer=123)

Set seed for random number generator of `Agent` (an its `Bandit` and `Policy`)   

# Arguments
- `agent::Agent`: agent to change
- `seed:Integer`: seed to set
"""
function set_seed!(agent::Agent, seed::Integer=123)
    # Reset seed in rng
    Random.seed!(agent.rng, seed)
    agent.seed = seed
    
    # Do the same for bandit and policy
    set_seed!(agent.bandit, seed)
    set_seed!(agent.policy, seed)
end

"""
    get_seed(agent::Agent) -> Integer

Get seed for random number generator from `Agent`.   

# Arguments
- `agent::Agent`: agent to get seed from
"""
function get_seed(agent::Agent)
    return agent.seed
end


"""
    show(agent::Agent)
Prints information about agent into the specified I/O.
"""
function Base.show(io::IO, ::MIME"text/plain", agent::Agent)
    print(io, "$(agent.name)")
end
