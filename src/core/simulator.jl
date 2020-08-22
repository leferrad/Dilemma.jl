using Dilemma
using Dates
using DataFrames
using ProgressMeter
using Random

export
    Simulator,
    reset_simulator!,
    run_simulation!

"""
    Simulator(agents, horizon, repetitions; kwargs...) -> Simulator

A `Simulator` takes one or more `Agent` instances, a horizon (the length
of an individual simulation) and the number of repetitions (how many times
to repeat each simulation).
It then runs all simulations (in parallel, by default), keeping a log of
all `Policy` and `Bandit` interactions in a `History` instance.

!!! note
    To be able to fairly evaluate and compare each agent's performance,
    and to make sure that simulations are replicable, for each separate agent,
    seeds are set equally and deterministically for each agent over all
    horizon x repetitions time steps.

# Arguments

- `agents::Union{Agent, Vector{Agent}}`: The agent/s to be used in the simulation
- `horizon::Integer`: The number of pulls or time steps to run each agent,
    where ``t = {1, ..., T}``
- `repetitions::Integer`:  How many times to repeat each agent's simulation over
    ``t = {1, ..., T}``, with a new seed on each repeat (itself
    deterministically).

# Keywords

- `save_context::Bool`: To indicate if context fields must be saved in `History`. 
- `do_parallel::Bool`: To indicate that repetitions must run in parallel
    (through `Threads` methods).
- `seed::Union{Nothing, Integer}`: If nothing, simulation is not reproducible
- `interval::Integer`: Frequency to save records in `History`.

# Examples
```julia-repl
julia> using Dilemma
julia> bandit = BernoulliBandit(5, 0.5)
julia> agents = [
            Agent(EpsilonGreedyPolicy(0.1), bandit, "ϵ-greedy"),
            Agent(RandomPolicy(), bandit, "Random"),
       ]
julia> horizon = 100
julia> repetitions = 10
julia> simulator = Simulator(agents, horizon, repetitions)
julia> hist = run_simulation!(simulator)
julia> size(hist)
(2970, 15)
```
"""
mutable struct Simulator
    agents::Union{Agent, Vector{Agent}}
    horizon::Int
    repetitions::Int
    save_context::Bool
    do_parallel::Bool
    seed::Union{Nothing, Int}
    interval::Int

    history::History
    n_agents::Int

    function Simulator(
        agents::Union{Agent, Vector{Agent}},
        horizon::Integer=100,
        repetitions::Integer=10
        ;
        save_context::Bool=false,
        do_parallel::Bool=true,
        seed::Union{Nothing, Integer}=123,
        interval::Integer=1
    )

        if !(agents isa Vector)
            agents = [agents]
        end

        if horizon <= 0
            throw(ArgumentError("Argument 'horizon' must be > 0"))
        end

        if repetitions <= 0
            throw(ArgumentError("Argument 'repetitions' must be > 0"))
        end

        n_agents = length(agents)
        history = History() 

        simulator = new(agents, horizon, repetitions,
                    save_context, do_parallel,
                    seed, interval,
                    history, n_agents)

        reset_simulator!(simulator)

        return simulator
    end
end

function reset_simulator!(simulator::Simulator)
    # reset history data and meta data tables
    simulator.history = History(save_context=simulator.save_context) 
    update_meta!(simulator.history, "horizon", simulator.horizon)
    update_meta!(simulator.history, "repetitions", simulator.repetitions)
    update_meta!(simulator.history, "n_agents", simulator.n_agents)

    # process agents
    agent_names = Dict()

    for i in 1:simulator.n_agents
        # get agent
        agent = simulator.agents[i]
        reset_agent!(agent)  # reset agent metadata

        # check names collisions to rename agents
        cur_agent = agent.name

        if cur_agent in keys(agent_names)
            agent_names[cur_agent] += 1
        else
            agent_names[cur_agent] = 1
        end

        cur_agent_n = agent_names[cur_agent] 

        if cur_agent_n > 1
            new_name = string(cur_agent, ".", cur_agent_n)
            set_name!(simulator.agents[i], new_name)
        end

        # add names to history meta
        agent_name = simulator.agents[i].name
        bandit_name = string(typeof(simulator.agents[i].bandit))
        policy_name = string(typeof(simulator.agents[i].policy))

        update_meta!(simulator.history, "bandit", bandit_name, agent_name=agent_name)
        update_meta!(simulator.history, "policy", policy_name, agent_name=agent_name)
    end

end

function run_step(
    agent::Agent,
    simulation::Integer,
    horizon::Integer=100,
    interval::Integer=1;
    save_context::Bool=false,
    seed::Integer=123
)

    @debug "Running simulation $simulation with agent $(agent.name)"
    @debug "Feeding agent with seed=$seed"
    set_seed!(agent, seed)

    data = DataFrame(t=Int[], k=Int[], d=Any[], sim=Int[], choice=Int[],
                     reward=Float64[], optimal_arm=Any[], optimal_reward=Any[],
                     propensity=Float64[], agent=String[], regret=Float64[],
                     cum_reward=Float64[], cum_regret=Float64[],
                     seed=Int[])


    save_context && (agent.bandit.d !== nothing) ?
        context = zeros(Union{Float64, Missing}, horizon, agent.bandit.d) :
        context = nothing

    context_cols = nothing
    loop_time = 0  # later replaced with step.t - 1

    while loop_time < horizon
        step = do_step!(agent)

        loop_time = step.t - 1  # 0-index based

        if (step.reward !== nothing) & (loop_time % interval == 0)

            # TODO: save even when reward is nothing?

            # TODO: have a Record struct to resolve values

            row = (
                step.t, 
                agent.bandit.k,
                step.context.d,
                simulation,
                step.action.choice,
                step.reward.value,
                step.reward.optimal_arm,
                step.reward.optimal_value,
                0.0,  # TODO: propensity??
                agent.name,
                step.regret,
                agent.cum_reward,
                agent.cum_regret,
                seed  # simulator & agent seed

                # TODO: compute cum_reward_rate -> cum_reward / t
                # TODO: compute cum_regret_rate -> cum_regret / t
            )

            push!(data, row)

            if context !== nothing
                if step.context.x === nothing
                    if step.context.d === nothing
                        # TODO: review this
                        x = context_cols !== nothing ?
                            [missing for _ in context_cols] :
                            [missing for _ in 1:size(context, 2)]
                    else
                        x = [missing for _ in 1:step.context.d]
                    end
                else
                    x = collect(step.context)
                    context_cols = step.context.keys  # TODO: overwritten?
                end
                context[loop_time, :] = x
            end

        end # if

    end # while

    if context !== nothing
        context = context[1:loop_time, :]
        # TODO: check proper context_cols?
        for i in 1:size(context,2)
            data[:,context_cols[i]] = context[:, i]
        end
    end

    return data
end


function _loop_simulator_multi_thread(simulator::Simulator)
    # Set seed to achieve reproducible results (if seed is not null)
    simulator.seed isa Integer ? Random.seed!(simulator.seed) : nothing

    n_total = simulator.repetitions * simulator.n_agents
    # Indexes of agent and simulation to be executed in parallel
    shared_agent_idxs = Vector{Int64}(undef, n_total)
    shared_sims_idxs =  Vector{Int64}(undef, n_total)

    idx = 1
    for s in 1:simulator.repetitions, a in 1:simulator.n_agents
        shared_agent_idxs[idx] = a
        shared_sims_idxs[idx] = s
        idx += 1
    end

    # Vector of histories to be filled & returned
    histories = Vector{DataFrame}(undef, n_total)
    # Progress bar (manually, to avoid problems with multiple threads)
    p = Progress(n_total, "Progress: ");
    update!(p, 0)
    ∇ = Threads.Atomic{Int}(0)

    seeds = Random.rand(1:1000, n_total)
    # TODO: use rng instead?

    # Run loop
    @inbounds Threads.@threads for i in 1:n_total
        @inbounds a, sim = shared_agent_idxs[i], shared_sims_idxs[i]
        agent = deepcopy(simulator.agents[a])  # TODO: needed copy?
        @inbounds histories[i] = run_step(agent,
                                          sim,
                                          simulator.horizon,
                                          simulator.interval,
                                          save_context=simulator.save_context,
                                          seed=seeds[i])
        # Update progress bar
        Threads.atomic_add!(∇, 1)
        Threads.threadid() == 1 && update!(p, ∇[])  # only in 1 thread (faster)
    end
    update!(p, ∇[])  # Final update

    # bind results
    histories = vcat(histories...)
    data = histories[(histories.sim .> 0) .& (histories.t .> 0), :]

    return data
end



function _loop_simulator_single_thread(simulator::Simulator)
    # Set seed to achieve reproducible results (if seed is not null)
    simulator.seed isa Integer ? Random.seed!(simulator.seed) : nothing

    n_total = simulator.repetitions * simulator.n_agents

    # Vector of histories to be filled & returned
    histories = Vector{DataFrame}(undef, n_total)

    seeds = Random.rand(1:1000, n_total)
    # TODO: use rng instead?

    # Run loop
    i = 1
    @showprogress 1 "Progress: " for sim in 1:simulator.repetitions, a in 1:simulator.n_agents
        agent = deepcopy(simulator.agents[a])  # TODO: needed copy?
        @inbounds histories[i] = run_step(agent,
                                            sim,
                                            simulator.horizon,
                                            simulator.interval,
                                            save_context=simulator.save_context,
                                            seed=seeds[i])
        i += 1
    end

    # bind results
    histories = vcat(histories...)  # TODO: do not replace same variable with different type
    data = histories[(histories.sim .> 0) .& (histories.t .> 0), :]  # TODO: needed?

    return data
end

"""
    run_simulation!(simulator::Simulator) -> History

Run a `Simulator` instance to test a set of `Agent` instances

# Arguments
- `simulator::Simulator`: instance to be runned.

# Returns
- `History`: results obtained after simulation.

# Throws
...

# Examples
```julia-repl
julia> using Dilemma
julia> bandit = BernoulliBandit(5, 0.5)
julia> agents = [
            Agent(ContextualEpsilonGreedyPolicy(0.1), bandit, "ctx ϵ-greedy"),
            Agent(ContextuaLinTSPolicy(0.1), bandit, "ctx LinTS"),
            Agent(ContextuaLinUCBPolicy(0.1), bandit, "ctx LinUCB"),
       ]
julia> horizon = 100
julia> repetitions = 10
julia> simulator = Simulator(agents, horizon, repetitions)
julia> hist = run_simulation!(simulator)
julia> size(hist)
(2970, 15)
```
"""
function run_simulation!(simulator::Simulator)
    # Random.seed!(simulator.seed)

    n_threads = simulator.do_parallel ? Threads.nthreads() : 1

    # some info messages
    @info "Starting simulation..."
    @info "Horizon: $(simulator.horizon)"
    @info "N° of repetitions: $(simulator.repetitions)"
    @info "N° of threads: $(n_threads)"
    println()

    # Choose func based on execution mode (serial vs parallel)
    loop_func = (n_threads > 1) ? 
        _loop_simulator_multi_thread :
        _loop_simulator_single_thread

    # Run loop 
    duration = @elapsed data = loop_func(simulator)

    # Update history
    update_data!(simulator.history, data)

    println()
    @info "Completed simulation in $(round(duration, digits=3)) seconds"

    update_meta!(simulator.history, "sim_end_time", Dates.now())
    update_meta!(simulator.history, "sim_total_duration", duration)

    @debug "Computing statistics..."
    # TODO: not always necessary, add option arg to class?
    update_stats!(simulator.history)

    return simulator.history

end
