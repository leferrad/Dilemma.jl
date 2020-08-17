using CSV
using Dilemma 
using DataFrames: DataFrame
using DataFramesMeta: @transform, combine, groupby
using Distributions: Normal
using Statistics: mean, std


export
    History,
    update_meta!,
    update_data!,
    read_history,
    write_history

"""
    History(;save_context)

Structure to record all the interactions between `Agent` instances
and a `Simulator` during a simulation. Information is stored into
a `DataFrame`, containing aggregations that could be plotted with
`Dilemma.plot` methods.

This is handled directly by a `Simulator`, so it should be transparent for the user in the program.

# Fields
- `save_context::Bool`: to indicate if context values should be also recorded
- `data::Union{DataFrame,Nothing}`: structure where records are stored (`nothing` when simulation is not started) 
- `meta::Dict`: structure to store meta data of simulations provided by the user 
- `stats::Union{DataFrame,Nothing}`: struct where cumulative statistics from simulation are stored
- `columns::Vector{String}`: list of columns to be stored in `data`
"""
mutable struct History
    save_context::Bool
    data::Union{DataFrame, Nothing}
    meta::Dict
    stats::Union{DataFrame, Nothing}
    columns::Vector{String}

    function History(;save_context::Bool=false)
        # Expected columns in "data"
        columns = [
            "t", "agent", "k", "d", "sim", "choice", "reward",
            "optimal_arm", "optimal_reward", "propensity",
            "regret", "cum_reward", "cum_regret", "seed"
        ]

        # Create new instance
        hist = new(save_context, nothing, Dict(), nothing, columns)

        return hist
    end
end

"""
    update_meta!(history, key, value, agent_name) -> History

Update meta data of a `History` instance, by adding a new entry.   

# Arguments
- `history::History`: history to be updated
- `key::String`: key of meta data to update
- `value::Any`: value of new entry to add
- `agent_name::Union{String, Nothing}`: optionally, an agent name 
    to associate with the new entry
"""
function update_meta!(
    history::History, 
    key::String, 
    value::Any; 
    agent_name::Union{String, Nothing}=nothing
)
    # Entry to add in meta data
    entry = Dict(key => value)

    if agent_name === nothing
        # Add entry in top of meta data
        history.meta = merge(history.meta, entry)
    else
        # Entry is associated to an agent name
        if !(agent_name in keys(history.meta))
            history.meta[agent_name] = Dict()
        end
        history.meta[agent_name] = merge(history.meta[agent_name], entry)
    end

    return history
end

"""
    update_data!(history, data) -> History

Update data of a `History` instance, by setting a new instance.   

# Arguments
- `history::History`: history to be updated
- `data::DataFrame`: new instance
"""
function update_data!(history::History, data::DataFrame)
    # Check proper size
    if size(data,1) == 0
        throw(ArgumentError("Argument 'data' must be a not empty DataFrame"))
    end

    # Check proper columns
    if !all([c in names(data) for c in history.columns])
        throw(ArgumentError("Argument 'data' must have all of these columns: $(history.columns). "*
                            "Got: $(names(data))"))
    end

    # Update data
    history.data = data

    # Compute stats from new data
    update_stats!(history)

    return history
end

"""
    update_stats!(history)

Update statistics of a `History` instance, by computing aggregations over `history.sdata`.
Used internally by `update_data!`.

# Arguments
- `history::History`: history to be updated
"""
function update_stats!(history::History)
    # Checks over history where done in update_data!()

    # Group by agent, sim and obtain
    # max of t as a reference of the longest sim for the given agent in sim
    gdf = combine(
        groupby(history.data, [:agent, :sim]), 
        :t => maximum => :t
    )

    # Update meta data with aggregations
    update_meta!(history, "min_t", minimum(gdf.t))
    update_meta!(history, "max_t", maximum(gdf.t))
    update_meta!(history, "agents", length(unique(history.data.agent)))
    update_meta!(history, "repetitions", length(unique(history.data.sim)))

    # Compute stats over aggregated data
    # μ  -> mean of aggregation
    # σ  -> std dev of aggregation
    history.stats = combine(
        groupby(history.data, [:t, :agent]),

        :reward => length => :sims_count,

        :reward => mean => :reward_μ,
        :reward => std => :reward_σ,

        :regret => mean => :regret_μ,
        :regret => std => :regret_σ,

        :cum_reward => mean => :cum_reward_μ,
        :cum_reward => std => :cum_reward_σ,

        :cum_regret => mean => :cum_regret_μ,
        :cum_regret => std => :cum_regret_σ,
    )
end

"""
    read_history(filename; delim=",") -> History

Read a CSV containing records to be loaded in a `History` instance.
This method will only load the content of `history.data` (not the meta content).

# Arguments
- `filename::String`: path to CSV file with records
- `delim`: delimiter to parse CSV file
"""
function read_history(filename::String; delim=",")
    data = CSV.File(filename, delim=delim) |> DataFrame 
    history = History()
    update_data!(history, data)

    return history
end

"""
    write_history(history, filename; delim=",")

Write a CSV containing records from a `History` instance.
This method will only save the content of `history.data` (not the meta content).

# Arguments
- `history::History`: history with records to save
- `filename::String`: path to output CSV file
- `delim::String`: delimitator to parse CSV file
"""
function write_history(history::History, filename::String; delim=",")
    CSV.write(filename, history.data, delim=delim)
end