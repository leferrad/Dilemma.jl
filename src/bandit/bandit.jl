using Dilemma
using Random

export
    Bandit,
    observe,
    pull,
    get_seed,
    set_seed!

# TODO: add name field? with default value

"""
    Bandit
An abstract type for all implementations of Bandit algorithms

# Common methods
A `Bandit` is a ...

* [`observe`](@ref) : get the context for a given moment
* [`pull`](@ref) : get the reward from pulling a chosen arm
"""
abstract type Bandit end

"""
Return a list with number of arms bandit.k,
number of feature dimensions bandit.d and, where
applicable, a bandit.d dimensional context vector or
bandit.d x bandit.k dimensional context matrix X.
"""
function observe end

"""
# Return a list with the reward of the chosen arm and, if available,
# optimal arm reward and index
"""
function pull end


function get_seed(bandit::Bandit)
    hasproperty(bandit, :rng) && (bandit.rng isa AbstractRNG) ?
        reinterpret(Int32, bandit.rng.seed)[1] :
        nothing
end


function set_seed!(bandit::Bandit, seed::Integer=123)
    hasproperty(bandit, :rng) && (bandit.rng isa AbstractRNG) ?
        Random.seed!(bandit.rng, seed) :
        nothing
end
