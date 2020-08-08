using Dilemma

export
    Policy,
    Theta,
    choose,
    initialize!,
    learn!,
    get_seed,
    set_seed!


abstract type Policy end

abstract type Theta end

function choose end

function initialize! end

function learn! end


function get_seed(policy::Policy)
    hasproperty(policy, :rng) && (policy.rng isa AbstractRNG) ?
        reinterpret(Int32, policy.rng.seed)[1] :
        nothing
end


function set_seed!(policy::Policy, seed::Integer=123)
    hasproperty(policy, :rng) && (policy.rng isa AbstractRNG) ?
        Random.seed!(policy.rng, seed) :
        nothing
end
