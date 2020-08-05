using Dilemma
using Distributions
using LinearAlgebra

"""Get a dummy Context for testing purposes"""
function get_dummy_context(d::Integer=5)
    x = rand(d)
    return Context(x)
end

"""Get a dummy Action for testing purposes"""
function get_dummy_action(choice::Integer=1, k::Integer=5)
    return Action(choice, k=k)
end

"""Get a dummy Reward for testing purposes"""
function get_dummy_reward(reward::Real=0.0, optimal_arm::Integer=1,
                          optimal_reward::Real=1.0)
    return Reward(reward, optimal_arm, optimal_reward)
end
