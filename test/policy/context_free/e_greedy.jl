using Distributions
using Test

using Dilemma

"""Happy test of EpsilonGreedyPolicy methods"""
function test_happy()
    t = 1
    policy = EpsilonGreedyPolicy(0.1)

    k, p = 5, 0.5
    bandit = BernoulliBandit(k, p)

    initialize!(policy, bandit)
    @test size(policy.θ, 1) == k

    old_θ = deepcopy(policy.θ)

    action = choose(policy, t, bandit)
    @test action.choice in 1:k

    context = get_dummy_context()
    reward = get_dummy_reward()
    learn!(policy, t, context, action, reward)

    @test old_θ != policy.θ

end

# TODO: test happy when parameters were not set yet

"""Test mismatch between an initialized Policy and a bandit"""
function test_bad_policy_parameters_mismatch_k()
    policy = EpsilonGreedyPolicy(0.1)

    # set initial parameters with some k
    bandit1 = BernoulliBandit(5, 0.5)
    initialize!(policy, bandit1)

    # use get_action with other k
    bandit2 = BernoulliBandit(3, 0.5)
    @test_throws DimensionMismatch choose(policy, 1, bandit2)
end

@testset "e_greedy" begin
    @testset "unit" begin
        @testset "happy" begin
            test_happy()
        end
        @testset "bad" begin
            test_bad_policy_parameters_mismatch_k()
        end
    end
end