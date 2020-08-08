using Distributions
using Test

using Dilemma

"""Happy test of ContextualEpsilonGreedyPolicy methods"""
function test_happy()
    t = 1
    policy = ContextualEpsilonGreedyPolicy(0.1)

    k, d = 5, 5
    context = get_dummy_context(k, d)

    initialize!(policy, context)
    @test size(policy.θ, 1) == k
    @test size(policy.θ[1].A) == (d, d)
    @test size(policy.θ[1].b) == (d, 1)

    old_θ = deepcopy(policy.θ)

    action = choose(policy, t, context)
    @test action.choice in 1:k

    reward = get_dummy_reward()
    learn!(policy, t, context, action, reward)

    @test old_θ != policy.θ

end

# TODO: test happy when parameters were not set yet

"""Test mismatch between an initialized Policy and a bandit"""
function test_bad_policy_parameters_mismatch_context_k()
    policy = ContextualEpsilonGreedyPolicy(0.1)

    # set initial parameters with some k
    context1 = get_dummy_context(5, 5)
    initialize!(policy, context1)

    # use get_action with other k
    context2 = get_dummy_context(3, 3)
    @test_throws DimensionMismatch choose(policy, 1, context2)
end

"""Test mismatch between Policy parameter θ for some arm and context.d"""
function test_bad_policy_parameter_arm_mismatch_context_d()
    policy = ContextualEpsilonGreedyPolicy(0.1)

    # set initial parameters with some d
    context1 = get_dummy_context(5, 5)
    initialize!(policy, context1)

    # use get_action with other d
    context2 = get_dummy_context(5, 3)
    @test_throws DimensionMismatch choose(policy, 1, context2)
end

@testset "e_greedy" begin
    @testset "unit" begin
        @testset "happy" begin
            test_happy()
        end
        @testset "bad" begin
            test_bad_policy_parameters_mismatch_context_k()
            test_bad_policy_parameter_arm_mismatch_context_d()
        end
    end
end
