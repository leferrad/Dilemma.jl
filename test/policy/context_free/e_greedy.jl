using Dilemma
using Distributions
using Test

seed = 123


"""Happy test of EpsilonGreedyPolicy methods"""
function test_happy()
    t = 1
    policy = EpsilonGreedyPolicy(0.1, seed=seed)

    k, p = 5, 0.5
    bandit = BernoulliBandit(k, p, seed=seed)

    initialize!(policy, bandit)
    @test size(policy.θ, 1) == k

    old_θ = deepcopy(policy.θ)

    action = choose(policy, t, bandit)
    @test action.choice in 1:k

    context = get_dummy_context()
    reward = get_dummy_reward()
    learn!(policy, t, context, action, reward)

    @test old_θ != policy.θ

    # Test choose & learn! without calling initialize
    policy2 = EpsilonGreedyPolicy(0.1, seed=seed)
    action2 = choose(policy2, t, bandit)
    learn!(policy2, t, context, action2, reward)

end

"""Happy test of EpsilonGreedyPolicy in a loop of learning"""
function test_happy_loop()
    t = 1
    t = 1
    policy = EpsilonGreedyPolicy(0.1, seed=seed)

    k, p = 5, 0.5
    bandit = BernoulliBandit(k, p, seed=seed)

    initialize!(policy, bandit)

    traces = []

    for i in 1:10
        action = choose(policy, t, bandit)
        context = observe(bandit, t)
        reward = pull(bandit, t, action)
        learn!(policy, t, context, action, reward)

        push!(traces, (t=t, context=context, action=action, reward=reward))
    end

    # Test more than 1 action selected
    @test length(Set([tr[:action] for tr in traces])) > 1
end


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


@testset "context_free_e_greedy" begin
    @testset "unit" begin
        @testset "happy" begin
            test_happy()
            test_happy_loop()
        end
        @testset "bad" begin
            test_bad_policy_parameters_mismatch_k()
        end
    end
end
