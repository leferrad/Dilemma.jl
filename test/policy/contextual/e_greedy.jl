using Distributions
using Test

using Dilemma

"""Create a dummy Contextual Bandit for testing purposes"""
mutable struct TestContextualBandit <: Bandit
    k::Int
    d::Union{Int, Nothing}

    function TestContextualBandit(k=3, d=5)
        new(k, d)
    end
end

# Extend methods to register TestContextualBandit
Dilemma.observe(bandit::TestContextualBandit, t::Integer) =  bandit.d === nothing ? Context() : get_dummy_context(bandit.d)  
Dilemma.pull(bandit::TestContextualBandit, t::Integer, action::Action) = get_dummy_reward()


"""Happy test of ContextualEpsilonGreedyPolicy methods"""
function test_happy_e_greedy()
    t = 1
    policy = ContextualEpsilonGreedyPolicy(0.1)

    k, d = 3, 5
    bandit = TestContextualBandit(k, d)

    # Test initialize behavior
    initialize!(policy, bandit)
    @test size(policy.θ, 1) == k

    # Test parameters conserved after learn!
    old_θ = deepcopy(policy.θ)
    action = choose(policy, t, bandit)
    @test action.choice in 1:k

    context = get_dummy_context(d) 
    reward = get_dummy_reward()
    learn!(policy, t, context, action, reward)

    @test old_θ != policy.θ

    # Test choose & learn! without calling initialize
    policy2 = ContextualEpsilonGreedyPolicy(0.1)
    action2 = choose(policy2, t, bandit)
    learn!(policy2, t, context, action2, reward)
end


"""Test mismatch between an initialized Policy and a bandit's k-arms"""
function test_bad_policy_parameters_mismatch_k()
    policy = ContextualEpsilonGreedyPolicy(0.1)

    # set initial parameters with some k
    bandit1 = TestContextualBandit(3, 3)
    initialize!(policy, bandit1)

    # use choose with other k
    bandit2 = TestContextualBandit(5, 3)
    @test_throws DimensionMismatch choose(policy, 1, bandit2)
end


"""Test mismatch between an initialized Policy and a bandit's dimension"""
function test_bad_policy_parameters_mismatch_d()
    policy = ContextualEpsilonGreedyPolicy(0.1)

    # set initial parameters with some d
    bandit1 = TestContextualBandit(5, 3)
    initialize!(policy, bandit1)

    # use choose with other d
    bandit2 = TestContextualBandit(5, 5)
    @test_throws DimensionMismatch choose(policy, 1, bandit2)
end


"""Test error obtained when Context has d == nothing"""
function test_bad_policy_context_d_nothing()
    policy = ContextualEpsilonGreedyPolicy(0.1)
    # set bandit with d == nothing
    bandit = TestContextualBandit(5, nothing)
    @test_throws ErrorException initialize!(policy, bandit)
end


"""Test bad call of learn! with a not initialized Policy"""
function test_bad_learn_policy_not_initialized()
    policy = ContextualEpsilonGreedyPolicy(0.1)

    t = 1
    context = get_dummy_context() 
    action = Action(1)
    reward = get_dummy_reward()
    @test_throws ErrorException learn!(policy, t, context, action, reward)
end

@testset "e_greedy" begin
    @testset "unit" begin
        @testset "happy" begin
            test_happy_e_greedy()
        end
        @testset "bad" begin
            test_bad_policy_parameters_mismatch_k()
            test_bad_policy_parameters_mismatch_d()
            test_bad_policy_context_d_nothing()
            test_bad_learn_policy_not_initialized()
        end
    end
end
