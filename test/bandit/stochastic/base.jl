using Dilemma
using Distributions
using Test

seed = 123

"""Happy path testing"""
function test_bandit_happy()
    D = [Uniform(), Bernoulli()]
    bandit = StochasticBandit(D, seed=seed)
    @test bandit.k == length(D)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

"""Happy path testing, bandit with offset > 0"""
function test_bandit_happy_offset()
    D = [Uniform(), Bernoulli()]
    t = 1
    offset = 5.0
    bandit = StochasticBandit(D, offset=offset, seed=seed)

    # Bernoulli arm with reward values in [0,1]
    action = Action(2)
    reward = pull(bandit, t, action)
    @test 0.0 <= (reward.value - offset) <= 1.0
end

# TODO: test arms

"""Test show call"""
function test_bandit_show()
    D = [Uniform(), Bernoulli()]
    bandit = StochasticBandit(D)
    println("Testing show()...")
    show(bandit)
    println("")
end

"""Test bad action in pull() call"""
function test_bad_action_pull()
    D = [Uniform(), Bernoulli()]
    bandit = StochasticBandit(D, seed=seed)

    # bandit has 2 arms, 5 is an invalid arm
    bad_action = Action(5)

    @test_throws ArgumentError pull(bandit, 1, bad_action)
end

@testset "bandit_stochastic_base" begin
    @testset "unit" begin
        @testset "happy" begin
            test_bandit_happy()
            test_bandit_happy_offset()
            test_bandit_show()
        end
        @testset "bad" begin
            test_bad_action_pull()
        end
    end
end
