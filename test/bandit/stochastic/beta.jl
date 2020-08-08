using Dilemma
using Distributions
using Test

function test_beta_happy_1()
    a = [1, 5, 1]
    b = [2, 5, 10]
    bandit = BetaBandit(a, b)
    @test bandit.k == length(a)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_beta_happy_2()
    k, a, b = 5, 1, 5
    bandit = BetaBandit(k, a, b)
    @test bandit.k == k

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_beta_happy_3()
    D = [Beta(), Beta(1, 5)]
    bandit = BetaBandit(D)
    @test bandit.k == length(D)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

@testset "bandit_stochastic_beta" begin
    @testset "unit" begin
        @testset "happy" begin
            test_beta_happy_1()
            test_beta_happy_2()
            test_beta_happy_3()
        end
        @testset "bad" begin
        end
    end
end
