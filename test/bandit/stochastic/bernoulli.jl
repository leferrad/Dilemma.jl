using Dilemma
using Distributions
using Test

function test_bernoulli_happy_1()
    p = [0.1, 0.2, 0.2, 0.3, 0.2]
    bandit = BernoulliBandit(p)
    @test bandit.k == length(p)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_bernoulli_happy_2()
    k, p = 5, 0.3
    bandit = BernoulliBandit(k, p)
    @test bandit.k == k

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_bernoulli_happy_3()
    D = [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(0.3)]
    bandit = BernoulliBandit(D)
    @test bandit.k == length(D)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

@testset "bandit_stochastic_bernoulli" begin
    @testset "unit" begin
        @testset "happy" begin
            test_bernoulli_happy_1()
            test_bernoulli_happy_2()
            test_bernoulli_happy_3()
        end
        @testset "bad" begin
        end
    end
end
