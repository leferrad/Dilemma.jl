using Dilemma
using Distributions
using Test

function test_uniform_happy_1()
    a = [-1.0, 0.5, 0.0]
    b = [1.0, 1.0, 1.0]
    bandit = UniformBandit(a, b)
    @test bandit.k == length(a)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_uniform_happy_2()
    k, a, b = 5, -1, 1
    bandit = UniformBandit(k, a, b)
    @test bandit.k == k

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_uniform_happy_3()
    D = [Uniform(), Uniform(-5, 5)]
    bandit = UniformBandit(D)
    @test bandit.k == length(D)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

@testset "bandit_stochastic_uniform" begin
    @testset "unit" begin
        @testset "happy" begin
            test_uniform_happy_1()
            test_uniform_happy_2()
            test_uniform_happy_3()
        end
        @testset "bad" begin
        end
    end
end
