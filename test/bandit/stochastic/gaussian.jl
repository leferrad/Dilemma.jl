using Dilemma
using Distributions
using Test

function test_gaussian_happy_1()
    μ = [0.1, 0.2, 0.2, 0.3, 0.2]
    σ = [0.1, 0.1, 0.2, 0.2, 0.1]
    bandit = GaussianBandit(μ, σ)
    @test bandit.k == length(μ)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_gaussian_happy_2()
    k, μ, σ = 5, 0.3, 0.1
    bandit = GaussianBandit(k, μ, σ)
    @test bandit.k == k

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

function test_gaussian_happy_3()
    D = [Normal(), Normal(0.5, 0.1)]
    bandit = GaussianBandit(D)
    @test bandit.k == length(D)

    t = 1
    context = observe(bandit, t)
    # no contextual bandit
    @test context.x === nothing

    action = Action(1)
    reward = pull(bandit, t, action)
    @test reward isa Reward
end

@testset "bandit_stochastic_gaussian" begin
    @testset "unit" begin
        @testset "happy" begin
            test_gaussian_happy_1()
            test_gaussian_happy_2()
            test_gaussian_happy_3()
        end
        @testset "bad" begin
        end
    end
end
