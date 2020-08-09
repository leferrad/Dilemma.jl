using Dilemma
using Random
using Test

seed = 123

"""Happy test of Step"""
function test_step_happy()
    # create valid Steps
    t1, c1, a1, r1, = 1, Context(), Action(1), Reward(1.0)
    s1 = Step(t1, c1, a1, r1)
end

"""Happy test of Agent"""
function test_agent_happy()
    # create valid Agents
    k, p = 3, 0.5
    agent1 = Agent(RandomPolicy(seed=seed), BernoulliBandit(k, p, seed=seed))
    @test agent1.name == "Agent: Policy=RandomPolicy, Bandit=StochasticBandit{Bernoulli{Float64}}"
    @test agent1.t == 1
    @test agent1.cum_regret == 0.0
    @test agent1.cum_reward == 0.0
    @test agent1.rng == MersenneTwister(seed)

    # test custom name
    agent2 = Agent(RandomPolicy(seed=seed), BernoulliBandit(k, p, seed=seed), name="custom_agent")
    @test agent2.name == "custom_agent"
end


"""Test reset_agent()!"""
function test_reset_agent()
    # Create agent
    k, p = 3, 0.5
    agent = Agent(RandomPolicy(seed=seed), BernoulliBandit(k, p, seed=seed))
    # Test initial state
    @test agent.t == 1
    @test agent.cum_reward == 0.0
    @test rand(agent.rng) == rand(MersenneTwister(seed))

    # Make some steps
    for i in 1:3
        do_step!(agent)
    end

    # Test changed values
    @test agent.t == 4
    @test agent.cum_reward != 0.0
    @test rand(agent.rng) != rand(MersenneTwister(seed))

    # Reset agent
    reset_agent!(agent)

    # Test reseted values
    @test agent.t == 1
    @test agent.cum_reward == 0.0
    @test agent.rng == MersenneTwister(seed)
end


"""Test seed methods"""
function test_seed_agent()
    # Create agent
    k, p = 3, 0.5
    agent = Agent(RandomPolicy(seed=seed), BernoulliBandit(k, p, seed=seed))

    # Test get_seed
    @test get_seed(agent) == seed

    # Test set_seed!
    seed2 = 321
    set_seed!(agent, seed2)
    @test get_seed(agent) == seed2
    @test rand(agent.rng) == rand(MersenneTwister(seed2))

    # Test getting same results after a reset of seed
    set_seed!(agent, seed2)
    steps1 = []
    for i in 1:3
        push!(steps1, do_step!(agent))
    end

    set_seed!(agent, seed2)
    steps2 = []
    for i in 1:3
        push!(steps2, do_step!(agent))
    end

    # same actions selected by policy
    @test all([s1.action.choice == s2.action.choice for (s1, s2) in zip(steps1, steps2)])
    # same results from bandit
    @test all([s1.reward.value == s2.reward.value for (s1, s2) in zip(steps1, steps2)])

end


"""Test do_step!()"""
function test_do_step_agent()
    # Create agent
    k, p = 3, 0.5
    agent = Agent(RandomPolicy(seed=seed), BernoulliBandit(k, p, seed=seed))

    # Make some steps
    steps = []
    n = 3
    for i in 1:n
        push!(steps, do_step!(agent))
    end

    # Test changed values
    @test agent.t == (n+1)
    @test agent.cum_reward == sum([s.reward.value for s in steps])
    @test agent.cum_regret == sum([(s.reward.optimal_value - s.reward.value) for s in steps])
    @test rand(agent.rng) != rand(MersenneTwister(seed))
end

"""Bad path test of Step"""
function test_step_bad()    
    # negative t
    @test_throws ArgumentError Step(-1, Context(), Action(1), Reward(1.0))
end


@testset "agent" begin
    @testset "unit" begin
        @testset "happy" begin
            test_step_happy()
            test_agent_happy()
            test_reset_agent()
            test_seed_agent()
            test_do_step_agent()
        end
        @testset "bad" begin
            test_step_bad()
        end
    end
end
