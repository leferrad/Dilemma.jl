using Dilemma
using Test


"""Happy test for Simulator"""
function test_simulator_happy(;do_parallel=true, seed=123)
    agents = Agent(
        RandomPolicy(),
        BernoulliBandit(5, 0.5), 
        name="TestAgent")
    horizon, repetitions = 20, 10
    simulator = Simulator(agents, horizon, repetitions, 
        do_parallel=do_parallel, seed=seed)

    @test simulator.n_agents == 1
    @test simulator.history isa History

    hist = run_simulation!(simulator)

    # TODO: assert exact results

    # TODO: test things

    # TODO: reset_simulator!


    return hist
end

"""Test Simulator with do_parallel=true"""
function test_simulator_parallel()
    test_simulator_happy(do_parallel=true)
end

"""Test Simulator with do_parallel=false"""
function test_simulator_serial()
    test_simulator_happy(do_parallel=false)
end

"""Test Simulator with reproducible results (deterministic)"""
function test_simulator_reproducible()
    h1 = test_simulator_happy(seed=123)
    h2 = test_simulator_happy(seed=123)

    @test h1.data == h2.data
end


"""Test Simulator with no reproducible results (stochastic)"""
function test_simulator_not_reproducible()
    h1 = test_simulator_happy(seed=nothing)
    h2 = test_simulator_happy(seed=nothing)

    @test h1.data != h2.data
end

# TODO: text with a contextual bandit


"""Test Simulator with 1 agent"""
function test_simulator_1_agent()
    agent = Agent(
        RandomPolicy(),
        BernoulliBandit(5, 0.5), 
        name="TestAgent")
    horizon, repetitions = 20, 10
    simulator = Simulator(agent, horizon, repetitions)

    @test simulator.agents isa Vector && length(simulator.agents) == 1

end


"""Test Simulator with 3 agents"""
function test_simulator_3_agents()
    agents = [
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent1"),
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent2"),
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent3"),
    ]
    horizon, repetitions = 20, 10
    simulator = Simulator(agents, horizon, repetitions)

    @test simulator.agents isa Vector && length(simulator.agents) == length(agents)
end



"""Test reset_simulator!"""
function test_reset_simulator()
    # all agents with same name, to be renamed
    agents = [
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent"),
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent"),
        Agent(
            RandomPolicy(),
            BernoulliBandit(5, 0.5), 
            name="TestAgent"),
    ]
    horizon, repetitions = 20, 10
    simulator = Simulator(agents, horizon, repetitions)

    # make some loops
    hist = run_simulation!(simulator)

    # call reset
    reset_simulator!(simulator)

    @test simulator.history.data === nothing
    @test simulator.history.meta["horizon"] == horizon
    @test simulator.history.meta["repetitions"] == repetitions
    @test simulator.history.meta["n_agents"] == length(agents)

    # test renamed agents
    agent_names = [a.name for a in simulator.agents]
    @test agent_names == ["TestAgent", "TestAgent.1", "TestAgent.2"]
end



"""Test bad path Simulator"""
function test_simulator_bad()
    agent = Agent(
        RandomPolicy(),
        BernoulliBandit(5, 0.5), 
        name="TestAgent")

    # test horizon <= 0, repetitions <= 0
    @test_throws ArgumentError Simulator(agent, -1, 10)
    @test_throws ArgumentError Simulator(agent, 1, 0)
end

@testset "simulator" begin
    @testset "unit" begin
        @testset "happy" begin
            test_simulator_serial()
            test_simulator_parallel()
            test_simulator_1_agent()
            test_simulator_3_agents()
        end

        @testset "bad" begin
            test_simulator_bad()
        end

        @testset "reproducibility" begin
            test_simulator_reproducible()
            test_simulator_not_reproducible()
        end

    end
end
