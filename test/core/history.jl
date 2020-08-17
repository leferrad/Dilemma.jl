using DataFrames
using Dilemma
using Test

"""Happy test for History"""
function test_history_happy()
    # Create history
    history = History()

    # Update data
    data = DataFrame(
        t=[1, 2, 3, 1, 2], 
        k=[3, 3, 3, 3, 3], 
        d=[2, 2, 2, 2, 2], 
        sim=[1, 1, 1, 2, 2], 
        choice=[2, 2, 3, 2, 3],
        reward=[0.0, 0.0, 1.0, 0.0, 1.0], 
        optimal_arm=[3, 3, 3, 3, 3], 
        optimal_reward=[1.0, 1.0, 1.0, 1.0, 1.0],
        propensity=[0.0, 0.0, 1.0, 0.0, 1.0], 
        agent=["test_agent1", "test_agent1", "test_agent1", "test_agent1", "test_agent1"], 
        regret=[1.0, 1.0, 0.0, 1.0, 0.0],
        cum_reward=[0.0, 0.0, 1.0, 0.0, 1.0], 
        cum_regret=[1.0, 2.0, 2.0, 1.0, 1.0],
        seed=[123, 123, 123, 123, 123]
    )
    update_data!(history, data)

    # Test proper updates
    @test history.data == data
    @test history.meta == Dict("agents" => 1,"max_t" => 3,"repetitions" => 2,"min_t" => 2)
    @test history.stats.t == [1,2,3]
    @test history.stats.agent == ["test_agent1", "test_agent1", "test_agent1"]
    @test history.stats.sims_count == [2,2,1]
    @test history.stats.reward_μ == [0.0,0.5,1.0]
    @test history.stats.regret_μ == [1.0,0.5,0.0]
    @test all([isnan(x) || x == y for (x, y) in zip(history.stats.reward_σ, [0.0,0.7071067811865476, NaN])])
    @test all([isnan(x) || x == y for (x, y) in zip(history.stats.regret_σ, [0.0,0.7071067811865476, NaN])])
    @test history.stats.cum_reward_μ == [0.0,0.5,1.0]
    @test history.stats.cum_regret_μ == [1.0,1.5,2.0]
    @test all([isnan(x) || x == y for (x, y) in zip(history.stats.cum_reward_σ, [0.0,0.7071067811865476, NaN])])
    @test all([isnan(x) || x == y for (x, y) in zip(history.stats.cum_regret_σ, [0.0,0.7071067811865476, NaN])])

    # Update meta
    key, value, agent_name = "test_key", 123, "agent"
    update_meta!(history, key, value, agent_name=agent_name)

    @test agent_name in keys(history.meta)
    @test key in keys(history.meta[agent_name])
    @test history.meta[agent_name][key] == value

    update_meta!(history, key, value)  # no agent associated
    @test key in keys(history.meta)
    @test history.meta[key] == value

    # Save history
    filename = tempdir() * "/history.csv"
    write_history(history, filename)

    # Load history
    hist = read_history(filename)
    @test history.data == hist.data
end

"""Bad path test of History"""
function test_history_bad()    
    # Create history
    history = History()

    # Update with empty data
    empty_data = DataFrame()
    @test_throws ArgumentError update_data!(history, empty_data)
    
    # Update with data having wrong columns
    bad_data = DataFrame(
        a=[1, 2, 3], 
        b=[3, 3, 3], 
        c=[2, 2, 2] 
    )
    @test_throws ArgumentError update_data!(history, bad_data)

    # Bad read_history
    @test_throws ArgumentError read_history("/not/existing/path.csv")

    # Bad write_history
    @test_throws ArgumentError write_history(history, "/not/existing/path.csv")
end


@testset "history" begin
    @testset "unit" begin
        @testset "happy" begin
            test_history_happy()
        end
        @testset "bad" begin
            test_history_bad()
        end
    end
end
