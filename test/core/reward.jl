using Dilemma
using Test


"""Happy test of Reward"""
function test_reward_happy()
    # create valid Rewards
    r1 = Reward(1.0)
    r2 = Reward(2.0, 1, 3.0)
end

"""Bad path test of Reward"""
function test_reward_bad()    
    # optimal_value lower than value
    @test_throws ArgumentError Reward(5.0, 1, 1.0)
end


@testset "reward" begin
    @testset "unit" begin
        @testset "happy" begin
            test_reward_happy()
        end
        @testset "bad" begin
            test_reward_bad()
        end
    end
end
