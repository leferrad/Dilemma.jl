using Dilemma
using Test

mutable struct TestBandit <: Bandit
    """Subtype for testing"""
end


"""Test get_seed() call"""
function test_policy_get_seed()
    bandit = TestBandit()

    seed = get_seed(bandit)
    @test seed === nothing
end

@testset "bandit" begin
    @testset "unit" begin
        test_bandit_get_seed()
    end
end
