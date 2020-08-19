using Dilemma
using Test

mutable struct TestPolicy <: Policy
    """Subtype for testing"""
end

"""Test Policy having methods not implemented"""
function test_policy_not_implemented()
    policy = TestPolicy()
    @test_throws MethodError choose(policy)
    @test_throws MethodError initialize!(policy)
    @test_throws MethodError learn!(policy)
end

"""Test get_seed() call"""
function test_policy_get_seed()
    policy = TestPolicy()

    seed = get_seed(policy)
    @test seed === nothing
end

@testset "policy" begin
    @testset "unit" begin
        test_policy_not_implemented()
        test_policy_get_seed()
    end
end
