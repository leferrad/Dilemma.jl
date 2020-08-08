using Dilemma
using Test

mutable struct TestPolicy <: Policy
    """Subtype for testing"""
end

function test_policy_not_implemented()
    policy = TestPolicy()
    @test_throws MethodError choose(policy)
    @test_throws MethodError initialize!(policy)
    @test_throws MethodError learn!(policy)
end

@testset "policy" begin
    @testset "unit" begin
        test_policy_not_implemented()
    end
end
