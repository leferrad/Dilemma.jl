using Dilemma
using Test


"""Happy test of Action"""
function test_action_happy()
    space = (:arm1, :arm2, :arm3)
    choice = 1

    # create valid Actions
    a1 = Action(choice, space)
    a2 = Action(choice, k=length(space))

    @test length(a1.space) == length(space)
    @test length(a2.space) == length(space)
end

"""Bad path test of Action"""
function test_action_bad()    
    # negative choice
    @test_throws ArgumentError Action(-1, k=3)
    # choice out of space range
    @test_throws ArgumentError Action(10, k=3)
end


@testset "action" begin
    @testset "unit" begin
        @testset "happy" begin
            test_action_happy()
        end
        @testset "bad" begin
            test_action_bad()
        end
    end
end
