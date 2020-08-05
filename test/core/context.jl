using Dilemma
using Distributions
using Test


"""Happy test of Context"""
function test_context_happy()
    # test empty constructor
    ctx1 = Context()
    @test ctx1.x === nothing
    @test ctx1.d === nothing
    @test ctx1.keys === nothing
    @test ctx1.values === nothing
    @test collect(ctx1) === nothing

    # test constructor with namedtuple
    x2 = (a=1, b="2", c=3.0)

    ctx2 = Context(x2)
    @test ctx2.x == (a=1, b="2", c=3.0)
    @test ctx2.d == 3
    @test ctx2.keys == (:a, :b, :c)
    @test ctx2.values == (1, "2", 3.0)
    @test collect(ctx2) == [1, "2", 3.0]

    # test constructor with dict
    x3 = Dict(:a=>1, :b=>2, :c=>3)

    # TODO: what about order?
    ctx3 = Context(x3)
    @test ctx3.x == (a=1, b=2, c=3)
    @test ctx3.d == 3
    @test ctx3.keys == (:a, :b, :c)
    @test ctx3.values == (1, 2, 3)
    @test collect(ctx3) == [1, 2, 3]

    # test constructor with vector
    x4 = [1.0, 2.0, 3.0, 4.0]

    ctx4 = Context(x4)
    @test ctx4.x == (x1=1.0, x2=2.0, x3=3.0, x4=4.0)
    @test ctx4.d == 4
    @test ctx4.keys == (:x1, :x2, :x3, :x4)
    @test ctx4.values == (1.0, 2.0, 3.0, 4.0)
    @test collect(ctx4) == [1.0, 2.0, 3.0, 4.0]

    # test constructor with tuple
    x5 = (1.0, "2")

    ctx5 = Context(x5)
    @test ctx5.x == (x1=1.0, x2="2")
    @test ctx5.d == 2
    @test ctx5.keys == (:x1, :x2)
    @test ctx5.values == (1.0, "2")
    @test collect(ctx5) == [1.0, "2"]

end

"""Happy test of RandomContextFunction"""
function test_random_context_function_happy()
    seed = 123
    t = 1
    # test constructor with namedtuple
    D1 = (a=Normal(), b=Uniform(), c=Bernoulli())
    ctx_func1 = RandomContextFunction(D1, seed=seed)
    ctx1 = ctx_func1(t)
    @test ctx1 isa Context
    @test isapprox(collect(ctx1), [1.1902678809862768, 0.940515000715187, false], atol=4)

    # test constructor with dict
    D2 = Dict(:a=>Normal(), :b=>Uniform(), :c=>Bernoulli())
    ctx_func2 = RandomContextFunction(D2, seed=seed)
    ctx2 = ctx_func2(t)
    @test ctx2 isa Context
    @test isapprox(collect(ctx2), [1.1902678809862768, 0.940515000715187, false], atol=4)

    # test constructor with vector
    D3 = [Normal(), Uniform(), Bernoulli()]
    ctx_func3 = RandomContextFunction(D3, seed=seed)
    ctx3 = ctx_func3(t)
    @test ctx3 isa Context
    @test isapprox(collect(ctx3), [1.1902678809862768, 0.940515000715187, false], atol=4)

    # test constructor with tuple
    D4 = (Normal(), Uniform(), Bernoulli())
    ctx_func4 = RandomContextFunction(D4, seed=seed)
    ctx4 = ctx_func4(t)
    @test ctx4 isa Context
    @test isapprox(collect(ctx4), [1.1902678809862768, 0.940515000715187, false], atol=4)

    # test constructor with int and dist
    ctx_func5 = RandomContextFunction(3, Uniform(), seed=seed)
    ctx5 = ctx_func5(t)
    @test ctx5 isa Context
    @test isapprox(collect(ctx5), [0.7684476751965699, 0.940515000715187, 0.6739586945680673], atol=4)    
end

@testset "context" begin
    @testset "unit" begin
        @testset "happy" begin
            test_context_happy()
            test_random_context_function_happy()
        end
        @testset "bad" begin
            # ...
        end
    end
end
