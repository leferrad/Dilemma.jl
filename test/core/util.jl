using Dilemma
using Random
using Test


"""Happy test of ones_in_zeros method"""
function test_ones_in_zeros_happy()
    n = 5
    idx = [2, 3]
    vec = ones_in_zeros(n, idx)
    @test all([vec[i] == 0 for i in 1:n if i ∉ idx])
end

# TODO:  function test_sample_mvnorm_happy()
# TODO: test bad not symmetric matrix in sample_mvnorm
# TODO: function sherman_morrison_inv(A_inv, X) 

"""Happy test of draw_index_from_probs method"""
function test_draw_index_from_probs_happy()
    probs = [0.3, 0.5, 0.1, 0.1]
    rng = MersenneTwister(123)
    idx = draw_index_from_probs(probs, rng)
    @test idx == 2
end


"""Bad path test of draw_index_from_probs method"""
function test_draw_index_from_probs_bad()
    # probs don't sum up 1
    probs = [3.0, 5.0, 0.1]
    rng = MersenneTwister(123)
    
    @test_throws ArgumentError draw_index_from_probs(probs, rng)

end


@testset "util" begin
    @testset "unit" begin
        @testset "happy" begin
            test_ones_in_zeros_happy()
            test_draw_index_from_probs_happy()
        end
        @testset "bad" begin
            test_draw_index_from_probs_bad()
        end
    end
end
