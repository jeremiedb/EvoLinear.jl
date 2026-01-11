using Test
using EvoLinear
using DataFrames
using Statistics: mean, std
using Random: seed!

@testset "EvoLinear.jl" begin
    @testset "Core API" begin
        include("linear.jl")
    end
    @testset "MLJ API" begin
        include("MLJ.jl")
    end
end
