using EvoLinear
using Test
using Statistics: mean, std
using Random: seed!

@testset "EvoLinear.jl" begin
    @testset "Core API" begin
        include("linear.jl")
        include("spline.jl")
    end
    @testset "MLJ API" begin
        include("MLJ.jl")
    end
end
