using EvoLinear
using Test
using Random: seed!

@testset "EvoLinear.jl" begin

    @testset "Core API" begin
        include("core.jl")
    end
    # Write your tests here.
end
