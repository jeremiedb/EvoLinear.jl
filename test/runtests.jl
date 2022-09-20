using EvoLinear
using Test
using Random: seed!

@testset "EvoLinear.jl" begin
    @testset "Core API" begin
        include("core.jl")
    end
    @testset "MLJ API" begin
        include("MLJ.jl")
    end
end
