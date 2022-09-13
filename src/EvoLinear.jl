module EvoLinear

using Base.Threads: @threads
using StatsBase
using Statistics: mean, std

include("structs.jl")
include("loss.jl")
include("fit.jl")

end
