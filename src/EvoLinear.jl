module EvoLinear

using Base.Threads: @threads
using StatsBase
using Statistics: mean, std

export EvoLinearRegressor
include("structs.jl")
include("loss.jl")
include("fit.jl")

end
