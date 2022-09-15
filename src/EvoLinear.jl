module EvoLinear

using Base.Threads: @threads
using StatsBase
using Statistics: mean, std

using LoopVectorization

export EvoLinearRegressor

include("structs.jl")
include("loss.jl")
include("metric.jl")
include("predict.jl")
include("fit.jl")

end
