module EvoLinear

using Base.Threads: @threads
using Random
using StatsBase
using Statistics: mean, std
using LoopVectorization

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoLinearRegressor

include("structs.jl")
include("loss.jl")
include("metric.jl")
include("predict.jl")
include("fit.jl")
include("MLJ.jl")

include("splines/Splines.jl")
using .Splines

end
