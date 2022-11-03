module Linear

using ..EvoLinear
using ..EvoLinear.Metrics
using ..EvoLinear.CallBacks
using ..EvoLinear: sigmoid, logit

using Base.Threads: @threads
using Random
# using StatsBase
using Statistics: mean, std
using LoopVectorization

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoLinearTypes, EvoLinearRegressor, init, fit!, get_loss_type

include("structs.jl")
include("loss.jl")
include("predict.jl")
include("fit.jl")

end