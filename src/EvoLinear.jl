module EvoLinear

import Base.Threads: @threads
import Statistics: mean, std
import Tables
import Random

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoLinearRegressor, EvoLinearModel

include("utils.jl")

include("losses.jl")
using .Losses

include("metrics.jl")
using .Metrics

include("structs.jl")

include("callbacks.jl")
using .CallBacks

include("fit.jl")
include("predict.jl")

include("MLJ.jl")

end
