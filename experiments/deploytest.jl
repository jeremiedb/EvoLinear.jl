using EvoLinear
using Serialization
using JLD2 

nobs = 1_000_000
nfeats = 10
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)
y = x * coef .+ rand(T, nobs) * T(0.1)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=0e-1, L2=1)
m = EvoLinear.fit(config; x, y, metric=:mae);

models_path = joinpath(@__DIR__, "../data/models")
JLD2.save(joinpath(models_path, "model-evolinear-test.jld2"), Dict("model" => m))
serialize(joinpath(models_path, "model-evolinear-test.dat"), m)