@testset "Constructors" begin
    seed!(121)
    nobs = 1_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m = EvoLinear.EvoLinearModel(:mse; coef=rand(3), bias=rand(), info=Dict{Symbol,Any}())
    m = EvoLinear.EvoLinearModel(:mse; coef=rand(Float32, 3), bias=rand(Float32), info=Dict{Symbol,Any}())
    m = EvoLinear.EvoLinearModel(EvoLinear.loss_types[:mse]; coef=rand(3), bias=rand(), info=Dict{Symbol,Any}())
end

@testset "MSE" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats)
    dtrain.y = x * coef .+ rand(T, nobs) * T(0.1)
    target_name = :y

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m0 = EvoLinear.fit(config, dtrain; target_name, feature_names)
    all(m0.coef .≈ coef)

    m0B = EvoLinear.fit(config, dtrain; target_name)
    all(m0.coef .== m0B.coef)

    m1, cache = EvoLinear.init(config, dtrain; target_name, feature_names)
    for i = 1:config.nrounds
        EvoLinear.fit!(m1, cache, config)
    end

    coef_diff = m0.coef .- m1.coef
    @info "max coef diff" maximum(coef_diff)
    @info "min coef diff" minimum(coef_diff)
    @test all(m0.coef .≈ m1.coef)
    p = m0(dtrain)

    metric_mse = EvoLinear.Metrics.mse(p, dtrain[!, target_name])
    metric_mae = EvoLinear.Metrics.mae(p, dtrain[!, target_name])
    @test metric_mse < 0.12
    @test metric_mae < 0.28
end

@testset "L1/L2 regularization" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats)
    dtrain.y = x * coef .+ rand(T, nobs) * T(0.1)
    target_name = :y

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=0)
    m0 = EvoLinear.fit(config, dtrain; target_name, feature_names)
    @test sum(m0.coef .== 0) >= 0

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=1, L2=0)
    m1 = EvoLinear.fit(config, dtrain; target_name, feature_names)
    @test sum(m1.coef .== 0) >= 10

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=10)
    m2 = EvoLinear.fit(config, dtrain; target_name, feature_names)
    @test sum(abs.(m2.coef)) < sum(abs.(m0.coef))
end


@testset "Logistic" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats)
    dtrain.y = EvoLinear.sigmoid.(x * coef .+ rand(T, nobs) * T(0.1))
    dtrain.w = ones(T, nobs)
    target_name = :y
    weight_name = :w

    config = EvoLinearRegressor(nrounds=10, loss=:logloss, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config, dtrain; target_name, feature_names)

    p = m(dtrain)
    p1 = EvoLinear.predict(m, dtrain)
    @test all(p .== p1)

    metric = EvoLinear.Metrics.logloss(p, dtrain[!, target_name])
    metric_w = EvoLinear.Metrics.logloss(p, dtrain[!, target_name], dtrain[!, weight_name])
    @test metric ≈ metric_w
    @test metric < 0.2
end


@testset "Poisson" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats) ./ 10
    dtrain.y = exp.(x * coef .+ rand(T, nobs) * T(0.1))
    dtrain.w = ones(T, nobs)
    target_name = :y
    weight_name = :w

    config = EvoLinearRegressor(nrounds=10, loss=:poisson, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config, dtrain; target_name, feature_names, weight_name)
    p = m(dtrain)

    metric = EvoLinear.Metrics.poisson_deviance(p, dtrain[!, target_name])
    metric_w = EvoLinear.Metrics.poisson_deviance(p, dtrain[!, target_name], dtrain[!, weight_name])

    @test metric ≈ metric_w
    @test metric < 0.005
end

@testset "Gamma" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats) ./ 10
    dtrain.y = exp.(x * coef .+ rand(T, nobs) * T(0.1))
    dtrain.w = ones(T, nobs)
    target_name = :y
    weight_name = :w

    config = EvoLinearRegressor(nrounds=10, loss=:gamma, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config, dtrain; target_name, feature_names, weight_name)
    p = m(dtrain)

    metric = EvoLinear.Metrics.gamma_deviance(p, dtrain[!, target_name])
    metric_w = EvoLinear.Metrics.gamma_deviance(p, dtrain[!, target_name], dtrain[!, weight_name])

    @test metric ≈ metric_w
    @test metric < 0.005
end

@testset "Tweedie" begin
    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    dtrain = DataFrame(x, :auto)
    feature_names = names(dtrain)
    coef = randn(T, nfeats) ./ 10
    dtrain.y = exp.(x * coef .+ rand(T, nobs) * T(0.1))
    dtrain.w = ones(T, nobs)
    target_name = :y
    weight_name = :w

    config = EvoLinearRegressor(nrounds=10, loss=:tweedie, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config, dtrain; target_name, feature_names, weight_name)
    p = m(dtrain)

    metric = EvoLinear.Metrics.tweedie_deviance(p, dtrain[!, target_name])
    metric_w = EvoLinear.Metrics.tweedie_deviance(p, dtrain[!, target_name], dtrain[!, weight_name])

    @test metric ≈ metric_w
    @test metric < 0.01
end
