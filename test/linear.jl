@testset "Constructors" begin

    seed!(121)

    nobs = 1_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m = EvoLinear.Linear.EvoLinearModel(:mse; coef=rand(3), bias=rand())
    m = EvoLinear.Linear.EvoLinearModel(:mse; coef=rand(Float32, 3), bias=rand(Float32))
    m = EvoLinear.Linear.EvoLinearModel(EvoLinear.Linear.loss_types[:mse]; coef=rand(3), bias=rand())

end

@testset "MSE" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m0 = EvoLinear.fit(config; x_train, y_train, metric=:mse)
    m1, cache = EvoLinear.Linear.init(config, x_train, y_train)
    for i = 1:config.nrounds
        EvoLinear.fit!(m1, cache, config)
    end

    coef_diff = m0.coef .- m1.coef
    @info "max coef diff" maximum(coef_diff)
    @info "min coef diff" minimum(coef_diff)
    @test all(m0.coef .≈ m1.coef)
    p = m0(x_train)

    metric_mse = EvoLinear.Metrics.mse(p, y_train)
    metric_mae = EvoLinear.Metrics.mae(p, y_train)
    @test metric_mse < 0.12
    @test metric_mae < 0.28

end

@testset "L1/L2 regularization" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=0)
    m0 = EvoLinear.fit(config; x_train, y_train, metric=:mse)

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=0)
    m1 = EvoLinear.fit(config; x_train, y_train, metric=:mse)
    @test sum(m1.coef .== 0) >= 5

    config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m2 = EvoLinear.fit(config; x_train, y_train, metric=:mse)
    @test sum(abs.(m2.coef)) < sum(abs.(m0.coef))

end


@testset "Logistic" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = EvoLinear.sigmoid(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x_train, y_train, metric=:logloss)

    p = m(x_train)
    p1 = EvoLinear.Linear.predict_proj(m, x_train)

    @test all(p .== p1)

    metric = EvoLinear.Metrics.logloss(p, y_train)
    metric_w = EvoLinear.Metrics.logloss(p, y_train, w)
    @test metric ≈ metric_w
    @test metric < 0.2

end


@testset "Poisson" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 10
    y_train = exp.(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinearRegressor(nrounds=10, loss=:poisson_deviance, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x_train, y_train, metric=:poisson_deviance)
    p = m(x_train)

    metric = EvoLinear.Metrics.poisson_deviance(p, y_train)
    metric_w = EvoLinear.Metrics.poisson_deviance(p, y_train, w)
    @test metric ≈ metric_w
    @test metric < 0.005

end

@testset "Gamma" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 10
    bias = T(1.0)
    y_train = exp.(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinearRegressor(nrounds=10, loss=:gamma_deviance, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x_train, y_train, metric=:gamma_deviance)
    p = m(x_train)

    metric = EvoLinear.Metrics.gamma_deviance(p, y_train)
    metric_w = EvoLinear.Metrics.gamma_deviance(p, y_train, w)
    @test metric ≈ metric_w
    @test metric < 0.005

end

@testset "Tweedie" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 10
    bias = T(1.0)
    y_train = exp.(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinearRegressor(nrounds=10, loss=:tweedie_deviance, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x_train, y_train, metric=:tweedie_deviance)
    p = m(x_train)

    metric = EvoLinear.Metrics.tweedie_deviance(p, y_train)
    metric_w = EvoLinear.Metrics.tweedie_deviance(p, y_train, w)
    @test metric ≈ metric_w
    @test metric < 0.01

end