@testset "Constructors" begin

    seed!(121)

    nobs = 1_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinearRegressor(nrounds = 10, loss = :mse, L1 = 0, L2 = 1)
    m = EvoLinear.Linear.EvoLinearModel(:mse; coef = rand(3), bias = rand())
    m = EvoLinear.Linear.EvoLinearModel(:mse; coef = rand(Float32, 3), bias = rand(Float32))
    m = EvoLinear.Linear.EvoLinearModel(EvoLinear.Linear.loss_types[:mse]; coef = rand(3), bias = rand())

end

@testset "MSE" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoSplineRegressor(nrounds = 10, loss = :mse)
    m0 = EvoLinear.fit(config; x_train, y_train, metric = :mse)
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

@testset "Logistic" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = EvoLinear.sigmoid(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoSplineRegressor(nrounds = 10, loss = :logistic)
    m = EvoLinear.fit(config; x_train, y_train, metric = :logloss)
    p = m(x_train)

    metric = EvoLinear.Metrics.logloss(p, y_train)
    @test metric < 0.2

end
