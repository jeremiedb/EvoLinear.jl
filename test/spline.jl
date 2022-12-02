@testset "Constructors" begin

    seed!(121)

    nobs = 1_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoSplineRegressor(nrounds = 10, loss = :mse)
    m = EvoLinear.EvoSplineModel(config; nfeats, mean = mean(y_train))

end

@testset "MSE" begin

    seed!(121)

    nobs = 10_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

    config = EvoSplineRegressor(nrounds = 100, loss = :mse, knots = Dict(1 => 4, 10 => 4))
    m0 = EvoLinear.fit(config; x_train, y_train, metric = :mse)
    m1, cache = EvoLinear.Linear.init(config, x_train, y_train)
    for i = 1:config.nrounds
        EvoLinear.fit!(m1, cache)
    end

    coef_diff = m0.linear.w .- m1.linear.w
    @info "max coef diff" maximum(coef_diff)
    @info "min coef diff" minimum(coef_diff)
    @test maximum(abs.(coef_diff)) < 0.01

    p = m0(x_train')

    metric_mse = EvoLinear.Metrics.mse(p, y_train)
    metric_mae = EvoLinear.Metrics.mae(p, y_train)
    @test metric_mse < 0.02
    @test metric_mae < 0.12

end

@testset "Logistic" begin

    seed!(121)
    nobs = 10_000
    nfeats = 10
    T = Float32

    x_train = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y_train = EvoLinear.sigmoid(x_train * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoSplineRegressor(nrounds = 100, loss = :logistic, knots = Dict(1 => 4, 10 => 4))
    m = EvoLinear.fit(config; x_train, y_train, metric = :logloss)
    p = m(x_train')

    metric = EvoLinear.Metrics.logloss(p, y_train)
    @test metric < 0.45

end
