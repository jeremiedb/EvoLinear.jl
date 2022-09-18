@testset "MSE" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)

    y = x * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m0 = EvoLinear.fit(config; x, y, metric=:mse)
    m1, cache = EvoLinear.init(config; x, y)
    for i in 1:config.nrounds
    EvoLinear.fit!(m1, cache, config)
    end

    coef_diff = m0.coef .- m1.coef
    @info "max coef diff" maximum(coef_diff)
    @info "min coef diff" minimum(coef_diff)
    @test all(m0.coef .≈ m1.coef)

    p = EvoLinear.predict_proj(m0, x)
    metric_mse = EvoLinear.mse(p, y)
    metric_mae = EvoLinear.mae(p, y)
    @test metric_mse < 0.12
    @test metric_mae < 0.28

end

@testset "L1/L2 regularization" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)

    y = x * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=0)
    m0 = EvoLinear.fit(config; x, y, metric=:mse)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=0)
    m1 = EvoLinear.fit(config; x, y, metric=:mse)
    @test sum(m1.coef .== 0) >= 5

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=0, L2=1)
    m2 = EvoLinear.fit(config; x, y, metric=:mse)
    @test sum(abs.(m2.coef)) < sum(abs.(m0.coef))

end


@testset "Logistic" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)
    y = EvoLinear.sigmoid(x * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x, y, metric=:logloss)

    p = m(x)
    p1 = EvoLinear.predict_proj(m, x)

    @test all(p .== p1)

    metric = EvoLinear.logloss(p, y)
    metric_w = EvoLinear.logloss(p, y, w)
    @test metric ≈ metric_w
    @test metric < 0.2

end


@testset "Poisson" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 10
    y = exp.(x * coef .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:poisson, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x, y, metric=:poisson_deviance)

    p = m(x)
    p1 = EvoLinear.predict_proj(m, x)

    @test all(p .== p1)

    metric = EvoLinear.poisson_deviance(p, y)
    metric_w = EvoLinear.poisson_deviance(p, y, w)
    @test metric ≈ metric_w
    @test metric < 0.005

end

@testset "Gamma" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 5
    bias = 1
    y = exp.(x * coef .+ bias .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:gamma, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x, y, metric=:gamma_deviance)

    p = m(x)
    p1 = EvoLinear.predict_proj(m, x)

    @test all(p .== p1)

    metric = EvoLinear.gamma_deviance(p, y)
    metric_w = EvoLinear.gamma_deviance(p, y, w)
    @test metric ≈ metric_w
    @test metric < 0.005

end

@testset "Tweedie" begin

    seed!(121)
    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats) ./ 5
    bias = 1
    y = exp.(x * coef .+ bias .+ rand(T, nobs) * T(0.1))
    w = ones(T, nobs)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:tweedie, L1=1e-2, L2=1e-3)
    m = EvoLinear.fit(config; x, y, metric=:tweedie_deviance)

    p = m(x)
    p1 = EvoLinear.predict_proj(m, x)

    @test all(p .== p1)

    metric = EvoLinear.tweedie_deviance(p, y)
    metric_w = EvoLinear.tweedie_deviance(p, y, w)
    @test metric ≈ metric_w
    @test metric < 0.01

end