@testset "MSE" begin

    seed!(121)

    nobs = 100_000
    nfeats = 100
    T = Float32

    x = randn(T, nobs, nfeats)
    coef = randn(T, nfeats)

    y = x * coef .+ rand(T, nobs) * T(0.1)

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-3)
    m0 = EvoLinear.fit(config; x, y, metric=:mse)

    m1, cache = EvoLinear.init(config; x, y)
    EvoLinear.fit!(m1, cache, config)

    @test all(m0.coef .== m1.coef)

    p = EvoLinear.predict_proj(m0, x)
    metric_mse = EvoLinear.mse(p, y)
    metric_mae = EvoLinear.mae(p, y)
    @test metric_mse < 0.2
    @test metric_mae < 0.3

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

    config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-1, L2=1e-3)
    m = EvoLinear.fit(config; x, y, metric=:logloss)

    p = EvoLinear.predict_proj(m, x)
    p = m(x)
    
    metric_logloss = EvoLinear.logloss(p, y)
    metric_logloss_w = EvoLinear.logloss(p, y, w)
    
    metric_mae = EvoLinear.mae(p, y)
    @test metric_mse < 0.2
    @test metric_mae < 0.3

end