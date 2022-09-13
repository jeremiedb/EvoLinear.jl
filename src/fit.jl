function init(config::EvoLinearRegressor, x)

    T = Float32
    m = EvoLearner(zeros(T, size(x, 2)), zero(T))
    p = predict(m, x)
    p_coef = predict_coef(m, x)

    ∇¹ = init_∇¹(x)
    ∇² = init_∇²(x)

    cache = (
        m=m,
        p=p,
        p_coef=p_coef,
        ∇¹=∇¹,
        ∇²=∇²
    )
    return m, cache

end

function fit(config::EvoLinearRegressor; x, y, w=nothing)

    m, cache = init(config::EvoLinearRegressor, x)
    fit!(m, cache, config; x, y, w)

    p = predict(m, x)
    metric = mse(p, y)
    @info metric

    return m
end

function fit!(m::EvoLearner, cache, config::EvoLinearRegressor; x, y, w=nothing)

    m = cache.m
    ∇¹, ∇² = cache.∇¹ .* 0, cache.∇² .* 0

    p = predict(m, x)
    # update_∇!(∇¹, ∇², x, y, p, w)
    update_∇¹!(∇¹, x, y, p, w)
    update_∇²!(∇², x, y, p, w)

    update_coef!(m, ∇¹, ∇²)
    update_bias!(m, x, y, w)

    return nothing
end

function update_coef!(m, ∇¹, ∇²)
    m.coef .+= -∇¹ ./ ∇²
    return nothing
end
function update_bias!(m, x, y, w)
    p̄ = mean(predict_coef(m, x))
    ȳ = mean(y)
    m.bias = ȳ - p̄
    return nothing
end

function predict(m, x)
    p = x * m.coef .+ m.bias
    return p
end
function predict_coef(m, x)
    p = x * m.coef
    return p
end
