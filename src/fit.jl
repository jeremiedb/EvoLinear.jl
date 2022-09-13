function init(config::EvoLinearRegressor, x)

    T = Float32
    loss = loss_types[config.loss]
    m = EvoLinearModel{loss}(zeros(T, size(x, 2)), zero(T))
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

function fit!(m::EvoLinearModel{L}, cache, config::EvoLinearRegressor; x, y, w=nothing) where {L}

    m = cache.m
    ∇¹, ∇² = cache.∇¹ .* 0, cache.∇² .* 0

    ####################################################
    # update all coefs then bias
    ####################################################
    # @info "predict time"
    p = predict(m, x)
    # update_∇!(L, ∇¹, ∇², x, y, p, w)
    update_∇¹!(L, ∇¹, x, y, p, w)
    update_∇²!(L, ∇², x, y, p, w)
    update_coef!(m, ∇¹, ∇²)
    # @info "bias update time"
    update_bias!(m, x, y, w)

    ####################################################
    # update bias following each feature update
    ####################################################
    # for feat in axes(x, 2)
    #     p = predict(m, x)
    #     update_∇¹!(L, ∇¹, x, y, p, w, feat)
    #     update_∇²!(L, ∇², x, y, p, w, feat)
    #     update_coef!(m, ∇¹, ∇², feat)
    #     update_bias!(m, x, y, w)
    # end

    return nothing
end

function update_coef!(m, ∇¹, ∇²)
    m.coef .+= -∇¹ ./ ∇²
    return nothing
end
function update_coef!(m, ∇¹, ∇², feat)
    m.coef[feat] += -∇¹[feat] ./ ∇²[feat]
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

function predict(m::EvoLinearModel{Logistic}, x)
    p = x * m.coef .+ m.bias
    return p
end
function predict_coef(m::EvoLinearModel{Logistic}, x)
    p = x * m.coef
    return p
end
