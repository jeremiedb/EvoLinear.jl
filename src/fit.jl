function init(config::EvoLinearRegressor, x)

    T = Float32
    loss = loss_types[config.loss]
    m = EvoLinearModel{loss}(zeros(T, size(x, 2)), zero(T))
    p_linear = predict_linear(m, x)
    p_proj = predict_proj(m, x)

    ∇¹ = init_∇¹(x)
    ∇² = init_∇²(x)
    ∇b = zeros(T, 2)

    cache = (
        p_linear=p_linear, p_proj=p_proj,
        ∇¹=∇¹, ∇²=∇², ∇b=∇b
    )
    return m, cache

end

function fit(config::EvoLinearRegressor; x, y, w=nothing)
    m, cache = init(config::EvoLinearRegressor, x)

    for i in 1:config.nrounds
        fit!(m, cache, config; x, y, w)
    end
    
    p = predict_proj(m, x)
    metric = mse(p, y)
    @info metric

    return m
end

function fit!(m::EvoLinearModel{L}, cache, config::EvoLinearRegressor;
    x, y, w=nothing) where {L}

    ∇¹, ∇², ∇b = cache.∇¹ .* 0, cache.∇² .* 0, cache.∇b .* 0

    if config.updater == :all
        ####################################################
        # update all coefs then bias
        ####################################################
        p = predict_proj(m, x)
        update_∇!(L, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇²)
        p = predict_proj(m, x)
        update_∇_bias!(L, ∇b, x, y, p, w)
        update_bias!(m, ∇b)
    elseif config.updater == :single
        @warn "single update needs to be fixed - preds update needs linear projection basis"
        ####################################################
        # update bias following each feature update
        ####################################################
        p = predict_proj(m, x)
        for feat in axes(x, 2)
            update_∇!(L, ∇¹, ∇², x, y, p, w, feat)
            Δ_coef = coef_update(m, ∇¹, ∇², feat)
            p .+= Δ_coef .* x[:, feat]
            m.coef[feat] += Δ_coef
        end
        update_∇_bias!(L, ∇b, x, y, p, w)
        update_bias!(m, ∇b)
    else
        @warn "invalid updater"
    end
    return nothing
end

function update_coef!(m, ∇¹, ∇²)
    m.coef .+= -∇¹ ./ ∇²
    return nothing
end
function update_coef!(m, ∇¹, ∇², feat)
    m.coef[feat] += -∇¹[feat] / ∇²[feat]
    return nothing
end
function coef_update(m, ∇¹, ∇², feat)
    -∇¹[feat] / ∇²[feat]
end
function update_bias!(m, ∇b)
    m.bias += -∇b[1] / ∇b[2]
    return nothing
end


function predict_linear(m, x)
    p = x * m.coef .+ m.bias
    return p
end
function predict_proj(m::EvoLinearModel{MSE}, x)
    p = predict_linear(m, x)
    return p
end
function predict_proj(m::EvoLinearModel{Logistic}, x)
    p = sigmoid(predict_linear(m, x))
    return p
end
