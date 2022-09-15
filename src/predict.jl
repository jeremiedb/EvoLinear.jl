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
function predict_proj(m::EvoLinearModel{L}, x) where {L<:Union{Poisson,Gamma,Tweedie}}
    p = exp.(predict_linear(m, x))
    return p
end