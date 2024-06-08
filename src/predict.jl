
function (m::EvoLinearModel{L})(x::AbstractMatrix; proj::Bool=true) where {L}
    p = x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return p
end
function (m::EvoLinearModel{L})(p::AbstractVector, x::AbstractMatrix; proj::Bool=true) where {L}
    p .= x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return nothing
end
function (m::EvoLinearModel{L})(data; proj::Bool=true) where {L}

    Tables.istable(data) || error("data must be Table compatible")

    T = Float32
    feature_names = m.info[:feature_names]
    nobs = Tables.DataAPI.nrow(data)
    nfeats = length(feature_names)

    x = zeros(T, nobs, nfeats)
    @threads for j in axes(x, 2)
        @views x[:, j] .= Tables.getcolumn(data, feature_names[j])
    end

    p = x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return p
end

function proj!(::L, p) where {L<:Type{MSE}}
    return nothing
end
function proj!(::L, p) where {L<:Type{Logistic}}
    p .= sigmoid.(p)
    return nothing
end
function proj!(::L, p) where {L<:Union{Type{Poisson},Type{Gamma},Type{Tweedie}}}
    p .= exp.(p)
    return nothing
end

# """
#     predict_linear(m, x)

# Returns the predictions on the linear basis from model `m` using the features matrix `x`.

# # Arguments

# - `m::EvoLinearModel`: model generating the predictions.
# - `x`: features matrix `[nobs, num_features]` for which predictions are generated.
# """
# function predict_linear(m::EvoLinearModel, x)
#     p = x * m.coef .+ m.bias
#     return p
# end

# function predict_linear!(p, m::EvoLinearModel, x)
#     p .= x * m.coef .+ m.bias
#     return nothing
# end

# """
#     predict_proj(m, x)

# Returns the predictions on the projected basis from model `m` using the features matrix `x`.

# - `MSE`: `pred_proj = pred_linear`
# - `Logistic`: `pred_proj = sigmoid(pred_linear)`
# - `Poisson`: `pred_proj = exp(pred_linear)`
# - `Gamma`: `pred_proj = exp(pred_linear)`
# - `Tweedie`: `pred_proj = exp(pred_linear)`

# # Arguments

# - `m::EvoLinearModel`: model generating the predictions.
# - `x`: features matrix `[nobs, num_features]` for which predictions are generated.
# """
# function predict_proj(m::EvoLinearModel{MSE}, x)
#     p = predict_linear(m, x)
#     return p
# end
# function predict_proj(m::EvoLinearModel{Logistic}, x)
#     p = sigmoid(predict_linear(m, x))
#     return p
# end
# function predict_proj(m::EvoLinearModel{L}, x) where {L<:Union{Poisson,Gamma,Tweedie}}
#     p = exp.(predict_linear(m, x))
#     return p
# end