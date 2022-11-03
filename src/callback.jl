
module CallBacks

export CallBackLinear, init_logger, update_logger!

using ..EvoLinear.Metrics

struct CallBackLinear{F,M,V,Y}
    feval::F
    x::M
    p::V
    y::Y
    w::V
end
function (cb::CallBackLinear)(logger, iter, m)
    m(cb.p, cb.x; proj = true)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return nothing
end

function CallBackLinear(; metric, x_eval, y_eval, w_eval = nothing, T = Float32)
    feval = metric_dict[metric]
    x = convert(Matrix{T}, x_eval)
    p = zeros(T, length(y_eval))
    y = convert(Vector{T}, y_eval)
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)
    return CallBackLinear(feval, x, p, y, w)
end

function init_logger(; T, metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :iter => Int[],
        :metrics => T[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger, iter, metric)
    logger[:nrounds] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    if iter > 0
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end

end