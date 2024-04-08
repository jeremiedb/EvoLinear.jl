module CallBacks

using ..EvoLinear.Metrics

export init_logger, update_logger!

struct CallBack{F,M,V,Y}
    feval::F
    x::M
    p::V
    y::Y
    w::V
end

function init_logger(; metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :nrounds => 0,
        :iter => Int[],
        :metrics => Float32[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger, m, cb, iter)

    m(cb.p, cb.x; proj=true)
    metric = cb.feval(cb.p, cb.y, cb.w)

    logger[:nrounds] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    if iter == 0
        logger[:best_metric] = metric
    else
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