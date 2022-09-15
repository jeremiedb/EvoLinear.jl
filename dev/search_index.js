var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"EvoLinearRegressor","category":"page"},{"location":"api/","page":"API","title":"API","text":"EvoLinear.fit\nEvoLinear.predict_linear\nEvoLinear.predict_proj","category":"page"},{"location":"api/#EvoLinear.fit","page":"API","title":"EvoLinear.fit","text":"fit(config::EvoLinearRegressor;\n    x, y, w=nothing,\n    x_eval=nothing, y_eval=nothing, w_eval=nothing,\n    metric=:mse,\n    print_every_n=1,\n    tol=1e-5)\n\nProvided aconfig, EvoLinear.fit takes x and y as features and target inputs, plus optionally w as weights and train a Linear boosted model.\n\nArguments\n\nconfig::EvoLinearRegressor: \n\nKeyword arguments\n\nx::AbstractMatrix: Features matrix. Dimensions are [nobs, num_features].\ny::AbstractVector: Vector of observed targets.\nw=nothing: Vector of weights. Can be be either a Vector or nothing. If nothing, assumes a vector of 1s. \nmetric=:mse: Evaluation metric to be tracked through each iteration.\n\n\n\n\n\n","category":"function"},{"location":"api/#EvoLinear.predict_linear","page":"API","title":"EvoLinear.predict_linear","text":"predict_linear(m, x)\n\nReturns the predictions on the linear basis from model m using the features matrix x.\n\nArguments\n\nm::EvoLinearModel\nx\n\n\n\n\n\n","category":"function"},{"location":"api/#EvoLinear.predict_proj","page":"API","title":"EvoLinear.predict_proj","text":"predict_proj(m, x)\n\nReturns the predictions on the projected basis from model m using the features matrix x.\n\nMSE: predproj = predlinear\nLogistic: predproj = sigmoid(predlinear)\nPoisson: predproj = exp(predlinear)\nGamma: predproj = exp(predlinear)\nTweedie: predproj = exp(predlinear)\n\nArguments\n\nm::EvoLinearModel\nx\n\n\n\n\n\n","category":"function"},{"location":"#EvoLinear.jl","page":"Home","title":"EvoLinear.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ML library implementing linear boosting with L1/L2 regularization. For Tree based boosting, consider EvoTrees.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently supports:    ","category":"page"},{"location":"","page":"Home","title":"Home","text":"mse (squared-error)\nlogistic (logloss) regression\nPoisson\nGamma\nTweedie","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/jeremiedb/EvoLinear.jl","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"EvoLinear.fit takes x::Matrix and y::Vector as inputs, plus optionally w::Vector as weights.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using EvoLinear\nconfig = EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-2)\nm = EvoLinear.fit(config; x, y, metric=:mse)\np = EvoLinear.predict_proj(m, x)","category":"page"},{"location":"","page":"Home","title":"Home","text":"using EvoLinear\nconfig = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-1, L2=1e-2)\nm = EvoLinear.fit(config; x, y, metric=:mse)\np = EvoLinear.predict_proj(m, x)","category":"page"}]
}
