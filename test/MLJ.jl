using EvoLinear: logit, sigmoid
using StatsBase: sample
using MLJBase

##################################################
### Regression - small data
##################################################
features = rand(10_000) .* 5 .- 2
X = reshape(features, (size(features)[1], 1))
Y = sin.(features) .* 0.5 .+ 0.5
Y = logit(Y) + randn(size(Y))
Y = sigmoid(Y)
y = Y
X = MLJBase.table(X)

# linear regression
model = EvoLinearRegressor(loss=:mse, nrounds=10)
# logistic regression
model = EvoLinearRegressor(loss=:logistic, nrounds=4)

mach = machine(model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)

mach.model.nrounds += 2
fit!(mach, rows=train, verbosity=1)
mach.cache[:info][:nrounds]

# predict on train data
pred_train = predict(mach, selectrows(X, train))
mean(abs.(pred_train - selectrows(Y, train)))

# predict on test data
pred_test = predict(mach, selectrows(X, test))
mean(abs.(pred_test - selectrows(Y, test)))

@test MLJBase.iteration_parameter(EvoLinearRegressor) == :nrounds


##################################################
### Regression - matrix data
##################################################
X = MLJBase.matrix(X)
model = EvoLinearRegressor(loss=:logistic, nrounds=4)

mach = machine(model, X, y)
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(mach, rows=train, verbosity=1)

mach.model.nrounds += 2
fit!(mach, rows=train, verbosity=1)

pred_train = predict(mach, selectrows(X, train))
mean(abs.(pred_train - selectrows(Y, train)))


####################################################################################
# tests that `update` handles data correctly in the case of a cold restart:
####################################################################################
X = MLJBase.table(rand(5, 2))
y = rand(5)
model = EvoLinearRegressor(loss=:mse)
data = MLJBase.reformat(model, X, y);
f, c, r = MLJBase.fit(model, 2, data...);
c[:info]
model.L2 = 0.1
model.nrounds += 2
MLJBase.update(model, 2, f, c, data...)
c[:info][:nrounds]

X = rand(5, 2)
y = rand(5)
model = EvoLinearRegressor(loss=:mse)
data = MLJBase.reformat(model, X, y);
f, c, r = MLJBase.fit(model, 2, data...);
model.L2 = 0.1
model.nrounds += 2
MLJBase.update(model, 2, f, c, data...)
MLJBase.update(model, 2, f, c, data...)