# EvoLinear


L(β) = (p - y)²

p = p₀ + βx
L(β) = (p₀ + βx - y)²

L'(β) = 2x(p₀ + βx - y)
L'(β) = 2xp₀ + 2x²β - 2xy
L'(β) = 2x(p - y)
L''(β) = 2x²

L'(β) Is a vector of length = num features.
It needs to be updated after each iteration as it depends on the predictions (p).

L''(β) is constant throughout the iterations: only depends on input x. 
Is a vector of length = num features.

Algo:
Initialize L''. 
For each feature, calculate 2x.

Initialize L' base. 
For each feature, calculate x.

Calculate L'. 
Compute 2(p-y) and add it to L' base

Compute gain associated to each feature
Select highest gain
Update weight
Update bias
