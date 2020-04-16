
# Set up problem
prob = DoubleIntegrator(3,101)
rollout!(prob)
Jinv = LQR.InvertedQuadratic.(prob.obj.cost)
NN = num_vars(solver)
P = sum(solver.conSet.p)

# Initialize data stuctures
LQR.update_cholesky!(Jinv, prob.obj)
conSet = LQR.BlockConstraintSet(prob.model, get_constraints(prob))
shur = LQR.build_shur_factors(conSet, :U)
chol = LQR.build_shur_factors(conSet, :U)

# Update constraints and compute shur compliment
Z = prob.Z
δZ = [LQR.MutableKnotPoint(n,m, (@MVector zeros(n+m)), z.dt, z.t) for z in prob.Z]
evaluate!(conSet, Z)
jacobian!(conSet, Z)
LQR.calculate_shur_factors!(shur, Jinv, conSet.blocks)


cholesky!(chol, shur)
LQR.forward_substitution!(chol)
LQR.backward_substitution!(chol)
LQR.calculate_primals!(δZ, Jinv, L, conSet.blocks)

Z̄ = [StaticKnotPoint(z) for z in prob.Z]
Z̄ .= prob.Z .+ δZ
max_violation(conSet)
# evaluate!(conSet, Z̄)
# max_violation(conSet)

Plots.pgfplotsx()
Plots.partialcircle(0,2pi, 100, 1.0)
