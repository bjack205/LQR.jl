using TrajOptCore
using RobotDynamics
using RobotZoo
using TrajectoryOptimization
using TrajOptPlots
using MeshCat
using ForwardDiff
using LQR
using StaticArrays
using LinearAlgebra
using BenchmarkTools
const TO = TrajectoryOptimization

prob, = Problems.DubinsCar(:turn90, N=11)
TrajOptCore.add_dynamics_constraints!(prob)

solver1 = CholeskySolver(prob)
solver2 = LQR.SparseSolver(prob)

LQR.step!(solver1)
LQR.step!(solver2)

LQR.update!(solver1)
LQR.update!(solver2)
max_violation(solver1) ≈ max_violation(solver2)
LQR.residual(solver1, recalculate=false) ≈ LQR.residual(solver2, recalculate=false)
cost(solver1) ≈ cost(solver2)
LQR._solve!(solver1)
LQR._solve!(solver2)
states(solver1.δZ) ≈ states(solver2.δZ.Z_)
controls(solver1.δZ) ≈ controls(solver2.δZ.Z_)
TrajOptCore.get_primals(solver1, 1.0)
TrajOptCore.get_primals(solver2, 1.0)
states(solver1.Z̄) ≈ states(solver2.Z̄.Z_)
controls(solver1.Z̄) ≈ controls(solver2.Z̄.Z_)

TrajOptCore.second_order_correction!(solver1)
TrajOptCore.second_order_correction!(solver2)
dx1 = LQR.get_residual(solver1)
dx2 = -D2'*((D2*D2')\d2)
dx1 ≈ dx2

states(solver1.Z̄) ≈ states(solver2.Z̄.Z_)
controls(solver1.Z̄) ≈ controls(solver2.Z̄.Z_)


dz1 = LQR.get_step(solver1)
dz2 = solver2.δZ.Z
dz1 ≈ dz2
λ1 = LQR.get_multipliers(solver1)
λ2 = solver2.λ
λ1 ≈ λ2
G1,g1 = LQR.get_cost_expansion(solver1)
G2,g2 = solver2.G, solver2.g
G1 ≈ G2
g1 ≈ g2
D1,d1 = LQR.get_linearized_constraints(solver1)
D2,d2 = solver2.conSet.D, solver2.conSet.d
D1 ≈ D2
d1 ≈ d2

r1 = D1'λ1+g1
r2 = D2'λ2+g2
norm(r1)
norm(r2)

solver1.conSet.blocks[1].res
solver1.chol_blocks[1].λ
res = D'λ+g
res[1:n+m]
G1 = solver2.G[1:n+m,1:n+m]
dz1 = solver2.δZ.Z_[1].z
G1\res[1:n+m]


solver1.conSet.blocks[2]

max_violation(solver1) ≈ max_violation(solver2)
LQR.norm_residual(solver1)
solver1.res
@btime LQR.norm_residual($solver1)
cst = solver1.J[1]
solver1
size(solver1)
@btime [$cst.q; $cst.r]
solver1.J[1].q
solver1.res
get_cos
