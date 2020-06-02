using TrajOptCore
using RobotDynamics
using RobotZoo
using TrajectoryOptimization
using LQR
using StaticArrays
using LinearAlgebra
using BenchmarkTools
const TO = TrajectoryOptimization


prob, = Problems.DubinsCar(:parallel_park, N=31)
TrajOptCore.add_dynamics_constraints!(prob)
n,m,N = size(prob)

# Split constraint into inequalities and equalities
ineq = ConstraintList(n,m,N)
equl = ConstraintList(n,m,N)
for (ind,con) in zip(prob.constraints)
    set = TrajOptCore.sense(con) == Inequality() ? ineq : equl
    add_constraint!(set, con, ind)
end
ineq
equl

# Build AL Objective with inequality constraints
alobj = TO.ALObjective(prob.obj, ineq, prob.model)

# Build Problem with AL objective and only equality constraints
prob_al = Problem(prob, obj=prob.obj, constraints=equl)
rollout!(prob_al)

# Build solver
solver = LQR.SparseSolver(prob_al)
LQR.update!(solver)
LQR.step!(solver)
TrajOptCore.build_cost_expansion(alobj, prob.model)

solver.Gblocks
solver.J2[end].terminal

findmax_violation(alobj.constraints)
solver.J2[2]
conIn = alobj.constraints
C = conIn.convals[2].jac[2]
Imu = Diagonal(conIn.μ[2][2] .* conIn.active[2][2])
C'Imu*C

prob.obj[2].Q
