using LQR
using TrajOptCore
using TrajectoryOptimization
# using MeshCat
#
# if !isdefined(Main, :vis)
#     vis = Visualizer()
#     open(vis)
#     set_mesh!(vis, RobotZoo.DubinsCar())
# end

prob, = Problems.DubinsCar(:turn90, N=11)
TrajOptCore.add_dynamics_constraints!(prob)
n,m,N = size(prob)
NN = LQR.num_vars(prob)
P = sum(num_constraints(prob))
Z0 = LQR.Primals(prob).Z

al = AugmentedLagrangianSolver(prob)

# Build solver
solver = LQR.SparseSolver(prob)
LQR.update!(solver)
LQR.step!(solver)
@time LQR.solve!(solver)
