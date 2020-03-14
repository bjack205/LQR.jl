using TrajOptPlots
using Plots
using MeshCat
using RobotDynamics
if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
end
include("problems.jl")
model = RobotZoo.DoubleIntegrator(2)
set_mesh!(vis, model)
prob = DoubleIntegrator()

sol = LQRSolution(prob)
solver = DPSolver(prob)
LQR.compute_gain!(sol.K[1], solver, prob)
LQR.compute_ctg!(sol.K[1], solver, prob)

# @btime LQR.compute_ctg!($sol.K[1], $solver, $prob)
solve!(sol, solver, prob)
visualize!(vis, model, sol.Z_)
@btime solve!($sol, $solver, $prob)
