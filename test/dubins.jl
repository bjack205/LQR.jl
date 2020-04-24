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

if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, RobotZoo.DubinsCar())
end

prob, = Problems.DubinsCar(:turn90, N=11)
TrajOptCore.add_dynamics_constraints!(prob)
n,m,N = size(prob)
NN = LQR.num_vars(prob)
P = sum(num_constraints(prob))
Z0 = LQR.Primals(prob).Z

# Build solver
solver = LQR.SparseSolver(prob)
LQR.step!(solver)
@time LQR.solve!(solver)

merit = TrajOptCore.L1Merit()
ϕ = TrajOptCore.gen_ϕ(merit, solver)
ϕ′ = TrajOptCore.gen_ϕ′(merit, solver)
ls = TrajOptCore.SimpleBacktracking()
ls = TrajOptCore.SecondOrderCorrector()
crit = TrajOptCore.WolfeConditions()

# Initialize
solver.Z.Z .= Z0
iter = 0
merit.μ = 1


# Take a step
LQR.update!(solver)
TrajOptCore.update_penalty!(merit, solver)
merit.μ
@show max_violation(solver)
LQR._solve!(solver)
ϕ(0)
ϕ(1)
max_violation(solver, recalculate=false)
LQR.project!(solver)
ϕ()
max_violation(solver, recalculate=false)
α = TrajOptCore.line_search(ls, crit, merit, solver)
copyto!(solver.Z.Z, solver.Z̄.Z)
iter += 1

visualize!(vis, solver)
@btime TrajOptCore.cost_dhess(solver)
