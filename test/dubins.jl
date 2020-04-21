using MeshCat
using TrajOptPlots
using TrajectoryOptimization
using RobotZoo
using ForwardDiff
using TrajOptCore
using LQR
using StaticArrays
using LinearAlgebra
using RobotDynamics
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
merit = TrajOptCore.L1Merit(1.0)
ϕ = merit(solver)
ϕ′ = TrajOptCore.derivative(merit, solver)
ls = TrajOptCore.SimpleBacktracking()
crit = TrajOptCore.WolfeConditions()

# Take a step
x = Z0
solver.Z.Z .= Z0
LQR.update!(solver)
solver.G ≈ ∇²f(x)
solver.g ≈ ∇f(x)
solver.conSet.D ≈ ∇c(x)
solver.conSet.d ≈ c(x)

LQR.update!(solver)
@show max_violation(solver)
ϕ(0)
dy = K(x)\r2(x)
LQR._solve!(solver)
solver.δZ.Z ≈ dy[1:NN]
solver.λ ≈ dy[NN+1:end]

α = TrajOptCore.line_search(ls, crit, ϕ, ϕ′)
copyto!(solver.Z.Z, solver.Z̄.Z)

visualize!(vis, solver)
