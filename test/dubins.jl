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
using BenchmarkTools
const TO = TrajectoryOptimization

if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, RobotZoo.DubinsCar())
end

prob, = Problems.DubinsCar(:turn90, N=101)
TrajOptCore.add_dynamics_constraints!(prob)
n,m,N = size(prob)
NN = LQR.num_vars(prob)
P = sum(num_constraints(prob))
Z0 = LQR.Primals(prob).Z

# Build solver
solver = LQR.SparseSolver(prob)
@time LQR.solve!(solver)

LQR.step!(solver)
merit = TrajOptCore.L1Merit(1.0)
ϕ = merit(solver)
ϕ′ = TrajOptCore.derivative(merit, solver)
ls = TrajOptCore.SimpleBacktracking()
crit = TrajOptCore.WolfeConditions()

res = [zeros(n + 0*m*(k<N)) for k = 1:N]
con = solver.conSet.convals[1]
length(con.con)
λ = [zeros(length(con.con)) for k in con.inds]
TrajOptCore.norm_residual!(res, con, λ)
@btime TrajOptCore.norm_residual!($res, $con, $λ)

con.jac

# Initialize
solver.Z.Z .= Z0

# Take a step
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
