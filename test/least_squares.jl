using LQR
using StaticArrays
using LinearAlgebra
using RobotZoo
using BenchmarkTools
using SparseArrays
import LQR: KnotPoint

model = RobotZoo.DubinsCar()
n,m = size(model)
N,dt = 101, 0.01
x0,u0 = zeros(model)
x0 = @SVector [1,1,0]
z0 = KnotPoint(x0,u0,dt)

Q = Diagonal(@SVector fill(2.,n))
R = Diagonal(@SVector fill(1.,m))
prob = LQRProblem(model, Q, R, 10Q, z0, N)

solver = LeastSquaresSolver(prob,SparseMatrixCSC)
prob.A .= 0.99 * I(3)
prob.B .= 1
LQR.build_least_squares!(solver, prob)
solver.T[end-2:end,1:m] ≈ prob.A^(N-2) * prob.B
solver.L[end-2:end,:] ≈ prob.A^(N-1)
solver.Hx[end-2:end,end-2:end] ≈ sqrt(10Q)

T,L = zero(solver.T), zero(solver.L)
LQR.build_toeplitz(T,L,prob.A,prob.B)
T ≈ solver.T
L ≈ solver.L

A = solver.Hx*solver.T
b = solver.Hx*solver.L*prob.x0
LQR.buildAb!(solver,prob)
# @btime LQR.buildAb!($solver,$prob)
# @btime LQR.build_least_squares!($solver,$prob)
solver.Ā ≈ A
solver.b̄ ≈ b


# @btime LQR.build_least_squares!($solver, $prob)
prob = LQRProblem(model, Q, R, 10Q, z0, N)
solver = LeastSquaresSolver(prob)
solver.Hx
U = LQR.solve!(solver, prob)
A = solver.Hx*solver.T
b = solver.Hx*solver.L*prob.x0
R_ = solver.Hu
norm(A'*(A*U+b) + R_*U, Inf) < 1e-12

solver.opts[:matbuild] = :lsq
solver.opts[:solve_type] = :naive
@btime LQR.solve!($solver,$prob)
solver.opts[:matbuild] = :Ab
solver.opts[:solve_type] = :cholesky
@btime LQR.solve!($solver,$prob)

H = solver.H
cholesky(H)\solver.y
