include("problems.jl")
prob = DoubleIntegrator()
sol = LQRSolution(prob)
n,m = size(prob)
N = prob.N

solver = LeastSquaresSolver(prob)
prob.A .= 0.99 * I(n)
prob.B .= 1
LQR.build_least_squares!(solver, prob)
solver.T[end-n+1:end,1:m] ≈ prob.A^(N-2) * prob.B
solver.L[end-n+1:end,:] ≈ prob.A^(N-1)
solver.Hx[end-n+1:end,end-n+1:end] ≈ sqrt(prob.Qf)

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
prob = DoubleIntegrator()
solver = LeastSquaresSolver(prob)
solver.Hx
LQR.solve!(sol, solver, prob)
U = copy(sol.U_)
A = solver.Hx*solver.T
b = solver.Hx*solver.L*prob.x0
R_ = solver.Hu
norm(A'*(A*U+b) + R_*U, Inf) < 1e-12

solver.opts[:matbuild] = :lsq
solver.opts[:solve_type] = :naive
@btime LQR.solve!($sol, $solver,$prob)
solver.opts[:matbuild] = :Ab
solver.opts[:solve_type] = :cholesky
@btime LQR.solve!($sol, $solver,$prob)
prob.x0

H = solver.H
cholesky(H)\solver.y
A = ones(4,3)
D = Diagonal([1,2,3,4])
D*A
