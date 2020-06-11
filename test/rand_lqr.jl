import Pkg; Pkg.activate(joinpath(@__DIR__,".."))
using LQR
using StaticArrays
using LinearAlgebra
using RobotDynamics
using Random
using BenchmarkTools

struct DiscreteLinear{n,m,T} <: AbstractModel
    A::SizedMatrix{n,n,T,2}
    B::SizedMatrix{n,m,T,2}
end

function RobotDynamics.discrete_dynamics(::Type{Q}, model::DiscreteLinear, z::AbstractKnotPoint) where Q <: RobotDynamics.Explicit
    model.A*state(z) + model.B*control(z)
end

function RobotDynamics.discrete_jacobian!(::Type{Q}, ∇f, model::DiscreteLinear,
        z::AbstractKnotPoint{<:Any,n,m}) where {n,m,Q <: RobotDynamics.Explicit}
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ∇f[ix,ix] .= model.A
    ∇f[ix,iu] .= model.B
    return ∇f
end

RobotDynamics.state_dim(::DiscreteLinear{n}) where n = n
RobotDynamics.control_dim(::DiscreteLinear{<:Any,m}) where m = m

n = rand(10:20)
m = n ÷ 2
ix = SVector{n}(1:n)
iu = SVector{m}(n .+ (1:m))

s = shuffle([i < 1.7*n for i = 1:n])  # 70% nonzero entries in Q
Q = Diagonal(s .* @SVector rand(n))*10
R = Diagonal(@SVector ones(m))*0.1

A = Diagonal(randn(n)*0.01 .+ 1)
A = A ./ norm(A.diag)
B = randn(n,m)
eigvals(A)
@assert all(0 .< eigvals(Array(A)) .< 1)

# Check controllability
C = hcat([(A^i) * B for i = 0:n-1]...)
@assert rank(C) == n

# Create Model
model = DiscreteLinear{n,m,Float64}(A,B)
tf = 5.0
N = 11
dt = tf / (N-1)

x,u = rand(model)
z = KnotPoint(x, u, dt)

prob = LQR.LQRProblem(N*Q, Q, R, model.A, model.B, x, u, tf, N)

# Solve with Least Squares
ls = LQR.LeastSquaresSolver(prob)
Z_ls = LQR.Primals(n,m,N,tf)
LQR.solve!(Z_ls, ls, prob)

# Solve with Dynamic Programming
dp = LQR.DPSolver(prob)
Z_dp = LQR.Primals(n,m,N,tf)
LQR.solve!(Z_dp, dp, prob)
Z_ls.Z ≈ Z_dp.Z
norm(Z_ls.X_ - Z_dp.X_, Inf)
maximum(Z_ls.X_ - Z_dp.X_)
