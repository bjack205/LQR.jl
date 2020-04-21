# using TrajectoryOptimization
using TrajOptCore
using LQR
using RobotDynamics
using RobotZoo
using Random

using StaticArrays
using LinearAlgebra
using BenchmarkTools
using SparseArrays
import LQR: KnotPoint

function DoubleIntegrator(D=3, N=101; constrained=false, dense_cost=false)
    Random.seed!(1)
    model = RobotZoo.DoubleIntegrator(D)
    n,m = size(model)
    tf = 2.0
    dt = (N-1)/tf
    x0,u0 = zeros(model)
    x0 = [(@SVector fill(1.0,D)); (@SVector zeros(D))]
    xf = @SVector zeros(n)
    # x0 = @SVector fill(10.0,n)
    z0 = KnotPoint(x0,u0,dt)

    # Objective
    Q = Diagonal([(@SVector fill(10.0,D)); (@SVector fill(1.0,D))])
    R = Diagonal(@SVector fill(0.1,m))
    if dense_cost
        Q = @SMatrix rand(n,n) #SMatrix{n,n}(Q)
        R = @SMatrix rand(m,m) #SMatrix{m,m}(R)
        Q = Q'Q + I
        R = R'R + I
    end
    obj = LQRObjective(Q,R,10Q,xf,N)

    # Constraints
    conSet = ConstraintList(n,m,N)
    p = max(D-2,1) # constrain it to a plane
    A = SizedMatrix{p,n}([rand(p,D) rand(p,D)])
    b = zeros(p)
    con = LinearConstraint(n,m,A,b,Equality(),1:n)
    add_constraint!(conSet, con, 2:N-1)
    goal = GoalConstraint(xf)
    add_constraint!(conSet, goal, N:N)

    # dyn_con = DynamicsConstraint{RK3}(model, N)
    # add_constraint!(conSet, dyn_con, 1:N-1, 1)
    # xinit = GoalConstraint(x0)
    # add_constraint!(conSet, xinit, 1:1, 1)


    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    TrajOptCore.add_dynamics_constraints!(prob)
    prob
end

function Cartpole(method=:none)

    model = RobotZoo.Cartpole()
    n,m = size(model)
    N = 101
    tf = 5.
    dt = tf/(N-1)

    Q = 1.0e-2*Diagonal(@SVector ones(n))
    Qf = 100.0*Diagonal(@SVector ones(n))
    R = 1.0e-1*Diagonal(@SVector ones(m))
    x0 = @SVector zeros(n)
    xf = @SVector [0, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    u_bnd = 3.0
    conSet = ConstraintList(n,m,N)
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(xf)
    # add_constraint!(conSet, bnd, 1:N-1)
    add_constraint!(conSet, goal, N:N)

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem{RK3}(model, obj, conSet, x0, xf, Z, N, 0.0, tf)
    rollout!(prob)
    TrajOptCore.add_dynamics_constraints!(prob)

    return prob
end
# TrajectoryOptimization.add_dynamics_constraints!(prob)

# # Dynamics linearization
# A,B = linearize(RK3,model,z0)
# Ā,B̄ = SizedMatrix{n,n}(1.0I(n)), zero(B)
#
# if constrained
#     # Constraints
#     pu = 1 # set a random direction the controls can't use
#     p = px+pu
#
#     cons = map(1:N) do k
#         if k == 1
#             C_ = SizedMatrix{pu,n}(zeros(pu,n))
#             D_ = SizedMatrix{pu,m}(rand(pu,m))
#             d = SizedVector{pu}(zeros(pu))
#         elseif k == N
#             C_ = SizedMatrix{px,n}([rand(px,D) zeros(px,D)])
#             D_ = SizedMatrix{px,m}(zeros(pu,m))
#             d = SizedVector{px}(zeros(px))
#         else
#             C_ = SizedMatrix{p,n}([rand(px,D) zeros(px,D); zeros(pu,n)])
#             D_ = SizedMatrix{p,m}([zeros(px,m); rand(pu,m)])
#             d = SizedVector{p}(zeros(p))
#         end
#         JacobianBlock(Ā,B̄, C_,D_,d, A,B)
#     end
#
#     prob = LCRProblem(obj, cons, x0, tf, N)
# else
#     prob = LQRProblem(10Q, Q, R, A, B, x0, u0, tf, N)
# end
# return prob
