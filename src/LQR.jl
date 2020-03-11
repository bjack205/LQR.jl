module LQR
using RobotDynamics
# using TrajOptCore
using StaticArrays
using InplaceOps
using LinearAlgebra
using SparseArrays

abstract type AbstractSolver end

export
    LQRProblem,
    LeastSquaresSolver

include("lqr_problem.jl")
include("least_squares.jl")

end # module
