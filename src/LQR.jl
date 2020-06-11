module LQR
using TrajectoryOptimization
using RobotDynamics
# using TrajOptCore
using StaticArrays
# using InplaceOps
using LinearAlgebra
using SparseArrays
using BenchmarkTools
const TO = TrajectoryOptimization

abstract type AbstractSolver end

# using TrajectoryOptimization: num_constraints, ConVal


include("lqr_problem.jl")
include("least_squares.jl")
include("dynamic_programming.jl")
# include("conblocks.jl")
# include("block_cholesky.jl")
# include("jacobian_blocks.jl")
# include("constrained_problem.jl")
# include("meritfunctions.jl")
# include("line_search.jl")
# include("cholesky_solve.jl")
# include("cholesky_solver.jl")
# include("sparse_solver.jl")

end # module
