module LQR
using RobotDynamics
using TrajOptCore
using StaticArrays
using InplaceOps
using LinearAlgebra
using SparseArrays

abstract type AbstractSolver end

import TrajOptCore: num_constraints

export
    LQRProblem,
    LCRProblem,
    LQRSolution,
    LeastSquaresSolver,
    DPSolver,
    ConstraintBlock,
    solve!

export
    num_constraints,
    num_vars,
    dims

include("lqr_problem.jl")
include("least_squares.jl")
# include("dynamic_programming.jl")
include("conblocks.jl")
include("block_cholesky.jl")
include("jacobian_blocks.jl")
include("constrained_problem.jl")
include("cholesky_solve.jl")
include("cholesky_solver.jl")
include("sparse_solver.jl")

end # module
