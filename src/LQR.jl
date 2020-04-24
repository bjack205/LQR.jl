module LQR
using RobotDynamics
using TrajOptCore
using StaticArrays
using InplaceOps
using LinearAlgebra
using SparseArrays
using BenchmarkTools

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

function benchmark_solve!(solver; samples=10, evals=10)
    Z0 = deepcopy(get_trajectory(solver))
    b = @benchmark begin
        initial_trajectory!($solver,$Z0)
        solve!($solver)
    end samples=samples evals=evals
    return b
end

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
