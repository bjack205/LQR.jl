struct LCRProblem{n,m,T}
    obj::Objective{<:QuadraticCost}
    con::Vector{JacobianBlock{n,m,p,T} where p}
    x0::SVector{n,T}
    tf::T
    N::Int
end

TrajOptCore.num_constraints(prob::LCRProblem) = con_dim.(prob.con)
@inline Base.size(prob::LCRProblem{n,m}) where {n,m} = n,m,prob.N
@inline function num_vars(prob)
    n,m,N = size(prob)
    N*n + (N-1)*m
end

# function build_jacobian!(D, con::Vector{<:JacobianBlock{n,m,p}}) where {n,m,p}
#     N = length(con)
#
#     for k = 1:N
#         Di =
#     end
# end
