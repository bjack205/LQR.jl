struct LCRProblem{n,m,T}
    obj::Objective{<:Union{<:QuadraticCost,<:DiagonalCost}}
    A::Vector{SizedMatrix{n,n,T,2}}      # dynamics state Jacobian
    B::Vector{SizedMatrix{n,m,T,2}}      # dynamics control Jacobian
    C::Vector{SizedMatrix{<:Any,n,T,2}}  # constraint state Jacobian
    D::Vector{SizedMatrix{<:Any,n,T,2}}  # constraint control Jacobian
    d::Vector{MVector{<:Any,T}}    # constraint constant
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

function LCRProblem(prob::Problem)
    N = prob.N

    # Calculate constraints
    evaluate!(prob.constraints, prob.Z)
    jacobian!(prob.constraints, prob.Z)

    A,B, C,D,d = get_linearized_constraints(prob.constraints)

    LCRProblem(prob.obj, A, B, C, D, d, prob.x0, prob.tf, prob.N)
end

function get_linearized_constraints(conSet::ConstraintSet)

    n,m = size(conSet)
    N = length(conSet.p)
    p = copy(conSet.p) # don't count the dynamics constraint
    p[1:n-1] .-= n

    A = [SizedMatrix{n,n}(zeros(n,n)) for k = 1:N]
    B = [SizedMatrix{n,m}(zeros(n,m)) for k = 1:N]
    C = [SizedMatrix{p[k],n}(zeros(p[k],n)) for k = 1:N]
    D = [SizedMatrix{p[k],m}(zeros(p[k],m)) for k = 1:N]
    d = [SizedVector{p[k]}(zeros(p[k]))  for k = 1:N]

    # Copy constraint violation
    copy_vals!(d, conSet)

    # Copy Jacobians
    ix = 1:n
    iu = n .+ (1:m)
    blocks = build_jacobian_blocks(conSet)
    link_jacobians(blocks, conSet)
    for k = 1:N
        if k < N
            A[k] .= blocks[k].L[:,ix]
            B[k] .= blocks[k].L[:,iu]
            D[k] .= blocks[k].F[:,iu]
        end
        C[k] .= blocks[k].F[:,ix]
    end
    return A,B, C,D, d
end

function copy_vals!(d, conSet::ConstraintSet)
    off = zero(conSet.p)
    for con in conSet.constraints
        copy_vals!(d, con, off)
    end
end

function copy_vals!(d, con::ConstraintVals{<:Any,<:Stage,<:Any,p}, off) where p
    ip = SVector{p}(1:p)
    for (i,k) in enumerate(con.inds)
        d[k][ip .+ off[k]] .= con.vals[i]
        off[k] += p
    end
end
copy_vals!(d, con::ConstraintVals, off) = nothing

# function build_jacobian!(D, con::Vector{<:JacobianBlock{n,m,p}}) where {n,m,p}
#     N = length(con)
#
#     for k = 1:N
#         Di =
#     end
# end
