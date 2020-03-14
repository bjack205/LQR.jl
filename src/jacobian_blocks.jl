using TrajOptCore

struct JacobianBlock{n,m,p,T}
    Ā::SizedMatrix{n,n,T,2}
    B̄::SizedMatrix{n,m,T,2}
    C::SizedMatrix{p,n,T,2}
    D::SizedMatrix{p,m,T,2}
    d::SizedVector{p,T,1}
    A::SizedMatrix{n,n,T,2}
    B::SizedMatrix{n,m,T,2}
end
@inline JacobianBlock(n,m,p) = JacobianBlock{Float64}(n,m,p)
function JacobianBlock{T}(n,m,p) where T
    JacobianBlock(
        SizedMatrix{n,n,T,2}(zeros(T,n,n)) ,
        SizedMatrix{n,m,T,2}(zeros(T,n,m)),
        SizedMatrix{p,n,T,2}(zeros(T,p,n)),
        SizedMatrix{p,m,T,2}(zeros(T,p,m)),
        SizedVector{p,T,1}(zeros(T,p)),
        SizedMatrix{n,n,T,2}(zeros(T,n,n)),
        SizedMatrix{n,m,T,2}(zeros(T,n,m))
    )
end

Base.size(::JacobianBlock{n,m,p}) where {n,m,p} = 2n+p,n+m

con_dim(::JacobianBlock{<:Any,<:Any,p}) where p = p

function copy_jacobian!(D, con::JacobianBlock{n,m,p}) where {n,m,p}
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ip = SVector{p}(1:p)
    D[ix,ix] .= con.Ā
    D[ix,iu] .= con.B̄
    D[ip .+ n, ix] .= con.C
    D[ip .+ n, iu] .= con.D
    D[ix .+ (n + p), ix] .= con.A
    D[ix .+ (n + p), iu] .= con.B
end

struct ConstraintBlock{T}
    Y::Matrix{T}
    JYt::Matrix{T}
    YYt::Matrix{T}
    M::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    F::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
    L::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
end
function ConstraintBlock{T}(n1,p,n2,w) where T
    n = n1+p+n2
    Y = zeros(T,n,w)
    JYt = Matrix(Y')
    YYt = zeros(T,n,n)
    M = view(Y,1:n1,1:w)
    F = view(Y,n1 .+ (1:p),1:w)
    L = view(Y,(n1 + p) .+ (1:n2),1:w)
    ConstraintBlock{T}(Y,JYt,YYt,M,F,L)
end

@inline dims(con::ConstraintBlock) = size(con.M,1), size(con.F,1), size(con.L,1), size(con.Y,2)

function jacobian_dims(conSet::ConstraintSet)
    n,m = size(conSet)
    N = length(conSet.p)
    p_coupled_prev = zeros(Int,N)
    p_stage = zeros(Int,N)
    p_coupled_curr = zeros(Int,N)
    for con in conSet.constraints
        for (i,k) in enumerate(con.inds)
            if TrajOptCore.contype(con) <: Coupled
                p_coupled_prev[k+1] += length(con)
                p_coupled_curr[k] += length(con)
            elseif TrajOptCore.contype(con) <: Stage
                p_stage[k] += length(con)
            end
        end
    end
    return p_coupled_prev, p_stage, p_coupled_curr
end

function build_jacobian_blocks(conSet::ConstraintSet{T}) where T
    n,m = size(conSet)
    N = length(conSet.p)
    p_coupled_prev, p_stage, p_coupled_curr = jacobian_dims(conSet)
    [ConstraintBlock{T}(p_coupled_prev[i], p_stage[i], p_coupled_curr[i], n+m*(i<N)) for i = 1:N]
end

function link_jacobians(blocks, con::ConstraintVals{<:Any,W,<:Any,p}, off) where {W<:Stage,p}
    sizes = dims.(blocks)
    nm = sizes[1][4]
    if W == State
        iz = 1:state_dim(con)
    elseif W == Control
        m = control_dim(con)
        iz = (1:m) .+ (nm-m)
    else
        iz = 1:nm
    end
    ip = off .+ (1:p)
    ∇c = [view(blocks[k].Y, ip, iz) for k in con.inds, i = 1:1]
    ConstraintVals(con, ∇c)
end

"""
Creates a `ConstraintVals` whose constraint Jacobians are views into `blocks`
Each Jacobian block corresponds to a single knot point ``k`` has the following structure
    ``\begin{bmatrix} C_{prev} \\ S \\ C_{curr} \end{bmatrix}``
where ``C_{prev}`` are the blocks of a coupled constraint for the second time step,
``S`` are the constraints for the current time step, and
``C_{cur}`` and the blocks of a coupled constraint for the first (current) time step.

all time steps for which the current coupled constraint applies.
"""
function link_jacobians(blocks, con::ConstraintVals{<:Any,<:Coupled,<:Any,p}, off)  where p
    N = length(blocks)
    nm = size(blocks[1].Y,2)
    iz = 1:nm
    ip = off .+ (1:p)
    nw = TrajOptCore.widths(con.con)
    function build_view(k,i)
        if i == 1  # current step, place at bottom
            p_coupled, p_stage = dims(blocks[k])
            ip_ = ip .+  (p_coupled + p_stage)
            return view(blocks[k].Y, ip_, iz)
        else
            iz_ = 1:size(blocks[k+1].Y,2)
            return view(blocks[k+1].Y, ip, iz_)  # next time step, place a top of next block
        end
    end
    ∇c = [build_view(k,i) for k in con.inds, i = 1:2]
    ConstraintVals(con, ∇c)
end

function link_jacobians(blocks, conSet::ConstraintSet)
    off_stage = 0
    off_coupled = 0
    for (i,con) in enumerate(conSet.constraints)
        if TrajOptCore.contype(con.con) <: Stage
            conSet.constraints[i] = link_jacobians(blocks, con, off_stage)
            off_stage += length(con)
        elseif TrajOptCore.contype(con.con) <: Coupled
            conSet.constraints[i] = link_jacobians(blocks, con, off_coupled)
            off_coupled += length(con)
        else
            throw(ErrorException("$(TrajOptCore.contype(con)) not a supported constraint type."))
        end
    end
end

function jacobian_views!(D, conSet)
    p1, ps, p2 = jacobian_dims(conSet)
    N = length(p1)
    n,m = size(conSet)
    p_ = p1 .+ ps
    p  = p_ .+ p2
    off1, off2 = 0,0
    map(1:N) do k
        ip = off1 .+ (1:p[k])
        if k < N
            iz = off2 .+ (1:n+m)
        else
            iz = off2 .+ (1:n)
        end
        off1 += p_[k]  # shift down by p_coupled_prev + p_stage
        off2 += n+m
        view(D, ip, iz)
    end
end

function copy_jacobians!(D::Vector{<:SubArray}, blocks::Vector{<:ConstraintBlock})
    for (d,b) in zip(D, blocks)
        d .= b.Y
    end
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#      COST FUNCTIONS       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

function TrajOptCore.hessian(J::QuadraticCost{<:Diagonal{<:Any,<:SVector},<:Diagonal{<:Any,<:SVector}})
    if sum(J.H) == 0
        return Diagonal([J.Q.diag; J.R.diag])
    else
        Hx = [J.Q J.H']
        Hu = [J.H J.R]
        return [Hx; Hu]
    end
end

function TrajOptCore.hessian(J::DiagonalCost)
    return Diagonal([J.Q.diag; J.R.diag])
end


function build_H!(H, obj::Objective{<:Diagonal}, ::Size{sa}) where sa
    n,m = sa
    N = length(obj.cost)
    iz = SVector{n+m}(1:n+m)
    ix = SVector{n}(1:n)
    for k = 1:N-1
        H_ = TrajOptCore.hessian(obj.cost[k])
        H[iz,iz] = H_
        iz = iz .+ (n+m)
    end
    ix = iz[ix]
    H[ix,ix] = obj.cost[N].Q
    return nothing
end

function build_H(obj::Objective{<:Diagonal{n,m}}) where {n,m}
    N = length(obj.cost)
    NN = N*n + (N-1)*m
    Diagonal(zeros(NN))
    build_H!(H, obj)
end

function build_H!(H, obj::Objective{<:DiagonalCost{n,m}}) where {n,m}
    N = length(obj.cost)
    j = 1

    for (k,costfun) in enumerate(obj.cost)
        for v in diag(costfun.Q)
            H[j,j] = v
            j += 1
        end
        if k < N
            for v in diag(costfun.R)
                H[j,j] = v
                j += 1
            end
        end
    end
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#        BLOCK TRIANGULAR FACTORS         #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

struct BlockTriangular3{p1,ps,p2,T}
    A::SizedMatrix{p1,p1,T,2}
    B::SizedMatrix{ps,ps,T,2}
    C::SizedMatrix{p2,p2,T,2}
    D::SizedMatrix{ps,p1,T,2}
    E::SizedMatrix{p2,ps,T,2}
    F::SizedMatrix{p2,p1,T,2}
    μ::SVector{ps,T}  # lagrange multiplier for stage constraints
    λ::SVector{p2,T}  # lagrange multiplier for coupled constraints
    c::SVector{ps,T}  # stage constraint violation
    d::SVector{p2,T}  # coupled constraint violation
end

@inline BlockTriangular3(p1,ps,p2) = BlockTriangular3{Float64}(p1,ps,p2)
function BlockTriangular3{T}(p1,ps,p2) where T
    BlockTriangular3{p1,ps,p2,T}(
        SizedMatrix{p1,p1,T,2}(zeros(T,p1,p1)),
        SizedMatrix{ps,ps,T,2}(zeros(T,ps,ps)),
        SizedMatrix{p2,p2,T,2}(zeros(T,p2,p2)),
        SizedMatrix{ps,p1,T,2}(zeros(T,ps,p1)),
        SizedMatrix{p2,ps,T,2}(zeros(T,p2,ps)),
        SizedMatrix{p2,p1,T,2}(zeros(T,p2,p1)),
        SVector{ps}(zeros(T,ps)),
        SVector{p2}(zeros(T,p2)),
        SVector{ps}(zeros(T,ps)),
        SVector{p2}(zeros(T,p2))
    )
end

function copy_block!(A, block::BlockTriangular3{p1,ps,p2}, off::Int) where {p1,ps,p2}
    ip1 = off .+ SVector{p1}(1:p1)
    ips = off .+ SVector{ps}((1:ps) .+ p1)
    ip2 = off .+ SVector{p2}((1:p2) .+ (p1+ps))
    A[ip1,ip1] .+= block.A  # this works since the first A block is always empty
    A[ips,ips] .= block.B
    A[ip2,ip2] .= block.C
    A[ips,ip1] .= block.D
    A[ip2,ips] .= block.E
    A[ip2,ip1] .= block.F
end

function build_shur_factors(conSet::ConstraintSet)
    p1,ps,p2 = jacobian_dims(conSet)
    N = length(p1)
    map(1:N) do k
        BlockTriangular3(p1[k], ps[k], p2[k])
    end
end

@inline dims(::BlockTriangular3{p1,ps,p2}) where {p1,ps,p2} = (p1,ps,p2)

function calculate_shur_factors!(F::Vector{<:BlockTriangular3},
        obj::Objective{<:DiagonalCost}, blocks)
    @assert isempty(F[1].A)
    for (res,costfun,block) in zip(F,obj.cost,blocks)
        shur!(res, costfun, block)
    end
end

function copy_shur_factors(S, F::Vector{<:BlockTriangular3})
    off = 0
    for k in eachindex(F)
        copy_block!(S, F[k], off)
        off += sum(dims(F[k])[1:2])
    end
end

# function shur!(res::BlockTriangular3, J::QuadraticCost, con::JacobianBlock)
#     Q,R,H = J.Q, J.R, J.H
#     Ā,B̄ = con.Ā, con.B̄
#     C,D = con.C, con.D
#     A,B = con.A, con.B
#
#     _quad_expansion!(res.A, Ā, B̄, Q, R, H)
#     _quad_expansion!(res.B, C, D, Q, R, H)
#     _quad_expansion!(res.C, A, B, Q, R, H)
#     _quad_expansion!(res.D, C, D, Q, R, H, Ā, B̄)
#     _quad_expansion!(res.E, A, B, Q, R, H, C, D)
#     _quad_expansion!(res.F, A, B, Q, R, H, Ā, B̄)
# end
#
# function _quad_expansion!(res, A, B, Q, R, H, A2=A, B2=B)
#     res .= A*Q*A2' .+ A*H'B2' .+ B*H*A2' .+ B*R*B2'
# end

function shur!(res::BlockTriangular3{p1,ps,p2}, J::Union{<:DiagonalCost,<:QuadraticCost},
        con::ConstraintBlock) where {p1,ps,p2}
    n = length(J.q)
    if size(con.Y,2) == n  # terminal cost
        H = J.Q
    else
        H = TrajOptCore.hessian(J)
    end
    mul!(con.JYt, H, con.Y')
    mul!(con.YYt, con.Y, con.JYt)
    YYt = con.YYt

    ip1 = SVector{p1}(1:p1)
    ips = SVector{ps}((1:ps) .+ p1)
    ip2 = SVector{p2}((1:p2) .+ (p1+ps))

    res.A .= YYt[ip1,ip1]
    res.B .= YYt[ips,ips]
    res.C .= YYt[ip2,ip2]
    res.D .= YYt[ips,ip1]
    res.E .= YYt[ip2,ips]
    res.F .= YYt[ip2,ip1]
end
