using TrajOptCore
import TrajOptCore: ConstraintBlocks, hessian, gradient


# struct JacobianBlock{n,m,p,T}
#     Ā::SizedMatrix{n,n,T,2}
#     B̄::SizedMatrix{n,m,T,2}
#     C::SizedMatrix{p,n,T,2}
#     D::SizedMatrix{p,m,T,2}
#     d::SizedVector{p,T,1}
#     A::SizedMatrix{n,n,T,2}
#     B::SizedMatrix{n,m,T,2}
# end
# @inline JacobianBlock(n,m,p) = JacobianBlock{Float64}(n,m,p)
# function JacobianBlock{T}(n,m,p) where T
#     JacobianBlock(
#         SizedMatrix{n,n,T,2}(zeros(T,n,n)) ,
#         SizedMatrix{n,m,T,2}(zeros(T,n,m)),
#         SizedMatrix{p,n,T,2}(zeros(T,p,n)),
#         SizedMatrix{p,m,T,2}(zeros(T,p,m)),
#         SizedVector{p,T,1}(zeros(T,p)),
#         SizedMatrix{n,n,T,2}(zeros(T,n,n)),
#         SizedMatrix{n,m,T,2}(zeros(T,n,m))
#     )
# end
#
# Base.size(::JacobianBlock{n,m,p}) where {n,m,p} = 2n+p,n+m
#
# con_dim(::JacobianBlock{<:Any,<:Any,p}) where p = p
#
# function copy_jacobian!(D, con::JacobianBlock{n,m,p}) where {n,m,p}
#     ix = SVector{n}(1:n)
#     iu = SVector{m}(n .+ (1:m))
#     ip = SVector{p}(1:p)
#     D[ix,ix] .= con.Ā
#     D[ix,iu] .= con.B̄
#     D[ip .+ n, ix] .= con.C
#     D[ip .+ n, iu] .= con.D
#     D[ix .+ (n + p), ix] .= con.A
#     D[ix .+ (n + p), iu] .= con.B
# end
#
# struct ConstraintBlock{T}
#     Y::Matrix{T}
#     JYt::Matrix{T}
#     YYt::Matrix{T}
#     M::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
#     F::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
#     L::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
# end
# function ConstraintBlock{T}(n1,p,n2,w) where T
#     n = n1+p+n2
#     Y = zeros(T,n,w)
#     JYt = Matrix(Y')
#     YYt = zeros(T,n,n)
#     M = view(Y,1:n1,1:w)
#     F = view(Y,n1 .+ (1:p),1:w)
#     L = view(Y,(n1 + p) .+ (1:n2),1:w)
#     ConstraintBlock{T}(Y,JYt,YYt,M,F,L)
# end
#
# @inline dims(con::ConstraintBlock) = size(con.M,1), size(con.F,1), size(con.L,1), size(con.Y,2)
#
# function jacobian_dims(conSet::ConstraintSet)
#     n,m = size(conSet)
#     N = length(conSet.p)
#     p_coupled_prev = zeros(Int,N)
#     p_stage = zeros(Int,N)
#     p_coupled_curr = zeros(Int,N)
#     for con in conSet.constraints
#         for (i,k) in enumerate(con.inds)
#             if TrajOptCore.contype(con) <: Coupled
#                 p_coupled_prev[k+1] += length(con)
#                 p_coupled_curr[k] += length(con)
#             elseif TrajOptCore.contype(con) <: Stage
#                 p_stage[k] += length(con)
#             end
#         end
#     end
#     return p_coupled_prev, p_stage, p_coupled_curr
# end
#
# function build_jacobian_blocks(conSet::ConstraintSet{T}) where T
#     n,m = size(conSet)
#     N = length(conSet.p)
#     p_coupled_prev, p_stage, p_coupled_curr = jacobian_dims(conSet)
#     [ConstraintBlock{T}(p_coupled_prev[i], p_stage[i], p_coupled_curr[i], n+m*(i<N)) for i = 1:N]
# end
#
# function link_jacobians(blocks, con::ConstraintVals{<:Any,W,<:Any,p}, off) where {W<:Stage,p}
#     sizes = dims.(blocks)
#     nm = sizes[1][4]
#     if W == State
#         iz = 1:state_dim(con)
#     elseif W == Control
#         m = control_dim(con)
#         iz = (1:m) .+ (nm-m)
#     else
#         iz = 1:nm
#     end
#     ip = 1:p
#     ∇c = [view(blocks[k].Y, ip .+ (off[k] + dims(blocks[k])[1]), iz) for k in con.inds, i = 1:1]
#     for k in con.inds
#         off[k] += p
#     end
#     ConstraintVals(con, ∇c)
# end
#
# """
# Creates a `ConstraintVals` whose constraint Jacobians are views into `blocks`
# Each Jacobian block corresponds to a single knot point ``k`` has the following structure
#     ``\begin{bmatrix} C_{prev} \\ S \\ C_{curr} \end{bmatrix}``
# where ``C_{prev}`` are the blocks of a coupled constraint for the second time step,
# ``S`` are the constraints for the current time step, and
# ``C_{cur}`` and the blocks of a coupled constraint for the first (current) time step.
#
# all time steps for which the current coupled constraint applies.
# """
# function link_jacobians(blocks, con::ConstraintVals{<:Any,<:Coupled,<:Any,p}, off)  where p
#     N = length(blocks)
#     nm = size(blocks[1].Y,2)
#     iz = 1:nm
#     ip = 1:p
#     nw = TrajOptCore.widths(con.con)
#     function build_view(k,i)
#         if i == 1  # current step, place at bottom
#             p_coupled, p_stage = dims(blocks[k])
#             ip_ = ip .+  (p_coupled + p_stage + off[k])
#             off[k] += p
#             return view(blocks[k].Y, ip_, iz)
#         else
#             iz_ = 1:size(blocks[k+1].Y,2)
#             return view(blocks[k+1].Y, ip, iz_)  # next time step, place a top of next block
#         end
#     end
#     ∇c = [build_view(k,i) for k in con.inds, i = 1:2]
#     ConstraintVals(con, ∇c)
# end
#
# function link_jacobians(blocks, conSet::ConstraintSet)
#     off_stage = zero(conSet.p)
#     off_coupled = zero(conSet.p)
#     for (i,con) in enumerate(conSet.constraints)
#         if TrajOptCore.contype(con.con) <: Stage
#             conSet.constraints[i] = link_jacobians(blocks, con, off_stage)
#         elseif TrajOptCore.contype(con.con) <: Coupled
#             conSet.constraints[i] = link_jacobians(blocks, con, off_coupled)
#         else
#             throw(ErrorException("$(TrajOptCore.contype(con)) not a supported constraint type."))
#         end
#     end
# end
#
#
# function copy_jacobians!(D::Vector{<:SubArray}, blocks::Vector{<:ConstraintBlock})
#     for (d,b) in zip(D, blocks)
#         d .= b.Y
#     end
# end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#      COST FUNCTIONS       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# function TrajOptCore.hessian(J::QuadraticCost{<:Diagonal{<:Any,<:SVector},<:Diagonal{<:Any,<:SVector}})
#     if sum(J.H) == 0
#         return Diagonal([J.Q.diag; J.R.diag])
#     else
#         Hx = [J.Q J.H']
#         Hu = [J.H J.R]
#         return [Hx; Hu]
#     end
# end
#
# function TrajOptCore.hessian(J::DiagonalCost)
#     if J.terminal
#         J.Q
#     else
#         Diagonal([J.Q.diag; J.R.diag])
#     end
# end
#
# function TrajOptCore.gradient(J::DiagonalCost)
#     if J.terminal
#         J.q
#     else
#         [J.q; J.r]
#     end
# end

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

abstract type BlockTriangular3{p1,ps,p2,T} end

struct BlockUpperTriangular3{p1,ps,p2,T} <: BlockTriangular3{p1,ps,p2,T}
    A::SizedMatrix{p1,p1,T,2}
    B::SizedMatrix{ps,ps,T,2}
    C::SizedMatrix{p2,p2,T,2}
    D::SizedMatrix{p1,ps,T,2}
    E::SizedMatrix{ps,p2,T,2}
    F::SizedMatrix{p1,p2,T,2}
    μ::SizedVector{ps,T,1}  # lagrange multiplier for stage constraints
    λ::SizedVector{p2,T,1}  # lagrange multiplier for coupled constraints
    c::SizedVector{ps,T,1}  # stage constraint violation
    d::SizedVector{p2,T,1}  # coupled constraint violation
end

@inline BlockUpperTriangular3(p1,ps,p2) = BlockLowerTriangular3{Float64}(p1,ps,p2)
function BlockUpperTriangular3{T}(p1,ps,p2,A=SizedMatrix{p1,p1}(zeros(T,p1,p1))) where T
    BlockUpperTriangular3{p1,ps,p2,T}(
        A,  # allow for adjacent A,C blocks to be linked
        SizedMatrix{ps,ps,T,2}(zeros(T,ps,ps)),
        SizedMatrix{p2,p2,T,2}(zeros(T,p2,p2)),
        SizedMatrix{p1,ps,T,2}(zeros(T,p1,ps)),
        SizedMatrix{ps,p2,T,2}(zeros(T,ps,p2)),
        SizedMatrix{p1,p2,T,2}(zeros(T,p1,p2)),
        SizedVector{ps}(zeros(T,ps)),
        SizedVector{p2}(zeros(T,p2)),
        SizedVector{ps}(zeros(T,ps)),
        SizedVector{p2}(zeros(T,p2))
    )
end

struct BlockLowerTriangular3{p1,ps,p2,T} <: BlockTriangular3{p1,ps,p2,T}
    A::SizedMatrix{p1,p1,T,2}
    B::SizedMatrix{ps,ps,T,2}
    C::SizedMatrix{p2,p2,T,2}
    D::SizedMatrix{ps,p1,T,2}
    E::SizedMatrix{p2,ps,T,2}
    F::SizedMatrix{p2,p1,T,2}
    μ::SizedVector{ps,T,1}  # lagrange multiplier for stage constraints
    λ::SizedVector{p2,T,1}  # lagrange multiplier for coupled constraints
    c::SizedVector{ps,T,1}  # stage constraint violation
    d::SizedVector{p2,T,1}  # coupled constraint violation
end

@inline BlockLowerTriangular3(p1,ps,p2) = BlockLowerTriangular3{Float64}(p1,ps,p2)
function BlockLowerTriangular3{T}(p1,ps,p2,A=SizedMatrix{p1,p1}(zeros(T,p1,p1))) where T
    BlockLowerTriangular3{p1,ps,p2,T}(
        A,  # allow for adjacent A,C blocks to be linked
        SizedMatrix{ps,ps,T,2}(zeros(T,ps,ps)),
        SizedMatrix{p2,p2,T,2}(zeros(T,p2,p2)),
        SizedMatrix{ps,p1,T,2}(zeros(T,ps,p1)),
        SizedMatrix{p2,ps,T,2}(zeros(T,p2,ps)),
        SizedMatrix{p2,p1,T,2}(zeros(T,p2,p1)),
        SizedVector{ps}(zeros(T,ps)),
        SizedVector{p2}(zeros(T,p2)),
        SizedVector{ps}(zeros(T,ps)),
        SizedVector{p2}(zeros(T,p2))
    )
end

function build_shur_factors(blocks::ConstraintBlocks{T}, uplo=:L) where T
    p1,ps,p2 = collect(zip(TrajOptCore.dims.(blocks)...))
    N = length(p1)
    if uplo == :L
        F = BlockLowerTriangular3{<:Any,<:Any,<:Any,T}[BlockLowerTriangular3{T}(p1[1], ps[1], p2[1]),]
        Block = BlockLowerTriangular3{T}
    elseif uplo == :U
        F = BlockUpperTriangular3{<:Any,<:Any,<:Any,T}[BlockUpperTriangular3{T}(p1[1], ps[1], p2[1]),]
        Block = BlockUpperTriangular3{T}
    end
    for k = 2:N
        push!(F, Block(p1[k], ps[k], p2[k], F[k-1].C))
    end
    return F
end

@inline dims(::BlockTriangular3{p1,ps,p2}) where {p1,ps,p2} = (p1,ps,p2)

function copy_shur_factors!(S, h, λ, F::Vector{<:BlockTriangular3})
    off = 0
    for k in eachindex(F)
        copy_block!(S, h, λ, F[k], off)
        off += sum(dims(F[k])[1:2])
    end
end

function copy_block!(S, h, λ, block::BlockTriangular3{p1,ps,p2}, off::Int) where {p1,ps,p2}
    ip1 = off .+ SVector{p1}(1:p1)
    ips = off .+ SVector{ps}((1:ps) .+ p1)
    ip2 = off .+ SVector{p2}((1:p2) .+ (p1+ps))
    S[ip1,ip1] .= block.A  # this works since the first A block is always empty
    S[ips,ips] .= block.B
    S[ip2,ip2] .= block.C
    S[ips,ip1] .= block.D
    S[ip2,ips] .= block.E
    S[ip2,ip1] .= block.F
    h[ip2] .= block.d
    h[ips] .= block.c
    λ[ip2] .= block.λ
    λ[ips] .= block.μ
end

function copy_block!(S, h, λ, block::BlockUpperTriangular3{p1,ps,p2}, off::Int) where {p1,ps,p2}
    ip1 = off .+ SVector{p1}(1:p1)
    ips = off .+ SVector{ps}((1:ps) .+ p1)
    ip2 = off .+ SVector{p2}((1:p2) .+ (p1+ps))
    S[ip1,ip1] .= block.A  # this works since the first A block is always empty
    S[ips,ips] .= block.B
    S[ip2,ip2] .= block.C
    S[ip1,ips] .= block.D
    S[ips,ip2] .= block.E
    S[ip1,ip2] .= block.F
    h[ip2] .= block.d
    h[ips] .= block.c
    λ[ip2] .= block.λ
    λ[ips] .= block.μ
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

function calculate_shur_factors!(F::Vector{<:BlockTriangular3},
        Jinv::Vector{<:InvertedQuadratic}, blocks)
    # @assert isempty(F[1].A)
    N = length(blocks)
    for k = 1:N-1
        shur!(Jinv[k], blocks[k])
        copy_shur!(F[k], blocks[k], blocks[k+1])
    end
    shur!(Jinv[N], blocks[N])
    copy_shur!(F[N], blocks[N])
    # for k = 1:N-1
    #     shur!(F[k], obj.cost[k], obj.cost[k+1], blocks[k], blocks[k+1])
    # end
    # _shur!(F[N], obj.cost[N], blocks[N])
 end

function shur!(Jinv, block)
    # H,g = hessian(Jinv), gradient(Jinv)
    # g = StaticKnotPoint(Jinv,q, Jinv.r)

    transpose!(block.JYt, block.Y)
    ldiv!(Jinv.chol, block.JYt)
    # mul!(block.YJ, block.Y, H)
    mul!(block.YYt, block.Y, block.JYt)
    YYt = block.YYt

    g = gradient(Jinv)
    mul!(block.r, block.YJ, g)
end

function copy_shur!(res::BlockTriangular3{p1,ps,p2}, block, block2) where {p1,ps,p2}
    copy_shur!(res, block)
    res.d .+= block2.r_[1]
end

function copy_shur!(res::BlockTriangular3{p1,ps,p2}, block) where {p1,ps,p2}
    YYt = block.YYt
    ip1 = SVector{p1}(1:p1)
    ips = SVector{ps}((1:ps) .+ p1)
    ip2 = SVector{p2}((1:p2) .+ (p1+ps))

    res.A .+= YYt[ip1,ip1]
    res.B .= YYt[ips,ips]
    res.C .= YYt[ip2,ip2]
    res.D .= YYt[ips,ip1]
    res.E .= YYt[ip2,ips]
    res.F .= YYt[ip2,ip1]

    res.c .= block.c .+ block.r_[2]
    res.d .= block.d .+ block.r_[3]
end

function copy_shur!(res::BlockUpperTriangular3{p1,ps,p2}, block) where {p1,ps,p2}
    YYt = block.YYt
    ip1 = SVector{p1}(1:p1)
    ips = SVector{ps}((1:ps) .+ p1)
    ip2 = SVector{p2}((1:p2) .+ (p1+ps))

    res.A .+= YYt[ip1,ip1]
    res.B .= YYt[ips,ips]
    res.C .= YYt[ip2,ip2]
    res.D .= YYt[ip1,ips]
    res.E .= YYt[ips,ip2]
    res.F .= YYt[ip1,ip2]

    res.c .= block.c .+ block.r_[2]
    res.d .= block.d .+ block.r_[3]
end

# function shur!(res::BlockTriangular3{p1,ps,p2},
#         Jinv::Union{<:DiagonalCost,<:QuadraticCost}, J2inv,
#         block::ConstraintBlock, block2::ConstraintBlock) where {p1,ps,p2}
#
#     _shur!(res, Jinv, block)
#
#     H2,g2 = hessian(J2inv), gradient(J2inv)
#     Hg = H2\g2
#     mul!(res.d, block2.D2, Hg, 1.0, 1.0)
#     # res.d .+= block2.D2*(H2\g2)
#     return nothing
# end
#
# function _shur!(res::BlockTriangular3{p1,ps,p2},
#         Jinv::Union{<:DiagonalCost,<:QuadraticCost},
#         block::ConstraintBlock) where {p1,ps,p2}
#     H,g = hessian(Jinv), gradient(Jinv)
#
#     mul!(block.YJ, block.Y, H)
#     mul!(block.YYt, block.YJ, block.Y')
#     YYt = block.YYt
#
#     ip1 = SVector{p1}(1:p1)
#     ips = SVector{ps}((1:ps) .+ p1)
#     ip2 = SVector{p2}((1:p2) .+ (p1+ps))
#
#     res.A .+= YYt[ip1,ip1]
#     res.B .= YYt[ips,ips]
#     res.C .= YYt[ip2,ip2]
#     res.D .= YYt[ips,ip1]
#     res.E .= YYt[ip2,ips]
#     res.F .= YYt[ip2,ip1]
#
#     # res.c .= block.c .+ block.C*(H\g)
#     # res.d .= block.d .+ block.D1*(H\g)
#     Hg = H*g
#     mul!(res.c, block.C, Hg)
#     res.c .+= block.c
#     mul!(res.d, block.D1, Hg)
#     res.d .+= block.d
#     return nothing
# end
