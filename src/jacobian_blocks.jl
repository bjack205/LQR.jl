using TrajOptCore

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

function build_H!(H, obj::Objective{<:TrajOptCore.QuadraticCostFunction})
    N = length(obj)
    n = length(obj[1].q)
    m = length(obj[1].r)
    ix = 1:n
    iu = n .+ (1:m)
    for k = 1:N-1
        H[ix,ix] .= obj[k].Q
        H[iu,iu] .= obj[k].R
        H[ix,iu] .= obj[k].H'
        H[iu,ix] .= obj[k].H
		ix = (n+m) .+ (ix)
		iu = (n+m) .+ (iu)
    end
    H[ix,ix] .= obj[N].Q
    return nothing
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

function build_shur_factors(conSet::BlockConstraintSet{T}, uplo=:L) where T
    p1,ps,p2 = collect(zip(dims.(conSet.blocks)...))
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


"""
	calculate_shur_factors!

Calculate the shur compliment S = D*(G\\D') and the residual r = D*(G\\g) - d
If `Ginv = false`, calculate `S = D*D'` and `r = -d`
"""
function calculate_shur_factors!(F::Vector{<:BlockTriangular3},
        Jinv::Vector{<:InvertedQuadratic}, blocks, Ginv::Bool=true)
    N = length(blocks)
	shur!(Jinv[1], blocks[1], Ginv)
    for k = 2:N
        shur!(Jinv[k], blocks[k], Ginv)
        copy_shur!(F[k-1], blocks[k-1], blocks[k])
    end
    copy_shur!(F[N], blocks[N])
 end

function shur!(Jinv, block, Ginv::Bool=true)
    transpose!(block.JYt, block.Y)
    if Ginv
		ldiv!(Jinv.chol, block.JYt)
		g = gradient(Jinv)
		mul!(block.r, block.YJ, g)
	else
		block.r .*= 0
	end
    mul!(block.YYt, block.Y, block.JYt)
    YYt = block.YYt
end

"""
	copy_shur!

Copy the shur compliment product stored in `block` to the block triangular block `res`
"""
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

    res.c .= block.r_[2] .- block.c
    res.d .= block.r_[3] .- block.d
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

    res.c .= block.r_[2] .- block.c
    res.d .= block.r_[3] .- block.d
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
