
"""
    ConstraintBlock

Jacobian block structure
```julia
Y = [
    D2   # previous dynamics (n0, n+m)
    C    # stage constraints (p, n+m)
    D    # current dynamics
]

Jacobian block violation
```julia
y = [
    c    # stage constraint (p,)
    d    # dynamics (n,)
]
```

Overall constraint Jacobian structure
```julia
[
  C1
  D11 D21
      C2
      D22 D32
          C3
          D33 D43 ...
               ⋮ ⋱
                    DN
                    CN
]
```
"""
struct ConstraintBlock{T,VT,MT,MV}
    y::VT                                            # constraint values [c; d]
    Y::MT                                            # constraint Jacobian [D2; C; D1]
    YYt::Matrix{T}                                   # outer product => Shur compliment
    JYt::Matrix{T}                                   # partial Shur compliment
    YJ::Transpose{T,Matrix{T}}
    r::Vector{T}                                     # partial residual - Y*Jinv*g
	r_::Vector{SubArray{T,1,VT,Tuple{UnitRange{Int}},true}}  # partitions of r: [D2; C; D1]*Jinv*g
	res::Vector{T}                                   # primal residual: Y'λ + g

    D2::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false} # view of Y for prev dynamics
    C::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}  # view of Y for stage cons
    D1::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false} # view of Y for dynamics
	c::SubArray{T,1,VT,Tuple{UnitRange{Int}},true}   # view of y for stage constraints
	d::SubArray{T,1,VT,Tuple{UnitRange{Int}},true}   # view of y for dynamics constraints
end

function ConstraintBlock(n1::Int, p::Int, n2::Int, w::Int)
    y = zeros(p+n2)
    Y = zeros(n1+p+n2, w)
    YYt = zeros(n1+p+n2, n1+p+n2)
    JYt = zeros(w, n1+p+n2)
    YJ = transpose(JYt)
    r = zeros(n1+p+n2)
	r_ = [view(r, 1:n1), view(r, n1 .+ (1:p)), view(r, (n1+p) .+ (1:n2))]
	res = zeros(w)

    D2 = view(Y, 1:n1, :)
    C = view(Y, n1 .+ (1:p), :)
    D1 = view(Y, (n1+p) .+ (1:n2), :)
	c = view(y, 1:p)
	d = view(y, p .+ (1:n2))
	rc = view(r, 1:p)
	rd = view(r, p .+ (1:n2))

    ConstraintBlock(y, Y, YYt, JYt, YJ, r, r_, res, D2, C, D1, c, d)
end

function ConstraintBlocks(model::AbstractModel, cons::ConstraintList)
	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
    N = length(cons.p)
    blocks = map(1:N) do k
        n1 = p = n2 = 0
        for (inds,con) in zip(cons)
            if k ∈ inds && con isa StageConstraint
                p += length(con)
			end
            if k ∈ inds && con isa CoupledConstraint
                n2 += length(con)
			end
            if (k-1) ∈ inds && con isa DynamicsConstraint
				n1 += n̄
            elseif (k-1) ∈ inds && con isa CoupledConstraint
                n1 += length(con)
            end
        end
        ConstraintBlock(n1, p, n2, n̄+m*(k<N))
    end
    return blocks
end

dims(block::ConstraintBlock) = size(block.D2,1), size(block.C,1), size(block.D1,1)

function copy_blocks!(D,d, blocks::Vector{<:ConstraintBlock})
	off1 = 0
	off2 = 0
	for k in eachindex(blocks)
		n1,p,n2 = dims(blocks[k])
		w = size(blocks[k].Y,2)
		i1 = off1 .+ (1:n1+p+n2)
		i2 = off2 .+ (1:w)
		D[i1,i2] .= blocks[k].Y
		d[(off1+n1) .+ (1:p+n2)] .= blocks[k].y
		off1 += n1+p
		off2 += w
	end
end


"""
	gen_con_inds(cons::ConstraintList, structure::Symbol)

Generate the indices into the concatenated constraint vector for each constraint.
Determines the bandedness of the Jacobian
"""
function gen_con_inds(conSet::ConstraintList, structure=:by_knotpoint)
	n,m = conSet.n, conSet.m
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    # cons = [[@SVector ones(Int,length(con)) for j in eachindex(conSet.inds[i])]
	# 	for (i,con) in enumerate(conSet.constraints)]
	cons = [[1:0 for j in eachindex(conSet.inds[i])] for i in 1:length(conSet)]

    # Dynamics and general constraints
    idx = 0
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
			for (j,k) in enumerate(conSet.inds[i])
				cons[i][TrajOptCore._index(con,k)] = idx .+ (1:conLen[i])
				idx += conLen[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				inds = conSet.inds[i]
				if k in inds
					j = k -  inds[1] + 1
					cons[i][j] = idx .+ (1:conLen[i])
					idx += conLen[i]
				end
			end
		end
	elseif structure == :by_block
		sort!(conSet)  # WARNING: may modify the input
		idx = zeros(Int,N)
		for k = 1:N
			for (i,(inds,con)) in enumerate(zip(conSet))
				if k ∈ inds
					j = k - inds[1] + 1
					cons[i][j] = idx[k] .+ (1:length(con))
					idx[k] += length(con)
				end
			end
		end
	end
    return cons
end


struct BlockConstraintSet{T} <: TrajOptCore.AbstractConstraintSet
	convals::Vector{ConVal}
	errvals::Vector{ConVal}
	blocks::Vector{ConstraintBlock{T,Vector{T},Matrix{T},Matrix{T}}}
	λ::Vector{<:Vector}
	active::Vector{<:Vector}
	c_max::Vector{T}
	p::Vector{Int}
end

function BlockConstraintSet(model::AbstractModel, cons::ConstraintList)
	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
	ncon = length(cons)

	# Initialize blocks
	blocks = ConstraintBlocks(model, cons)

	# Get indices of the constraints into the block
	cinds = gen_con_inds(cons, :by_block)

	# Create the ConVals
	useG = model isa LieGroupModel
	errvals = map(enumerate(zip(cons))) do (i,(inds,con))
	    C,c = TrajOptCore.gen_convals(blocks, cinds[i], con, inds)
	    ConVal(n̄, m, con, inds, C, c, useG)
	end
	convals = map(errvals) do errval
		ConVal(n, m, errval)
	end
	errvals = convert(Vector{ConVal}, errvals)
	convals = convert(Vector{ConVal}, convals)

	# Other vars
    λ = map(1:ncon) do i
        p = length(cons[i])
        [@SVector zeros(p) for i in cons.inds[i]]
    end
    a = map(1:ncon) do i
        p = length(cons[i])
        [@SVector ones(Bool,p) for i in cons.inds[i]]
    end
    c_max = zeros(ncon)

	BlockConstraintSet(convals, errvals, blocks, λ, a, c_max, copy(cons.p))
end

@inline TrajOptCore.get_convals(conSet::BlockConstraintSet) = conSet.convals
@inline TrajOptCore.get_errvals(conSet::BlockConstraintSet) = conSet.errvals

# function TrajOptCore.residual!(res, conSet::BlockConstraintSet)
# 	for (i,conval) in enumerate(TrajOptCore.get_errvals(conSet))
# 		TrajOptCore.residual!(res, conval, conSet.λ[i])
# 	end
# end
