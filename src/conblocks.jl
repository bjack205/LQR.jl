
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
    y::VT
    Y::MT
    YYt::Matrix{T}  # outer product => Shur compliment
    JYt::Matrix{T}   # partial Shur compliment
    YJ::Transpose{T,Matrix{T}}
    r::Vector{T}    # Shur compliment residual

    D2::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}
    C::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}
    D1::SubArray{T,2,MV,Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false}
end

function ConstraintBlock(n1::Int, p::Int, n2::Int, w::Int)
    y = zeros(p+n2)
    Y = zeros(n1+p+n2, w)
    YYt = zeros(n1+p+n2, n1+p+n2)
    JYt = zeros(w, n1+p+n2)
    YJ = Transpose(zeros(n1+p+n2, w))
    r = zero(y)

    D2 = view(Y, 1:n1, :)
    C = view(Y, n1 .+ (1:p), :)
    D1 = view(Y, (n1+p) .+ (1:n2), :)
    ConstraintBlock(y, Y, YYt, JYt, YJ, r, D2, C, D1)
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

    cons = [[@SVector ones(Int,length(con)) for j in eachindex(conSet.inds[i])]
		for (i,con) in enumerate(conSet.constraints)]

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
		idx = zeros(N)
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
