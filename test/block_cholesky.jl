using StaticArrays
using LinearAlgebra
using TrajOptCore
using BenchmarkTools
using Test

n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)

A = rand(n,n)
A = A'A
B = rand(m,m)
B = B'B
C = rand(m,n)*1e-3
M = [A C'; C B]
isposdef(M)
b = zeros(n+m)

chol = BlockCholesky(n,m)
@test !chol.block_diag
chol.uplo
cholesky!(chol, A, B, C)
@test chol.F.U ≈ cholesky(M).U

@test chol\b ≈ M\b
b1 = copy(b)
ldiv!(chol,b1)
@test b1 ≈ M\b

@btime cholesky!($chol, $A, $B, $C)
@btime ldiv!($chol, $b1)


# Block Diagonal
chol = BlockCholesky(n,m, block_diag=true)
@test chol.block_diag
cholesky!(chol, A, B)
M = cat(A,B,dims=(1,2))
Mchol = UpperTriangular(cat(chol.A_.factors, chol.B_.factors, dims=(1,2)))
@test Mchol ≈ cholesky(M).U

@test chol.F.U ≈ cholesky(M).U

b1 = copy(b)
ldiv!(chol, b1)
b1
@test b1 ≈ M\b
@test chol\b ≈ M\b

@btime cholesky!($chol, $A, $B)
@btime ldiv!($chol, $b1)


# Diagonal
A = Diagonal(@SVector rand(n))
B = Diagonal(@SVector rand(m))
M = Diagonal([A.diag; B.diag])
copy(M)

chol = BlockCholesky(copy(M),n,m)
cholesky!(chol, A, B)

b1 = copy(b)
ldiv!(chol, b1)
@test chol\b ≈ M\b
@test b1 ≈ M\b

@btime cholesky!($chol, $A, $B)
@btime ldiv!($chol, $b1)
