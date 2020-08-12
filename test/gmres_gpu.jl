using Solvent, Test, LinearAlgebra, Random, CUDA, KernelAbstractions


@kernel function laplace_kernel!(Ax, x, param)
    i = @index(Global, Linear)
    @inbounds begin
        Ax[i] =  x[i] 
        if i > 1
            Ax[i] += param*x[i-1]
        end
        if i < length(x)
            Ax[i] += param * x[i+1]
        end
    end
end

"""
    laplace!(AX, X, param=-0.01)
        
Applies the 1d Laplace operator `A*x`  where 
```
A[i,i] = 1
A[i,i-1] = A[i,i+1] = param
```
"""
function laplace!(AX, X, param=-0.01)
    event = laplace!(CUDADevice(), 256)(
        AX, X, eltype(X)(param); ndrange=length(X))        
    wait(event)
end

for T in (Float32, Float64)
    Random.seed!(44)
    
    n = 1000
    M = 20
    K = 10
    tol = sqrt(eps(T))

    b = CuArray(randn(n))
    x = copy(b)
    
    linearsolver = LinearSolver(
        laplaceop!,
        GeneralizedMinimalResidualMethod(M = M, K = K),
        x;
        rtol = tol,
        atol = tol,
    )

    linearsolve!(linearsolver, x, b)
    x_ref = SymTridiagonal(fill(T(1),n), fill(T(-0.1),n-1)) \ Array(b)
    @test Array(x) ≈ x_ref
end
