# Preconditioners
Preconditioning seeks to improve the performance of iterative methods
by transforming the problem into an equivalent but better conditioned
system.

```@docs
AbstractPreconditioner
Solvent.Preconditioner
Solvent.PCinitialize!
Solvent.PCapply!
```

## Implementations
```@docs
Solvent.Identity
Solvent.Jacobi
```