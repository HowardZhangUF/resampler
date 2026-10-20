"""Differentiable affine moment matching, using population covariances.

This is a baseline implementation, not a claim of a new resampling algorithm.
It moves particles rather than sampling ancestors. Full rank is required for
exact covariance matching; eigenvalues below ``floor`` are regularised.
"""
import torch
from torch.autograd.function import once_differentiable


class _SymmetricRoot(torch.autograd.Function):
    """Spectral root with a first-order Fréchet derivative.

    Divided differences avoid differentiating eigenvectors. Repeated positive
    eigenvalues therefore have finite gradients, unlike a naive eigh backward.
    """
    @staticmethod
    def forward(ctx,matrix,inverse,floor):
        eigenvalues,vectors=torch.linalg.eigh((matrix+matrix.transpose(-1,-2))*.5)
        clipped=eigenvalues.clamp_min(floor)
        roots=clipped.sqrt()
        values=roots.reciprocal() if inverse else roots
        ctx.save_for_backward(eigenvalues,clipped,roots,vectors)
        ctx.inverse=inverse;ctx.floor=floor
        return (vectors*values.unsqueeze(-2))@vectors.transpose(-1,-2)

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        eigenvalues,clipped,roots,vectors=ctx.saved_tensors
        raw_difference=eigenvalues.unsqueeze(-1)-eigenvalues.unsqueeze(-2)
        clipped_difference=clipped.unsqueeze(-1)-clipped.unsqueeze(-2)
        unequal=raw_difference!=0
        ratio=clipped_difference/torch.where(unequal,raw_difference,torch.ones_like(raw_difference))
        active=(eigenvalues>=ctx.floor).to(eigenvalues.dtype)
        ratio=torch.where(unequal,ratio,active.unsqueeze(-1)*active.unsqueeze(-2))
        kernel=1/(roots.unsqueeze(-1)+roots.unsqueeze(-2))
        if ctx.inverse:kernel=-kernel/(roots.unsqueeze(-1)*roots.unsqueeze(-2))
        symmetric_gradient=(gradient+gradient.transpose(-1,-2))*.5
        basis_gradient=vectors.transpose(-1,-2)@symmetric_gradient@vectors
        result=vectors@(kernel*ratio*basis_gradient)@vectors.transpose(-1,-2)
        return (result+result.transpose(-1,-2))*.5,None,None


def eigen_resample(particles, log_weights, floor=1e-10):
    """Return equally weighted particles; shapes (..., N, D), (..., N)."""
    weights = torch.softmax(log_weights, dim=-1)
    mean = (weights.unsqueeze(-1) * particles).sum(dim=-2, keepdim=True)
    centered = particles - mean
    target = centered.transpose(-1, -2) @ (weights.unsqueeze(-1) * centered)
    reference = particles - particles.mean(dim=-2, keepdim=True)
    source = reference.transpose(-1, -2) @ reference / particles.shape[-2]
    inverse_root=_SymmetricRoot.apply(source,True,floor)
    target_root=_SymmetricRoot.apply(target,False,floor)
    return mean + reference @ inverse_root @ target_root
