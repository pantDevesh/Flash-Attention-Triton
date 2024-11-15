"""
Flash Attention Implementation in Triton
"""

from re import M, S
from weakref import ref
from git import HEAD
import torch
import triton
import triton.language as tl

# to convert a normal function to a triton op, we need to use the triton.jit decorator
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M,
    O,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    NUM_HEADS
    SEQ_LEN,
    HEAD_DIM,
    STAGE,
):
    pass

class TritonAttention(torch.autograd.Function):
    
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM == HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V, "Head dimension mismatch"
        
        # preallocate output tensor
        O = torch.empty_like(Q) 
        stage = 3 if causal else 1
        
        # launch grid
        # we want to parallelize along the batch and head dimensions, as well as the sequence length divided by block size
        grid = lambda args:(
            # ceil division of the sequence length by the block size
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE"]), # which block in the sequence
            BATCH_SIZE*NUM_HEADS,  # Which head of which batch
            1, # Z-dimension in the Cuda launch grid
        )
        # Number of parallel programs: (BATCH_SIZE * NUM_HEADS * SEQ_LEN / BLOCK_SIZE)
        
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN),
            dtype=torch.float16,
            device='cuda'
        ) # M is the logsumexp for the backward pass one for each query
        
        _attn_fwd[grid](
            Q=Q, # pointer to the start of the Q tensor
            K=K, # pointer to the start of the K tensor
            V=V, # pointer to the start of the V tensor
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage,
        )
        # Save information for backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        return O


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal= False, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device='cuda'
        ).normal_(mean=0, std=1).requires_grad_()
    )
    
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device='cuda'
        ).normal_(mean=0, std=1).requires_grad_()
    )
    
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
            dtype=dtype,
            device='cuda'
        ).normal_(mean=0, std=1).requires_grad_()
    )
    
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5) # Softmax (q * k^T / sqrt(d_k))
    d0 = torch.rand_like(Q) # Required for backward pass
    
    # Naive Attention Implementation
    MASK = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).to(dtype).to('cuda')
    # print(MASK)
    # Q: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    # K: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    # V: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(d0)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    
    # triton Attention Implementation
    tri_out = TritonAttention.apply(Q, K, V, MASK, softmax_scale, causal)
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    
    # Check
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol), f"O mismatch"
    assert torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol), f"dV mismatch"
    assert torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol), f"dK mismatch"
    assert torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol), f"dQ mismatch"
    print("All checks pass")
    
    
    
test_op(1, 1, 4, 4)
     