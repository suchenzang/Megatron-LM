# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math
import importlib

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import gather_from_sequence_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .mappings import reduce_scatter_to_sequence_parallel_region

from .mappings import _reduce_scatter_along_first_dim, _gather_along_first_dim

from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim, split_tensor_by_given_split_sizes
from .utils import VocabUtility
from megatron import get_args, get_global_memory_buffer
# from megatron.model.fused_bias_gelu import bias_gelu, bias_gelu_back

global fused_layer_norm_cuda
fused_layer_norm_cuda = None



_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    # args = get_args()
    master_weight = master_weight.to(dtype=torch.float)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def _initialize_affine_bias_gpu(bias, init_method):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # Initialize master weight
    with get_cuda_rng_tracker().fork():
        init_method(bias)


def _initialize_affine_bias_cpu(bias, output_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # Initialize master weight
    master_bias = torch.empty(output_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_bias)
    master_bias = master_bias.to(dtype=torch.float)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    bias_list = torch.split(master_bias, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_bias_list = bias_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_bias_list, dim=partition_dim, out=bias)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx,
                 init_method=init.xavier_normal_, use_cpu_initialization=True, dtype=torch.half):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = padding_idx
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        # args = get_args()
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=dtype,
                ))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) \
                | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input.masked_fill_(input_mask, 0.0)
        else:
            masked_input = input_

            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)

        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0.0)

        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        return output


@torch.jit.script
def gelu(x):
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

@torch.jit.script
def gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel, apply_pre_gelu=False, apply_pre_ln=False):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel
        ctx.apply_pre_gelu = apply_pre_gelu
        ctx.apply_pre_ln = apply_pre_ln

        if sequence_parallel:
            assert not ctx.apply_pre_gelu
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            assert not ctx.apply_pre_ln
            total_input = input

        if ctx.apply_pre_gelu:
            total_input = gelu(total_input)

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        if ctx.apply_pre_gelu:
            grad_input = gelu_back(grad_input, total_input)
            total_input = gelu(total_input)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
				       total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                            group=get_tensor_model_parallel_group(),
                                                            async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # reduce scatter scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1


        if ctx.gradient_accumulation_fusion:
            import fused_dense_cuda
            fused_dense_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, use_cpu_initialization=True,
                 no_async_tensor_model_parallel_allreduce=True,
                 init_method_bias=None,
                 dtype=torch.half,
                 sequence_parallel=False,
                 gradient_accumulation_fusion=False):
        super(ColumnParallelLinear, self).__init__()
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        # args = get_args()
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=dtype
                                                ))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    dtype=dtype
                    ))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)

            if init_method_bias is not None:
                if use_cpu_initialization:
                    _initialize_affine_bias_cpu(
                        self.bias, self.output_size, self.output_size_per_partition,
                        0, init_method_bias
                    )
                else:
                    _initialize_affine_bias_gpu(
                        self.bias, init_method_bias
                    )
            else:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                not no_async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, bias, self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce, self.sequence_parallel, False)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, use_cpu_initialization=True,
                 dtype=torch.half,
                 sequence_parallel=False,
                 gradient_accumulation_fusion=False,
                 apply_pre_gelu=False):
        super(RowParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        # args = get_args()
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=dtype,
                                                ))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=dtype))
            setattr(self.bias, 'sequence_parallel', sequence_parallel)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.sequence_parallel = sequence_parallel
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.apply_pre_gelu = apply_pre_gelu


    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, None,
            self.gradient_accumulation_fusion, None, None, self.apply_pre_gelu)
        # All-reduce across all the partitions.
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class TransformerBlockAutograd(torch.autograd.Function):
    """
    This is custom FFN autograd function hardcoded for:
    bias: false,
    layernorm affine: false, ln eps: 1e-5
    sequence_parallel: true,
    activation: gelu,
    gelu, layernorm: always recomputed i.e. no activation memory for these
    """
    @staticmethod
    def forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype):
        import scaled_upper_triang_masked_softmax_cuda
        scaling = head_dim ** -0.5
        matmul_result = torch.empty(
            bsz * (embed_dim_per_partition // head_dim),
            seq_len,
            seq_len,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        # Scale q,k before matmul for stability see https://tinyurl.com/sudb9s96 for math
        matmul_result = torch.baddbmm(
            matmul_result,
            math.sqrt(scaling) * q.transpose(0, 1),
            math.sqrt(scaling) * k.transpose(0, 1).transpose(1, 2),
            beta=0.0,
        )
        # attn_probs = matmul_result
        scale_t = torch.tensor([1.0])
        attn_probs = scaled_upper_triang_masked_softmax_cuda.forward(matmul_result, scale_t[0])
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        return attn, attn_probs


    @staticmethod
    def backward_mha(grad_mha_output, q, k, v, attn_probs, seq_len, bsz, head_dim):
        import scaled_upper_triang_masked_softmax_cuda
        scaling = head_dim ** -0.5
        grad_mha_output= grad_mha_output.view(seq_len, -1, head_dim).transpose(0, 1)
        grad_v = torch.bmm(attn_probs.transpose(1,2), grad_mha_output).transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        grad_attn_probs_out = torch.bmm(grad_mha_output, v.transpose(1, 2))

        grad_attn_probs_in = scaled_upper_triang_masked_softmax_cuda.backward(
            grad_attn_probs_out, attn_probs, 1.0
        )
        grad_q = torch.bmm(
            math.sqrt(scaling) * grad_attn_probs_in,
            math.sqrt(scaling) * k.transpose(0,1)
        )
        grad_q = grad_q.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        grad_k = torch.bmm(
            math.sqrt(scaling) * grad_attn_probs_in.transpose(1,2),
            math.sqrt(scaling) * q.transpose(0,1)
        )
        grad_k = grad_k.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        grad_kvq_proj_output = torch.cat([grad_k, grad_v, grad_q], dim=-1)
        return grad_kvq_proj_output

    @staticmethod
    def forward(
        ctx,
        input,
        kvq_proj_weight,
        out_proj_weight,
        fc1_weight,
        fc2_weight,
        head_dim,
        recompute_fc1,
    ):
        global fused_layer_norm_cuda
        ctx.recompute_fc1 = recompute_fc1

        input = input.contiguous()

        # Take out residual connection for self attention
        residual = input
        dtype = input.dtype

        # Apply layer norm on (seq_len // #tp_size, bsz, embed_dim) tensor
        ctx.layer_norm_normalized_shape = torch.Size((input.size(-1),))
        ctx.eps = 1e-5

        # # Self attention layer norm
        mha_layer_norm_output, _, _ = fused_layer_norm_cuda.forward(input, ctx.layer_norm_normalized_shape, ctx.eps)

        # all gather output across first dim, i.e. seq_len dim for kvq_proj
        mha_layer_norm_output = _gather_along_first_dim(mha_layer_norm_output, cached_buffer_name='mpu')

        # apply kvq, output is (seq_len, bsz, 3 * embed_dim // #tp_size)
        kvq_out = torch.matmul(mha_layer_norm_output, kvq_proj_weight.t())
        # the order here doesn't matter as much as long its consistent sice initialization is same.
        # just matching the ordewr of metaseq MHA.
        k, v, q = split_tensor_along_last_dim(kvq_out, 3, contiguous_split_chunks=True)
        seq_len, bsz, embed_dim_per_partition = q.size()
        q  = q.view(seq_len, -1, head_dim)
        k  = k.view(seq_len, -1, head_dim)
        v  = v.view(seq_len, -1, head_dim).transpose(0, 1)

        attn, _ = TransformerBlockAutograd.forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype)

        out_proj_out = torch.matmul(attn, out_proj_weight.t())
        out_proj_out = _reduce_scatter_along_first_dim(out_proj_out)

        # out_proj_out = out_proj_out + residual
        out_proj_out = out_proj_out + residual

        # Take out residual connection for FFN
        residual = out_proj_out
        # No need to save mean and invvar cause we redo layernorm in backward
        ffn_layer_norm_output, _, _ = fused_layer_norm_cuda.forward(out_proj_out, ctx.layer_norm_normalized_shape, ctx.eps)

        # all gather output across first dim, i.e. seq_len dim
        ffn_layer_norm_output = _gather_along_first_dim(ffn_layer_norm_output, cached_buffer_name='mpu')

        # apply fc1, output is (seq_len, bsz, 4 * embed_dim // #tp_size)
        fc1_out = torch.matmul(ffn_layer_norm_output, fc1_weight.t())
        # apply gelu
        gelu_out = gelu(fc1_out)
        # apply fc2, output (seq_len, bsz, embed_dim) but needs to be
        # summed across tp for real output
        fc2_out = torch.matmul(gelu_out, fc2_weight.t())

        if ctx.recompute_fc1:
            fc1_out = None
        ctx.save_for_backward(
            input,
            q,
            k,
            v,
            out_proj_out,
            kvq_proj_weight,
            out_proj_weight,
            fc1_out,
            fc1_weight,
            fc2_weight,
        )
        ctx.bsz, ctx.seq_len, ctx.head_dim, ctx.embed_dim_per_partition = bsz, seq_len, head_dim, embed_dim_per_partition

        # apply scatter gather,
        # input: (seq_len, bsz, embed_dim)
        # output: (seq_len // #tp_size, bsz, embed_dim) (and embed_dim is summed across gpus)
        fc2_out_post_scatter_gather = _reduce_scatter_along_first_dim(fc2_out)
        final_out = fc2_out_post_scatter_gather + residual
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        global fused_layer_norm_cuda
        input, q, k, v, out_proj_out, kvq_proj_weight, out_proj_weight, fc1_out, fc1_weight, fc2_weight = ctx.saved_tensors
        bsz, seq_len, head_dim, embed_dim_per_partition = ctx.bsz, ctx.seq_len, ctx.head_dim, ctx.embed_dim_per_partition
        dtype = grad_output.dtype

        residual_grad = grad_output

        # gatther gradients async,
        # and we can overlap this with any recomptation.
        grad_output, handle = _gather_along_first_dim(grad_output, async_op=True)

        # Both of these operations are just recomputed from forward to save activation memory.
        ffn_layer_norm_output, ffn_layer_norm_mean, ffn_layer_norm_invvar = fused_layer_norm_cuda.forward(out_proj_out, ctx.layer_norm_normalized_shape, ctx.eps)
        # recompute gelu output for calculating fc2 weight gradient
        # note, remember "gelu_out = fc2_in"
        if not ctx.recompute_fc1:
            assert fc1_out is not None
            gelu_out = gelu(fc1_out)

        # Now wait for reduce scatter
        handle.wait()

        ffn_layer_norm_output, handle = _gather_along_first_dim(ffn_layer_norm_output, async_op=True, cached_buffer_name='mpu')

        grad_fc2_input = grad_output.matmul(fc2_weight)

        if ctx.recompute_fc1:
            handle.wait()
            assert fc1_out is None
            fc1_out = torch.matmul(ffn_layer_norm_output, fc1_weight.t())
            gelu_out = gelu(fc1_out)

        # calculate gelu backward
        grad_gelu_input = gelu_back(grad_fc2_input, fc1_out)

        # Reshape matrix and calculate gradient with respect to fc2 weight
        grad_output = TransformerBlockAutograd._collapse_first_dimensions(
            grad_output
        )
        gelu_out = TransformerBlockAutograd._collapse_first_dimensions(gelu_out)
        grad_fc2_weight = grad_output.t().matmul(gelu_out)

        grad_fc1_input = grad_gelu_input.matmul(fc1_weight)
        handle.wait()

        grad_gelu_input = TransformerBlockAutograd._collapse_first_dimensions(grad_gelu_input)
        ffn_layer_norm_output = TransformerBlockAutograd._collapse_first_dimensions(ffn_layer_norm_output)

        grad_fc1_input, handle = _reduce_scatter_along_first_dim(grad_fc1_input, async_op=True)

        grad_fc1_weight = grad_gelu_input.t().matmul(ffn_layer_norm_output)

        handle.wait()

        grad_attention_output = fused_layer_norm_cuda.backward(
            grad_fc1_input.contiguous(),
            ffn_layer_norm_mean,
            ffn_layer_norm_invvar,
            out_proj_out,
            ctx.layer_norm_normalized_shape,
            ctx.eps,
        )
        grad_attention_output = grad_attention_output + residual_grad

        residual_grad = grad_attention_output

        grad_attention_output, handle = _gather_along_first_dim(
            grad_attention_output,
            async_op=True,
        )

        # recalculate attention
        attn, attn_probs = TransformerBlockAutograd.forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype)

        handle.wait()

        grad_out_proj_input = grad_attention_output.matmul(out_proj_weight)
        grad_attention_output = TransformerBlockAutograd._collapse_first_dimensions(
            grad_attention_output
        )
        attn = TransformerBlockAutograd._collapse_first_dimensions(attn)
        grad_out_proj_weight = grad_attention_output.t().matmul(attn)

        grad_kvq_proj_output = TransformerBlockAutograd.backward_mha(
            grad_out_proj_input,
            q,
            k,
            v,
            attn_probs,
            seq_len,
            bsz,
            head_dim
        )

        mha_layer_norm_output, mha_layer_norm_mean, mha_layer_norm_invvar = fused_layer_norm_cuda.forward(input, ctx.layer_norm_normalized_shape, ctx.eps)
        mha_layer_norm_output, handle = _gather_along_first_dim(
            mha_layer_norm_output,
            async_op=True,
            cached_buffer_name='mpu',
        )
        grad_input = grad_kvq_proj_output.matmul(kvq_proj_weight)
        handle.wait()

        grad_input, handle = _reduce_scatter_along_first_dim(
            grad_input,
            async_op=True
        )
        mha_layer_norm_output = TransformerBlockAutograd._collapse_first_dimensions(mha_layer_norm_output)
        grad_kvq_proj_output = TransformerBlockAutograd._collapse_first_dimensions(grad_kvq_proj_output)
        grad_kvq_weight = grad_kvq_proj_output.t().matmul(mha_layer_norm_output)
        handle.wait()

        grad_input = fused_layer_norm_cuda.backward(
            grad_input.contiguous(),
            mha_layer_norm_mean,
            mha_layer_norm_invvar,
            input,
            ctx.layer_norm_normalized_shape,
            ctx.eps,
        )
        grad_input = grad_input + residual_grad
        return grad_input, grad_kvq_weight, grad_out_proj_weight, grad_fc1_weight, grad_fc2_weight, None, None

    @staticmethod
    def _collapse_first_dimensions(tensor):
        return tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
        )


class ParallelTransformerBlockAutograd(TransformerBlockAutograd):
    """
    This is custom FFN autograd function hardcoded for:
    bias: false,
    layernorm affine: false, ln eps: 1e-5
    sequence_parallel: true,
    activation: gelu,
    gelu, layernorm: always recomputed i.e. no activation memory for these
    """
    @staticmethod
    def forward(
        ctx,
        input,
        kvq_fc1_proj_weight,
        out_proj_weight,
        fc2_weight,
        head_dim,
        recompute_kvq_fc1
    ):
        global fused_layer_norm_cuda
        ctx.recompute_kvq_fc1 = recompute_kvq_fc1

        input = input.contiguous()

        # Take out residual connection for self attention
        residual = input
        dtype = input.dtype

        # Apply layer norm on (seq_len // #tp_size, bsz, embed_dim) tensor
        ctx.layer_norm_normalized_shape = torch.Size((input.size(-1),))
        ctx.eps = 1e-5

        # # Self attention layer norm
        layer_norm_output, _, _ = fused_layer_norm_cuda.forward(input, ctx.layer_norm_normalized_shape, ctx.eps)

        # all gather output across first dim, i.e. seq_len dim for kvq_proj
        layer_norm_output = _gather_along_first_dim(layer_norm_output, cached_buffer_name='mpu')

        # apply kvq, output is (seq_len, bsz, 3 * embed_dim // #tp_size)
        kvq_fc1_output = torch.matmul(layer_norm_output, kvq_fc1_proj_weight.t())
        # the order here doesn't matter as much as long its consistent sice initialization is same.
        # just matching the ordewr of metaseq MHA.'

        embed_dim_per_partition = kvq_fc1_output.size(kvq_fc1_output.dim() - 1) // 7
        split_sizes = [embed_dim_per_partition] * 3 + [4 * embed_dim_per_partition]
        k, v, q, fc1_out = split_tensor_by_given_split_sizes(kvq_fc1_output, split_sizes, contiguous_split_chunks=True)
        seq_len, bsz, embed_dim_per_partition = q.size()
        q  = q.view(seq_len, -1, head_dim)
        k  = k.view(seq_len, -1, head_dim)
        v  = v.view(seq_len, -1, head_dim).transpose(0, 1)

        attn, _ = TransformerBlockAutograd.forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype)

        out_proj_out = torch.matmul(attn, out_proj_weight.t())

        # apply gelu
        gelu_out = gelu(fc1_out)
        # apply fc2, output (seq_len, bsz, embed_dim) but needs to be
        # summed across tp for real output
        fc2_out = torch.matmul(gelu_out, fc2_weight.t())

        final_out = fc2_out + out_proj_out

        # apply scatter gather,
        # input: (seq_len, bsz, embed_dim)
        # output: (seq_len // #tp_size, bsz, embed_dim) (and embed_dim is summed across gpus)
        final_out = _reduce_scatter_along_first_dim(final_out)

        final_out = final_out + residual

        if ctx.recompute_kvq_fc1:
            q, k, v, fc1_out = None, None, None, None

        ctx.save_for_backward(
            input,
            q,
            k,
            v,
            fc1_out,
            kvq_fc1_proj_weight,
            out_proj_weight,
            fc2_weight,
        )
        ctx.bsz, ctx.seq_len, ctx.head_dim, ctx.embed_dim_per_partition = bsz, seq_len, head_dim, embed_dim_per_partition
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        global fused_layer_norm_cuda
        input, q, k, v, fc1_out, kvq_fc1_proj_weight, out_proj_weight, fc2_weight = ctx.saved_tensors
        bsz, seq_len, head_dim, embed_dim_per_partition = ctx.bsz, ctx.seq_len, ctx.head_dim, ctx.embed_dim_per_partition
        dtype = grad_output.dtype

        residual_grad = grad_output

        # gatther gradients async,
        # and we can overlap this with any recomptation.
        grad_output, handle = _gather_along_first_dim(grad_output, async_op=True)

        # Both of these operations are just recomputed from forward to save activation memory.
        layer_norm_output, layer_norm_mean, layer_norm_invvar = fused_layer_norm_cuda.forward(input, ctx.layer_norm_normalized_shape, ctx.eps)
        # recompute gelu output for calculating fc2 weight gradient
        # note, remember "gelu_out = fc2_in"
        if not ctx.recompute_kvq_fc1:
            assert fc1_out is not None
            gelu_out = gelu(fc1_out)

            # recalculate attention
            attn, attn_probs = TransformerBlockAutograd.forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype)

        # Now wait for reduce scatter
        handle.wait()

        layer_norm_output, handle = _gather_along_first_dim(layer_norm_output, async_op=True, cached_buffer_name='mpu')

        grad_fc2_input = grad_output.matmul(fc2_weight)

        if ctx.recompute_kvq_fc1:
            handle.wait()
            assert fc1_out is None

            kvq_fc1_output = torch.matmul(layer_norm_output, kvq_fc1_proj_weight.t())
            embed_dim_per_partition = kvq_fc1_output.size(kvq_fc1_output.dim() - 1) // 7
            split_sizes = [embed_dim_per_partition] * 3 + [4 * embed_dim_per_partition]
            k, v, q, fc1_out = split_tensor_by_given_split_sizes(kvq_fc1_output, split_sizes, contiguous_split_chunks=True)
            seq_len, bsz, embed_dim_per_partition = q.size()
            q  = q.view(seq_len, -1, head_dim)
            k  = k.view(seq_len, -1, head_dim)
            v  = v.view(seq_len, -1, head_dim).transpose(0, 1)

            gelu_out = gelu(fc1_out)
            attn, attn_probs = TransformerBlockAutograd.forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype)

        # calculate gelu backward
        grad_gelu_input = gelu_back(grad_fc2_input, fc1_out)

        # calculate gradient of outproj
        grad_out_proj_input = grad_output.matmul(out_proj_weight)

        # Reshape matrix and calculate gradient with respect to fc2 weight
        grad_output = TransformerBlockAutograd._collapse_first_dimensions(grad_output)
        gelu_out = TransformerBlockAutograd._collapse_first_dimensions(gelu_out)
        grad_fc2_weight = grad_output.t().matmul(gelu_out)

        attn = TransformerBlockAutograd._collapse_first_dimensions(attn)
        grad_out_proj_weight = grad_output.t().matmul(attn)


        grad_kvq_proj_output = TransformerBlockAutograd.backward_mha(
            grad_out_proj_input,
            q,
            k,
            v,
            attn_probs,
            seq_len,
            bsz,
            head_dim
        )

        grad_kvq_fc1_proj_output = torch.cat([grad_kvq_proj_output, grad_gelu_input], dim=-1)
        grad_input = grad_kvq_fc1_proj_output.matmul(kvq_fc1_proj_weight)

        handle.wait()

        grad_input, handle = _reduce_scatter_along_first_dim(
            grad_input,
            async_op=True
        )
        layer_norm_output = TransformerBlockAutograd._collapse_first_dimensions(layer_norm_output)
        grad_kvq_fc1_proj_output = TransformerBlockAutograd._collapse_first_dimensions(grad_kvq_fc1_proj_output)
        grad_kvq_fc1_weight = grad_kvq_fc1_proj_output.t().matmul(layer_norm_output)
        handle.wait()

        grad_input = fused_layer_norm_cuda.backward(
            grad_input.contiguous(),
            layer_norm_mean,
            layer_norm_invvar,
            input,
            ctx.layer_norm_normalized_shape,
            ctx.eps,
        )
        grad_input = grad_input + residual_grad
        return grad_input, grad_kvq_fc1_weight, grad_out_proj_weight, grad_fc2_weight, None, None

    @staticmethod
    def _collapse_first_dimensions(tensor):
        return tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
        )