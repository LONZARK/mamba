# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, init_value):
        super(SubNetwork, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        # Initialize weights to a constant value

        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_a.weight, dt_init_std_a)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std_a, dt_init_std_a)
        else:
            raise NotImplementedError


class Mamba(nn.Module):
    def __init__(
        self, 
        d_model_a,      # Model dimension = 64 for Mamba, the number of dimensions in the vector space
        d_model_v,
        d_state=16,   # Our setting is on sequences of length 4096, with a vocab size of 16 possible tokens (including the white “noise” token from Figure 2) and requiring models to memorize 16 “data” tokens
        d_conv=4,     # kernel_size, defines the size of the window to be convolved with the input
        expand=2,     # We always fix to E = 2 in our experiments and use two stacks of the block to match the 12D^2 parameters of a Transformer’s interleaved MHA (multi-head attention) and MLP blocks
        dt_rank="auto",  # math.ceil(self.d_model / 16) = 64/16 = 4
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model_a = d_model_a
        self.d_model_v = d_model_v
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner_a = int(self.expand * self.d_model_a) # 2 * 64 = 128
        self.d_inner_v = int(self.expand * self.d_model_v) # 2 * 64 = 128

        self.dt_rank_a = math.ceil(self.d_model_a / 16) if dt_rank == "auto" else dt_rank
        self.dt_rank_v = math.ceil(self.d_model_v / 16) if dt_rank == "auto" else dt_rank

        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Initialize a linear transformation layer with input features (self.d_model), 
        # output features (self.d_inner * 2), and optionally, a bias. Additional arguments 
        # are passed through **factory_kwargs.
        # Input - self.d_model = 64
        # Output - self.d_inner * 2 = 128 * 2 = 256
        self.in_proj_a = nn.Linear(self.d_model_a, self.d_inner_a * 2, bias=bias, **factory_kwargs)
        self.in_proj_v = nn.Linear(self.d_model_v, self.d_inner_v * 2, bias=bias, **factory_kwargs)


        self.conv1d_a = nn.Conv1d(
            in_channels=self.d_inner_a,  # the number of channels in the input data
            out_channels=self.d_inner_a,  # the number of channels produced by the convolution
            bias=conv_bias,  # whether a learnable bias is added to the output of the convolution
            kernel_size=d_conv,  # defines the size of the window to be convolved with the input
            groups=self.d_inner_a, # the convolution operation is performed separately for each input channel.
            padding=d_conv - 1, # padding=d_conv - 1 ensures that the output length is the same as the input length when the stride is 1
            **factory_kwargs,
        )

        self.conv1d_v = nn.Conv1d(
            in_channels=self.d_inner_v,  # the number of channels in the input data
            out_channels=self.d_inner_v,  # the number of channels produced by the convolution
            bias=conv_bias,  # whether a learnable bias is added to the output of the convolution
            kernel_size=d_conv,  # defines the size of the window to be convolved with the input
            groups=self.d_inner_v, # the convolution operation is performed separately for each input channel.
            padding=d_conv - 1, # padding=d_conv - 1 ensures that the output length is the same as the input length when the stride is 1
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj_a = nn.Linear(
            self.d_inner_a, self.dt_rank_a + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_v = nn.Linear(
            self.d_inner_v, self.dt_rank_v + self.d_state * 2, bias=False, **factory_kwargs
        )

        # print('self.x_proj_a', self.x_proj_a)
        # print('self.x_proj_v', self.x_proj_v)
        # print('-------------------------------------')
        self.dt_proj_a = nn.Linear(self.dt_rank_a, self.d_inner_a, bias=True, **factory_kwargs)
        self.dt_proj_v = nn.Linear(self.dt_rank_v, self.d_inner_v, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std_a = self.dt_rank_a**-0.5 * dt_scale
        dt_init_std_v = self.dt_rank_v**-0.5 * dt_scale


        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_a.weight, dt_init_std_a)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_a.weight, -dt_init_std_a, dt_init_std_a)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt_a = torch.exp(
            torch.rand(self.d_inner_a, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt_a = dt_a + torch.log(-torch.expm1(-dt_a))
        with torch.no_grad():
            self.dt_proj_a.bias.copy_(inv_dt_a)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_a.bias._no_reinit = True

        # S4D real initialization
        A_a = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner_a,
        ).contiguous()  # shape [128, 16]
        A_log_a = torch.log(A_a)  # Keep A_log in fp32
        self.A_log_a = nn.Parameter(A_log_a)  # shape [128, 16]
        self.A_log_a._no_weight_decay = True

        # D "skip" parameter
        self.D_a = nn.Parameter(torch.ones(self.d_inner_a, device=device))  # Keep in fp32
        self.D_a._no_weight_decay = True

        self.out_proj_a = nn.Linear(self.d_inner_a, self.d_model_a, bias=bias, **factory_kwargs)


        '''
        ------------------------------------------------------------------
        '''

        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_v.weight, dt_init_std_v)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_v.weight, -dt_init_std_v, dt_init_std_v)
        else:
            raise NotImplementedErro

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt_v = torch.exp(
            torch.rand(self.d_inner_v, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt_v = dt_v + torch.log(-torch.expm1(-dt_v))
        with torch.no_grad():
            self.dt_proj_v.bias.copy_(inv_dt_v)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj_v.bias._no_reinit = True

        # S4D real initialization
        A_v = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner_v,
        ).contiguous()  # shape [128, 16]
        A_log_v = torch.log(A_v)  # Keep A_log in fp32
        self.A_log_v = nn.Parameter(A_log_v)  # shape [128, 16]
        self.A_log_v._no_weight_decay = True

        # D "skip" parameter
        self.D_v = nn.Parameter(torch.ones(self.d_inner_v, device=device))  # Keep in fp32
        self.D_v._no_weight_decay = True

        self.out_proj_v = nn.Linear(self.d_inner_v, self.d_model_v, bias=bias, **factory_kwargs)



    def forward(self, hidden_states_a, hidden_states_v, inference_params=None, src_mask=None, src_key_padding_mask=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch_a, seqlen_a, dim_a = hidden_states_a.shape
        batch_v, seqlen_v, dim_v = hidden_states_a.shape

        conv_state_a, ssm_state_a = None, None
        conv_state_v, ssm_state_v = None, None

        if inference_params is not None:
            
            conv_state_a, ssm_state_a = self._get_states_from_cache(inference_params, batch_a)
            if inference_params.seqlen_offset > 0:
                out_a, _, _ = self.step(hidden_states_a, hidden_states_v, conv_state_a, ssm_state_a)
                out_v, _, _ = self.step(hidden_states_v, hidden_states_a, conv_state_v, ssm_state_v)
                return out_a

        # We do matmul and transpose BLH -> HBL at the same time
        # batch, seqlen, dim -> dim, batch, seqlen
        # print('hidden_states_a.shape', hidden_states_a.shape)
        xz_a = rearrange(
            self.in_proj_a.weight @ rearrange(hidden_states_a, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_a,
        )
        # print('xz_a.shape', xz_a.shape)

        # print('hidden_states_v.shape', hidden_states_v.shape)

        xz_v = rearrange(
            self.in_proj_v.weight @ rearrange(hidden_states_v, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_v,
        )

        # print('xz_v.shape', xz_v.shape)
        # print('hidden_states_a.shape', hidden_states_a.shape)
        # print('hidden_states_v.shape', hidden_states_v.shape)
        
        if self.in_proj_a.bias is not None:
            # print('self.in_proj.bias is not None')
            xz_a = xz_a + rearrange(self.in_proj_a.bias.to(dtype=xz_a.dtype), "d -> d 1")

        if self.in_proj_v.bias is not None:
            # print('self.in_proj.bias is not None')
            xz_v = xz_v + rearrange(self.in_proj_v.bias.to(dtype=xz_v.dtype), "d -> d 1")

        A_a = -torch.exp(self.A_log_a.float())  # (d_inner, d_state), shape [128, 16]
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            # print(' 11111111111111111111111 ')
            out = mamba_inner_fn(
                xz_a,                     # dim, batch, seqlen
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A_a,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # print(' 22222222222222222222 ')
            x_a, z_a = xz_a.chunk(2, dim=1)
            # print('xz_a.shape', xz_a.shape)
            # print('x_a.shape', x_a.shape)
            # print('z_a.shape', z_a.shape)
            # Compute short convolution
            if conv_state_a is not None:
                # print('conv_state_a is not None')
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state_a.copy_(F.pad(x_a, (self.d_conv - x_a.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                # print('causal_conv1d_fn is None')
                x_a = self.act(self.conv1d_a(x_a)[..., :seqlen_a])  # self.act = nn.SiLU(), self.conv1d = nn.Conv1d()
                # print('x_a.shape', x_a.shape)

            else:
                assert self.activation in ["silu", "swish"]
                x_a = causal_conv1d_fn(
                    x=x_a,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )


        '''
        ---------------------------------------
        '''

        A_v = -torch.exp(self.A_log_v.float())  # (d_inner, d_state), shape [128, 16]
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            # print(' 11111111111111111111111 ')
            out = mamba_inner_fn(
                xz_v,                     # dim, batch, seqlen
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A_v,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # print(' 22222222222222222222 ')
            x_v, z_v = xz_v.chunk(2, dim=1)
            # print('xz_v.shape', xz_v.shape)
            # print('x_v.shape', x_v.shape)
            # print('z_v.shape', z_v.shape)

            # Compute short convolution
            if conv_state_v is not None:
                # print('conv_state_v is not None')
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state_v.copy_(F.pad(x_v, (self.d_conv - x_v.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                # print('causal_conv1d_fn is None')
                x_v = self.act(self.conv1d_a(x_v)[..., :seqlen_v])  # self.act = nn.SiLU(), self.conv1d = nn.Conv1d()
                # print('x_v.shape', x_v.shape)

            else:
                assert self.activation in ["silu", "swish"]
                x_v = causal_conv1d_fn(
                    x=x_v,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )


            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl_a = self.x_proj_a(rearrange(x_a, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            # x_dbl_a.shape torch.Size([1280, 33])
            # print('x_dbl_a.shape', x_dbl_a.shape)

            # linear_layer = nn.Linear(in_features=33, out_features=33, bias=True)
            # linear_layer = linear_layer.to('cuda')
            # x_audio_transformed = linear_layer(x_dbl_a)


            dt_a, B_a, C_a = torch.split(x_dbl_a, [self.dt_rank_a, self.d_state, self.d_state], dim=-1)

            dt_a = self.dt_proj_a.weight @ dt_a.t()
            dt_a = rearrange(dt_a, "d (b l) -> b d l", l=seqlen_a)
            B_a = rearrange(B_a, "(b l) dstate -> b dstate l", l=seqlen_a).contiguous()
            C_a = rearrange(C_a, "(b l) dstate -> b dstate l", l=seqlen_a).contiguous()

            # print('dt_a.shape', dt_a.shape)
            # print('B_a.shape', B_a.shape)
            # print('C_a.shape', C_a.shape)

            assert self.activation in ["silu", "swish"]
            y_a = selective_scan_fn(
                x_a,
                dt_a,
                A_a,
                B_a,
                C_a,
                self.D_a.float(),
                z=z_a,
                delta_bias=self.dt_proj_a.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state_a is not None,
            )
            if ssm_state_a is not None:
                y_a, last_state = y_a
                ssm_state_a.copy_(last_state)

            y_a = rearrange(y_a, "b d l -> b l d")
            out_a = self.out_proj_a(y_a)

            # print('y_a.shape', y_a.shape)
            # print('out_a.shape', out_a.shape)


            '''
            -----------------------------------------------------
            '''
            x_dbl_v = self.x_proj_v(rearrange(x_v, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            # x_dbl_v.shape torch.Size([163840, 33])
            # print('x_dbl_v.shape', x_dbl_v.shape) 

            dt_v, B_v, C_v = torch.split(x_dbl_v, [self.dt_rank_v, self.d_state, self.d_state], dim=-1)

            dt_v = self.dt_proj_v.weight @ dt_v.t()
            dt_v = rearrange(dt_v, "d (b l) -> b d l", l=seqlen_v)
            B_v = rearrange(B_v, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()
            C_v = rearrange(C_v, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()

            # print('dt_v.shape', dt_v.shape)
            # print('B_v.shape', B_v.shape)
            # print('C_v.shape', C_v.shape)


            assert self.activation in ["silu", "swish"]
            y_v = selective_scan_fn(
                x_v,
                dt_v,
                A_v,
                B_v,
                C_v,
                self.D_v.float(),
                z=z_v,
                delta_bias=self.dt_proj_v.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state_v is not None,
            )
            if ssm_state_v is not None:
                y_v, last_state = y_v
                ssm_state_v.copy_(last_state)

            y_v = rearrange(y_v, "b d l -> b l d")
            out_v = self.out_proj_v(y_v)


            '''
            ----------------------------------------------------
            '''
            x_dbl_a = self.x_proj_a(rearrange(x_a, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            x_dbl_v = self.x_proj_v(rearrange(x_v, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

            # ratio = x_dbl_v.shape[0] // x_dbl_a.shape[0]  # 计算重复的次数
            # x_dbl_a = x_dbl_a.repeat(ratio, 1)  # 在第一维重复
            # print('x_dbl_a.shape', x_dbl_a.shape)
            # print('x_dbl_v.shape', x_dbl_v.shape)
            
            x_dbl_av = x_dbl_a + x_dbl_v
            # print('x_dbl_av.shape', x_dbl_av.shape)

            dt_av, B_av, C_av = torch.split(x_dbl_av, [self.dt_rank_v, self.d_state, self.d_state], dim=-1)

            dt_av = self.dt_proj_v.weight @ dt_av.t()
            dt_av = rearrange(dt_av, "d (b l) -> b d l", l=seqlen_a)
            B_av = rearrange(B_av, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()
            C_av = rearrange(C_av, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()


            assert self.activation in ["silu", "swish"]
            y_av = selective_scan_fn(
                x_v,
                dt_av,
                A_a,
                B_av,
                C_av,
                self.D_v.float(),
                z=z_v,
                delta_bias=self.dt_proj_v.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state_v is not None,
            )
            if ssm_state_v is not None:
                y_av, last_state = y_av
                ssm_state_v.copy_(last_state)

            y_av = rearrange(y_av, "b d l -> b l d")
            out_av = self.out_proj_v(y_av)


            '''
            -----------------------------------------------------------------
            '''
            x_dbl_a = self.x_proj_a(rearrange(x_a, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
            x_dbl_v = self.x_proj_v(rearrange(x_v, "b d l -> (b l) d"))  # (bl d)  #         self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)

            print('x_dbl_v.shape', x_a.shape)
            a = rearrange(x_a, "b d l -> (b l) d")  # batch, dim, seqlen --> batch*seqlen, dim
            print('a.shape', a.shape)
            print('x_dbl_v.shape', x_dbl_a.shape)

            # assert x_dbl_v.shape[0] % x_dbl_a.shape[0] == 0
            # x_dbl_v_reshaped = x_dbl_v.view(x_dbl_a.shape[0], -1, x_dbl_v.shape[1])
            # x_dbl_v = x_dbl_v_reshaped.mean(dim=1)
            # print('x_dbl_v.shape[0]', x_dbl_v.shape)

            x_dbl_va = x_dbl_a + x_dbl_v

            dt_va, B_va, C_va = torch.split(x_dbl_va, [self.dt_rank_v, self.d_state, self.d_state], dim=-1)

            print('dt_va.shape', dt_va.shape)
            print('B_va.shape', B_va.shape)
            print('C_va.shape', C_va.shape)

            dt_va = self.dt_proj_v.weight @ dt_va.t()
            print('dt_va.shape', dt_va.shape)

            dt_va = rearrange(dt_va, "d (b l) -> b d l", l=seqlen_a)
            B_va = rearrange(B_va, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()
            C_va = rearrange(C_va, "(b l) dstate -> b dstate l", l=seqlen_v).contiguous()

            print('dt_va.shape', dt_va.shape)
            print('B_va.shape', B_va.shape)
            print('C_va.shape', C_va.shape)

            assert self.activation in ["silu", "swish"]
            y_va = selective_scan_fn(
                x_a,
                dt_va,
                A_a,
                B_va,
                C_va,
                self.D_a.float(),
                z=z_a,
                delta_bias=self.dt_proj_a.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state_a is not None,
            )
            if ssm_state_a is not None:
                y_va, last_state = y_va
                ssm_state_a.copy_(last_state)

            y_va = rearrange(y_va, "b d l -> b l d")
            out_va = self.out_proj_a(y_va)


            exit()
        return out_a, out_v, out_av, out_va

    def step(self, hidden_states_a, hidden_states_v, conv_state, ssm_state):
        dtype = hidden_states_a.dtype

        assert hidden_states_a.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states_a.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W) # Roll state for causality.
            conv_state[:, :, -1] = x # Update the latest state with new input `x`.
            # Apply convolution using the updated state and the convolutional weights.
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)  # Apply activation function and ensure original dtype.
        
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner), a linear transformation to the data dt by using self.dt_proj.weight
        A = -torch.exp(self.A_log_a.float())  # (d_inner, d_state)  # Prepare the `A` matrix, ensuring float32.

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))  # Ensure `dt` is positive.
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # Discretize `A`.
            dB = torch.einsum("bd,bn->bdn", dt, B)  # Calculate `B`.
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)  # Update SSM state.
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)  # Compute output `y`.
            y = y + self.D.to(dtype) * x  # Add contribution from `D` and `x`.
            y = y * self.act(z)  # (B D)   # Apply activation function to `z` and scale `y`.
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



# batch_a, length_a, dim_a = 10, 2048, 1 
# batch_v, length_v, dim_v = 80, 2048, 1

# x_a = torch.randn(batch_a, length_a, dim_a).to("cuda")
# x_v = torch.randn(batch_v, length_v, dim_v).to("cuda")

# print(x_a.shape)
# print(x_v.shape)

# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model_a=dim_a, # Model dimension d_model
#     d_model_v=dim_v,
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")

# print(model)

# exit()

# out_a, out_v, out_av, out_va = model(x_a, x_v)

# print(x_a.shape)
# print(x_v.shape)

# assert out_a.shape == x_a.shape
