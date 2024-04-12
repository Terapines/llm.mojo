# ===----------------------------------------------------------------------=== #
# Copyright (C) 2020-2024 Terapines Technology (Wuhan) Co., Ltd
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https:#llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import math
import pathlib.path as path
from memory import *
from layernorm import *

alias M_PI: Float32 = 3.14159265358979323846264338327950288
alias NUM_PARAMETER_TENSORS = 16
alias NUM_ACTIVATION_TENSORS = 23
alias GPT2_EOT = 50256
alias SEEK_END = 2

# ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes


fn encoder_forward(
    out: Pointer[Float32],
    inp: Pointer[Int32],
    wte: Pointer[Float32],
    wpe: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
) raises -> None:
    for b in range(B):
        for t in range(T):
            # seek to the output position in out[b,t,:]
            var out_bt = out + b * T * C + t * C
            # get the index of the token at inp[b, t]
            var ix = inp[b * T + t]
            # seek to the position in wte corresponding to the token
            var wte_ix = wte + ix * C
            # seek to the position in wpe corresponding to the position
            var wpe_t = wpe + t * C
            # add the two vectors and store the result in out[b,t,:]
            for i in range(C):
                out_bt[i] = wte_ix[i] + wpe_t[i]


fn encoder_backward(
    dwte: Pointer[Float32],
    dwpe: Pointer[Float32],
    dout: Pointer[Float32],
    inp: Pointer[Int32],
    B: Int,
    T: Int,
    C: Int,
) raises -> None:
    for b in range(B):
        for t in range(T):
            var dout_bt = dout + b * T * C + t * C
            var ix = inp[b * T + t]
            var dwte_ix = dwte + ix * C
            var dwpe_t = dwpe + t * C
            for i in range(C):
                var d = dout_bt[i]
                dwte_ix[i] += d
                dwpe_t[i] += d


fn layernorm_forward(
    out: Pointer[Float32],
    mean: Pointer[Float32],
    rstd: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    bias: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
) raises -> None:
    var eps = 1e-5
    for b in range(B):
        for t in range(T):
            # seek to the input position inp[b,t,:]
            var x = inp + b * T * C + t * C
            # calculate the mean
            var m: Float32 = 0.0
            for i in range(C):
                m += x[i]
            m = m / C
            # calculate the variance (without any bias correction)
            var v: Float32 = 0.0
            for i in range(C):
                var xshift = x[i] - m
                v += xshift * xshift
            v = v / C
            # calculate the rstd
            var s = 1.0 / math.sqrt(v + eps)
            # seek to the output position in out[b,t,:]
            var out_bt = out + b * T * C + t * C
            for i in range(C):
                var n = (s * (x[i] - m))  # normalized output
                var o = n * weight[i] + bias[i]  # scale and shift it
                out_bt[i] = o  # write
            # cache the mean and rstd for the backward pass later
            mean[b * T + t] = m
            rstd[b * T + t] = s


fn layernorm_backward(
    dinp: Pointer[Float32],
    dweight: Pointer[Float32],
    dbias: Pointer[Float32],
    dout: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    mean: Pointer[Float32],
    rstd: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
) raises -> None:
    for b in range(B):
        for t in range(T):
            var dout_bt = dout + b * T * C + t * C
            var inp_bt = inp + b * T * C + t * C
            var dinp_bt = dinp + b * T * C + t * C
            var mean_bt = mean[b * T + t]
            var rstd_bt = rstd[b * T + t]

            # first: two reduce operations
            var dnorm_mean: Float32 = 0.0
            var dnorm_norm_mean: Float32 = 0.0
            for i in range(C):
                var norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
                var dnorm_i = weight[i] * dout_bt[i]
                dnorm_mean += dnorm_i
                dnorm_norm_mean += dnorm_i * norm_bti
            dnorm_mean = dnorm_mean / C
            dnorm_norm_mean = dnorm_norm_mean / C

            # now iterate again and accumulate all the gradients
            for i in range(C):
                var norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
                var dnorm_i = weight[i] * dout_bt[i]
                # gradient contribution to bias
                dbias[i] += dout_bt[i]
                # gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i]
                # gradient contribution to input
                var dval: Float32 = 0.0
                dval += dnorm_i  # term 1
                dval -= dnorm_mean  # term 2
                dval -= norm_bti * dnorm_norm_mean  # term 3
                dval *= rstd_bt  # final scale
                dinp_bt[i] += dval


fn matmul_forward(
    out: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    bias: Pointer[Float32],
    B: Int32,
    T: Int32,
    C: Int32,
    OC: Int32,
) raises -> None:
    # most of the running time is spent here and in matmul_backward
    # OC is short for "output channels"
    # inp is (B,T,C), weight is (OC, C), bias is (OC)
    # out will be (B,T,OC)
    # pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            var out_bt: Pointer[Float32] = out + b * T * OC + t * OC
            var inp_bt: Pointer[Float32] = inp + b * T * C + t * C
            for o in range(OC):
                var val: Float32 = bias[o] if bias else 0.0
                var wrow: Pointer[Float32] = weight + o * C
                for i in range(C):
                    val += inp_bt[i] * wrow[i]
                out_bt[o] = val


fn matmul_backward(
    dinp: Pointer[Float32],
    dweight: Pointer[Float32],
    dbias: Pointer[Float32],
    dout: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
    OC: Int,
) raises -> None:
    # most of the running time is spent here and in matmul_forward
    # this backward could be done in a single "round" of loops
    # but that doesn't afford an efficient parallelization strategy

    # backward into inp first, parallelize over B,T
    # pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            var dout_bt = dout + b * T * OC + t * OC
            var dinp_bt = dinp + b * T * C + t * C
            for o in range(OC):
                var wrow = weight + o * C
                var d = dout_bt[o]
                for i in range(C):
                    dinp_bt[i] += wrow[i] * d
    # backward into weight/bias, parallelize over output channels OC
    # pragma omp parallel for
    for o in range(OC):
        for b in range(B):
            for t in range(T):
                var dout_bt = dout + b * T * OC + t * OC
                var inp_bt = inp + b * T * C + t * C
                var dwrow = dweight + o * C
                var d = dout_bt[o]
                if dbias:
                    dbias[o] += d
                for i in range(C):
                    dwrow[i] += inp_bt[i] * d


fn attention_forward(
    out: Pointer[Float32],
    preatt: Pointer[Float32],
    att: Pointer[Float32],
    inp: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
    NH: Int,
) raises -> None:
    # input is (B, T, 3C) Q,K,V
    # preatt, att are (B, NH, T, T)
    # output is (B, T, C)
    var C3 = C * 3
    var hs = C / NH  # head size
    var scale = 1.0 / math.sqrt(hs)  # FIXME:

    # pragma omp parallel for collapse(3)
    for b in range(B):
        for t in range(T):
            for h in range(NH):
                var query_t = inp + b * T * C3 + t * C3 + h * hs
                var preatt_bth = preatt + b * NH * T * T + h * T * T + t * T
                var att_bth = att + b * NH * T * T + h * T * T + t * T

                # pass 1: calculate query dot key and maxval
                var maxval: Float32 = -10000.0  # TODO something better
                for t2 in range(t + 1):
                    var key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key

                    # (query_t) dot (key_t2)
                    var val: Float32 = 0.0
                    for i in range(hs):
                        val += query_t[i] * key_t2[i]
                    val *= scale
                    if val > maxval:
                        maxval = val
                    preatt_bth[t2] = val

                # pass 2: calculate the exp and keep track of sum
                var expsum: Float32 = 0.0
                for t2 in range(t + 1):
                    var expv = math.exp(preatt_bth[t2] - maxval)  # FIXME: expf
                    expsum += expv
                    att_bth[t2] = expv

                var expsum_inv = 0.0 if (expsum == 0.0) else (1.0 / expsum)

                # pass 3: normalize to get the softmax
                for t2 in range(T):
                    if t2 <= t:
                        att_bth[t2] *= expsum_inv
                    else:
                        # causal attention mask. not strictly necessary to set to zero here
                        # only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0
                # pass 4: accumulate weighted values into the output of attention
                var out_bth = out + b * T * C + t * C + h * hs
                for i in range(hs):
                    out_bth[i] = 0.0
                for t2 in range(t + 1):
                    var value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2  # +C*2 because it's value
                    var att_btht2 = att_bth[t2]
                    for i in range(hs):
                        out_bth[i] += att_btht2 * value_t2[i]


fn attention_backward(
    dinp: Pointer[Float32],
    dpreatt: Pointer[Float32],
    datt: Pointer[Float32],
    dout: Pointer[Float32],
    inp: Pointer[Float32],
    att: Pointer[Float32],
    B: Int,
    T: Int,
    C: Int,
    NH: Int,
) raises -> None:
    # inp/dinp are (B, T, 3C) Q,K,V
    # att/datt/dpreatt are (B, NH, T, T)
    # dout is (B, T, C)
    var C3 = C * 3
    var hs = C / NH  # head size
    var scale = 1.0 / math.sqrt(hs)  # FIXME: sqrtf

    for b in range(B):
        for t in range(T):
            for h in range(NH):
                var att_bth = att + b * NH * T * T + h * T * T + t * T
                var datt_bth = datt + b * NH * T * T + h * T * T + t * T
                var dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T
                var dquery_t = dinp + b * T * C3 + t * C3 + h * hs
                var query_t = inp + b * T * C3 + t * C3 + h * hs

                # backward pass 4, through the value accumulation
                var dout_bth = dout + b * T * C + t * C + h * hs
                for t2 in range(t + 1):
                    var value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2  # +C*2 because it's value
                    var dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2
                    for i in range(hs):
                        # in the forward pass this was:
                        # out_bth[i] += att_bth[t2] * value_t2[i]
                        # so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i]
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i]

                # backward pass 2 & 3, the softmax
                # note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in range(t + 1):
                    for t3 in range(t + 1):
                        var indicator: Float32 = 1.0 if t2 == t3 else 0.0
                        var local_derivative = att_bth[t2] * (indicator - att_bth[t3])
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2]

                # backward pass 1, the query @ key matmul
                for t2 in range(t + 1):
                    var key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key
                    var dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C  # +C because it's key
                    for i in range(hs):
                        # in the forward pass this was:
                        # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
                        # so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale


fn gelu_forward(out: Pointer[Float32], inp: Pointer[Float32], N: Int) raises -> None:
    var s = math.sqrt(2.0 / M_PI)  # FIXME: sqrtf
    for i in range(N):
        var x = inp[i]
        var cube = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1.0 + math.tanh(s * (x + cube)))  # FIXME:tanhf


fn gelu_backward(
    dinp: Pointer[Float32],
    inp: Pointer[Float32],
    dout: Pointer[Float32],
    N: Int32,
) raises -> None:
    var s = math.sqrt(2.0 / M_PI)  # FIXME: sqrtf
    for i in range(N):
        var x = inp[i]
        var cube = 0.044715 * x * x * x
        var tanh_arg = s * (x + cube)
        var tanh_out = math.tanh(tanh_arg)  # FIXME: tanhf
        var coshf_out = math.cosh(tanh_arg)  # FIXME: coshf
        var sech_out = 1.0 / (coshf_out * coshf_out)
        var local_grad = 0.5 * (1.0 + tanh_out) + x * 0.5 * sech_out * s * (
            1.0 + 3.0 * 0.044715 * x * x
        )
        dinp[i] += local_grad * dout[i]


fn residual_forward(
    out: Pointer[Float32], inp1: Pointer[Float32], inp2: Pointer[Float32], N: Int
) raises -> None:
    for i in range(N):
        out[i] = inp1[i] + inp2[i]


fn residual_backward(
    dinp1: Pointer[Float32], dinp2: Pointer[Float32], dout: Pointer[Float32], N: Int
) raises -> None:
    for i in range(N):
        dinp1[i] += dout[i]
        dinp2[i] += dout[i]


fn softmax_forward(
    probs: Pointer[Float32], logits: Pointer[Float32], B: Int, T: Int, V: Int
) raises -> None:
    # output: probs are (B,T,V) of the probabilities
    # input: logits is (B,T,V) of the unnormalized log probabilities
    # pragma omp parallel for collapse(2)
    for b in range(B):
        for t in range(T):
            # probs <- softmax(logits)
            var logits_bt = logits + b * T * V + t * V
            var probs_bt = probs + b * T * V + t * V

            var maxval: Float32 = -10000.0  # TODO something better
            for i in range(V):
                if logits_bt[i] > maxval:
                    maxval = logits_bt[i]

            var sum: Float32 = 0.0
            for i in range(V):
                probs_bt[i] = math.exp(logits_bt[i] - maxval)  # FIXME: expf
                sum += probs_bt[i]
            for i in range(V):
                probs_bt[i] /= sum


fn crossentropy_forward(
    losses: Pointer[Float32],
    probs: Pointer[Float32],
    targets: Pointer[Int32],
    B: Int,
    T: Int,
    V: Int,
) raises -> None:
    # output: losses is (B,T) of the individual losses at each position
    # input: probs are (B,T,V) of the probabilities
    # input: targets is (B,T) of integers giving the correct index in logits
    for b in range(B):
        for t in range(T):
            # loss = -log(probs[target])
            var probs_bt = probs + b * T * V + t * V
            var ix = targets[b * T + t]
            losses[b * T + t] = -math.log(probs_bt[ix])  # FIXME: logf


fn crossentropy_softmax_backward(
    dlogits: Pointer[Float32],
    dlosses: Pointer[Float32],
    probs: Pointer[Float32],
    targets: Pointer[Int32],
    B: Int,
    T: Int,
    V: Int,
) raises -> None:
    # backwards through both softmax and crossentropy
    for b in range(B):
        for t in range(T):
            var dlogits_bt = dlogits + b * T * V + t * V
            var probs_bt = probs + b * T * V + t * V
            var dloss = dlosses[b * T + t]
            var ix: Int32 = targets[b * T + t]
            for i in range(V):
                var p = probs_bt[i]
                var indicator = 1.0 if (i == int(ix)) else 0.0
                dlogits_bt[i] += (p - indicator) * dloss


# ----------------------------------------------------------------------------
# GPT-2 model definition


# the parameters of the model
# define NUM_PARAMETER_TENSORS 16
@value
@register_passable
struct ParameterTensors:
    var wte: Pointer[Float32]  # (V, C)
    var wpe: Pointer[Float32]  # (maxT, C)
    var ln1w: Pointer[Float32]  # (L, C)
    var ln1b: Pointer[Float32]  # (L, C)
    var qkvw: Pointer[Float32]  # (L, 3*C, C)
    var qkvb: Pointer[Float32]  # (L, 3*C)
    var attprojw: Pointer[Float32]  # (L, C, C)
    var attprojb: Pointer[Float32]  # (L, C)
    var ln2w: Pointer[Float32]  # (L, C)
    var ln2b: Pointer[Float32]  # (L, C)
    var fcw: Pointer[Float32]  # (L, 4*C, C)
    var fcb: Pointer[Float32]  # (L, 4*C)
    var fcprojw: Pointer[Float32]  # (L, C, 4*C)
    var fcprojb: Pointer[Float32]  # (L, C)
    var lnfw: Pointer[Float32]  # (C)
    var lnfb: Pointer[Float32]  # (C)
    fn __init__(inout self):
        self.wte = Pointer[Float32].alloc(4)
        self.wpe = Pointer[Float32].alloc(4)
        self.ln1w = Pointer[Float32].alloc(4)
        self.ln1b = Pointer[Float32].alloc(4)
        self.qkvw = Pointer[Float32].alloc(4)
        self.qkvb = Pointer[Float32].alloc(4)
        self.attprojw = Pointer[Float32].alloc(4)
        self.attprojb = Pointer[Float32].alloc(4)
        self.ln2w = Pointer[Float32].alloc(4)
        self.ln2b = Pointer[Float32].alloc(4)
        self.fcw = Pointer[Float32].alloc(4)
        self.fcb = Pointer[Float32].alloc(4)
        self.fcprojw = Pointer[Float32].alloc(4)
        self.fcprojb= Pointer[Float32].alloc(4)
        self.lnfw = Pointer[Float32].alloc(4)
        self.lnfb= Pointer[Float32].alloc(4)

# allocate memory for the parameters and point the individual tensors to the right places
fn malloc_and_point_parameters(
    params: Pointer[ParameterTensors], param_sizes: Pointer[UInt32]
) raises -> Pointer[Float32]:
    var num_parameters: UInt32 = 0
    for i in range(NUM_PARAMETER_TENSORS):
        num_parameters += param_sizes[i]
    # malloc all parameters all at once
    var params_memory = Pointer[Float32].alloc(int(num_parameters) * sizeof[Float32]())
    # assign all the tensor
    var ptrs = List(
        Pointer.address_of(params[0].wte),
        Pointer.address_of(params[0].wpe),
        Pointer.address_of(params[0].ln1w),
        Pointer.address_of(params[0].ln1b),
        Pointer.address_of(params[0].qkvw),
        Pointer.address_of(params[0].qkvb),
        Pointer.address_of(params[0].attprojw),
        Pointer.address_of(params[0].attprojb),
        Pointer.address_of(params[0].ln2w),
        Pointer.address_of(params[0].ln2b),
        Pointer.address_of(params[0].fcw),
        Pointer.address_of(params[0].fcb),
        Pointer.address_of(params[0].fcprojw),
        Pointer.address_of(params[0].fcprojb),
        Pointer.address_of(params[0].lnfw),
        Pointer.address_of(params[0].lnfb),
    )
    var params_memory_iterator = params_memory
    for i in range(NUM_PARAMETER_TENSORS):
        ptrs[i][0] = params_memory_iterator
        params_memory_iterator += param_sizes[i]
    return params_memory


@value
@register_passable
struct ActivationTensors:
    var encoded: Pointer[Float32]  # (B, T, C)
    var ln1: Pointer[Float32]  # (L, B, T, C)
    var ln1_mean: Pointer[Float32]  # (L, B, T)
    var ln1_rstd: Pointer[Float32]  # (L, B, T)
    var qkv: Pointer[Float32]  # (L, B, T, 3*C)
    var atty: Pointer[Float32]  # (L, B, T, C)
    var preatt: Pointer[Float32]  # (L, B, NH, T, T)
    var att: Pointer[Float32]  # (L, B, NH, T, T)
    var attproj: Pointer[Float32]  # (L, B, T, C)
    var residual2: Pointer[Float32]  # (L, B, T, C)
    var ln2: Pointer[Float32]  # (L, B, T, C)
    var ln2_mean: Pointer[Float32]  # (L, B, T)
    var ln2_rstd: Pointer[Float32]  # (L, B, T)
    var fch: Pointer[Float32]  # (L, B, T, 4*C)
    var fch_gelu: Pointer[Float32]  # (L, B, T, 4*C)
    var fcproj: Pointer[Float32]  # (L, B, T, C)
    var residual3: Pointer[Float32]  # (L, B, T, C)
    var lnf: Pointer[Float32]  # (B, T, C)
    var lnf_mean: Pointer[Float32]  # (B, T)
    var lnf_rstd: Pointer[Float32]  # (B, T)
    var logits: Pointer[Float32]  # (B, T, V)
    var probs: Pointer[Float32]  # (B, T, V)
    var losses: Pointer[Float32]  # (B, T)
    fn __init__(inout self):
        self.encoded = Pointer[Float32].alloc(4)
        self.ln1 = Pointer[Float32].alloc(4)
        self.ln1_mean = Pointer[Float32].alloc(4)
        self.ln1_rstd = Pointer[Float32].alloc(4)
        self.qkv = Pointer[Float32].alloc(4)
        self.atty = Pointer[Float32].alloc(4)
        self.preatt = Pointer[Float32].alloc(4)
        self.att = Pointer[Float32].alloc(4)
        self.attproj = Pointer[Float32].alloc(4)
        self.residual2= Pointer[Float32].alloc(4)
        self.ln2 = Pointer[Float32].alloc(4)
        self.ln2_mean = Pointer[Float32].alloc(4)
        self.ln2_rstd = Pointer[Float32].alloc(4)
        self.fch = Pointer[Float32].alloc(4)
        self.fch_gelu = Pointer[Float32].alloc(4)
        self.fcproj= Pointer[Float32].alloc(4)
        self.residual3 = Pointer[Float32].alloc(4)
        self.lnf = Pointer[Float32].alloc(4)
        self.lnf_mean = Pointer[Float32].alloc(4)
        self.lnf_rstd = Pointer[Float32].alloc(4)
        self.logits= Pointer[Float32].alloc(4)
        self.probs = Pointer[Float32].alloc(4)
        self.losses = Pointer[Float32].alloc(4)

fn malloc_and_point_activations(
    acts: Pointer[ActivationTensors], act_sizes: UInt32
) raises -> Pointer[Float32]:
    var num_activations: UInt32 = 0
    for i in range(NUM_ACTIVATION_TENSORS):
        num_activations += act_sizes[i]
    var acts_memory = Pointer[Float32].alloc(int(num_activations) * sizeof[Float32]())
    var ptrs = List(
        Pointer.address_of(acts[0].encoded),
        Pointer.address_of(acts[0].ln1),
        Pointer.address_of(acts[0].ln1_mean),
        Pointer.address_of(acts[0].ln1_rstd),
        Pointer.address_of(acts[0].qkv),
        Pointer.address_of(acts[0].atty),
        Pointer.address_of(acts[0].preatt),
        Pointer.address_of(acts[0].att),
        Pointer.address_of(acts[0].attproj),
        Pointer.address_of(acts[0].residual2),
        Pointer.address_of(acts[0].ln2),
        Pointer.address_of(acts[0].ln2_mean),
        Pointer.address_of(acts[0].ln2_rstd),
        Pointer.address_of(acts[0].fch),
        Pointer.address_of(acts[0].fch_gelu),
        Pointer.address_of(acts[0].fcproj),
        Pointer.address_of(acts[0].residual3),
        Pointer.address_of(acts[0].lnf),
        Pointer.address_of(acts[0].lnf_mean),
        Pointer.address_of(acts[0].lnf_rstd),
        Pointer.address_of(acts[0].logits),
        Pointer.address_of(acts[0].probs),
        Pointer.address_of(acts[0].losses),
    )
    var acts_memory_iterator = acts_memory
    for i in range(NUM_ACTIVATION_TENSORS):
        ptrs[i][0] = acts_memory_iterator
        acts_memory_iterator += act_sizes[i]
    return acts_memory


@value
@register_passable
struct GPT2Config:
    var max_seq_len: Int32  # max sequence length, e.g. 1024
    var vocab_size: Int32  # vocab size, e.g. 50257
    var num_layers: Int32  # number of layers, e.g. 12
    var num_heads: Int32  # number of heads in attention, e.g. 12
    var channels: Int32  # number of channels, e.g. 768

    fn __init__(inout self):
        self.max_seq_len = 1024
        self.vocab_size = 50257
        self.num_layers = 12
        self.num_heads = 12
        self.channels = 768

@value
@register_passable
struct GPT2:
    var config: GPT2Config
    # the weights of the model, and their sizes
    var params: ParameterTensors
    var param_sizes: Pointer[UInt32]  # FIXME: size
    var params_memory: Pointer[Float32]
    var num_parameters: Int
    # gradients of the weights
    var grads: ParameterTensors
    var grads_memory: Pointer[Float32]
    # buffers for the AdamW optimizer
    var m_memory: Pointer[Float32]
    var v_memory: Pointer[Float32]
    # the activations of the model, and their sizes
    var acts: ActivationTensors
    var act_sizes: Pointer[UInt32]  # FIXME: size
    var acts_memory: Pointer[Float32]
    var num_activations: Int32
    # gradients of the activations
    var grads_acts: ActivationTensors
    var grads_acts_memory: Pointer[Float32]
    # other run state configuration
    var batch_size: Int32  # the batch size (B) of current forward pass
    var seq_len: Int32  # the sequence length (T) of current forward pass
    var inputs: Pointer[Int32]  # the input tokens for the current forward pass
    var targets: Pointer[Int32]  # the target tokens for the current forward pass
    var mean_loss: Float32  # after a forward pass with targets, will be populated with the mean loss

    fn __init__(inout self):
        self.param_sizes = Pointer[UInt32].alloc(
            int(NUM_PARAMETER_TENSORS) * sizeof[Int32]()
        )
        self.act_sizes = Pointer[UInt32].alloc(
            int(NUM_PARAMETER_TENSORS) * sizeof[Int32]()
        )
        self.config = GPT2Config()
        self.params = ParameterTensors()
        self.params_memory = Pointer[Float32].alloc(4)
        self.grads = ParameterTensors()
        self.grads_memory=Pointer[Float32].alloc(4)
        self.m_memory=Pointer[Float32].alloc(4)
        self.v_memory=Pointer[Float32].alloc(4)
        self.acts= ActivationTensors()
        self.acts_memory=Pointer[Float32].alloc(4)
        self.num_activations = 0
        self.grads_acts = ActivationTensors()
        self.grads_acts_memory=Pointer[Float32].alloc(4)
        self.batch_size = 0
        self.seq_len = 0
        self.mean_loss = 0.0
        self.num_parameters = 0
        self.inputs = Pointer[Int32].alloc(4)
        self.targets = Pointer[Int32].alloc(4)

fn gpt2_build_from_checkpoint(
    model: Pointer[GPT2], checkpoint_path: String
) raises -> None:
    # read in model from a checkpoint file
    var model_file = open(checkpoint_path, "rb")
    if not model_file.handle:
        print("Error opening model fi 12 le\n")
        abort(1)
    var model_header = Pointer[Int32].alloc(256 * sizeof[Int32]())
    var _x = model_file.read(256 * sizeof[Int32]())
    storeToMem(model_header, _x._steal_ptr().bitcast[DType.uint8](), 256)
    if model_header[0] != 20240326:
        print("Bad magic model file")
        abort(1)
    if model_header[1] != 1:
        print("Bad version in model file")
        abort(1)

    # read in hyperparameters
    var maxT: Int32 = model_header[2]
    var V: Int32
    var L: Int32
    var NH: Int32
    var C: Int32
    model[0].config.max_seq_len = maxT
    model[0].config.vocab_size = V = model_header[3]
    model[0].config.num_layers = L = model_header[4]
    model[0].config.num_heads = NH = model_header[5]
    model[0].config.channels = C = model_header[6]
    print("[GPT-2]")
    print("max_seq_len: ", maxT)
    print("vocab_size: ", V)
    print("num_layers: ", L)
    print("num_heads: ", NH)
    print("channels: ", C)

    # allocate space for all the parameters and read them in
    model[0].param_sizes[0] = int(V * C)
    model[0].param_sizes[1] = int(maxT * C)
    model[0].param_sizes[2] = int(L * C)
    model[0].param_sizes[3] = int(L * C)
    model[0].param_sizes[4] = int(L * (3 * C) * C)
    model[0].param_sizes[5] = int(L * (3 * C))
    model[0].param_sizes[6] = int(L * C * C)
    model[0].param_sizes[7] = int(L * C)
    model[0].param_sizes[8] = int(L * C)
    model[0].param_sizes[9] = int(L * C)
    model[0].param_sizes[10] = int(L * (4 * C) * C)
    model[0].param_sizes[11] = int(L * (4 * C))
    model[0].param_sizes[12] = int(L * C * (4 * C))
    model[0].param_sizes[13] = int(L * C)
    model[0].param_sizes[14] = int(C)
    model[0].param_sizes[15] = int(C)

    # cound the number of paramaters
    var num_parameters = 0
    for i in range(NUM_PARAMETER_TENSORS):
        num_parameters += int(model[0].param_sizes[i])
    print("num_parameters: ", num_parameters)
    model[0].num_parameters = num_parameters

    # read in all the parameters from file
    var dd =  malloc_and_point_parameters(
        Pointer.address_of(model[0].params), model[0].param_sizes
    )
    model[0].params_memory = dd
    var _y = model_file.read(num_parameters * sizeof[Float32]())
    storeToMem(
        model[0].params_memory, _y._steal_ptr().bitcast[DType.uint8](), num_parameters
    )
    model_file.close()

    # other inits
    model[0].acts_memory = Pointer[Float32]().get_null()
    model[0].grads_memory = Pointer[Float32]().get_null()
    model[0].m_memory = Pointer[Float32]().get_null()
    model[0].v_memory = Pointer[Float32]().get_null()
    model[0].grads_acts_memory = Pointer[Float32]().get_null()
    model[0].inputs = Pointer[Int32]().get_null()
    model[0].targets = Pointer[Int32]().get_null()
    model[0].batch_size = 0
    model[0].seq_len = 0
    model[0].mean_loss = -1.0  # -1.0f will designate no loss


fn gpt2_forward(
    model: Pointer[GPT2],
    inputs: Pointer[Int32],
    targets: Pointer[Int32],
    B: Int,
    T: Int,
) raises -> None:
    # targets are optional and could be NULL

    # ensure the model was initialized or error out
    if not model[0].params_memory:
        print("Error: model was not initialized properly.")
        abort(1)

    # convenience parameters
    var V: Int32 = model[0].config.vocab_size
    var L: Int32 = model[0].config.num_layers
    var NH: Int32 = model[0].config.num_heads
    var C: Int32 = model[0].config.channels

    # allocate space for all the activations if needed (done here, lazily)
    if not model[0].acts_memory:
        # record the current B,T as well
        model[0].batch_size = B
        model[0].seq_len = T
        # and now allocate the space
        model[0].act_sizes[0] = int(B * T * C)
        model[0].act_sizes[1] = int(L * B * T * C)
        model[0].act_sizes[2] = int(L * B * T)
        model[0].act_sizes[3] = int(L * B * T)
        model[0].act_sizes[4] = int(L * B * T * 3 * C)
        model[0].act_sizes[5] = int(L * B * T * C)
        model[0].act_sizes[6] = int(L * B * NH * T * T)
        model[0].act_sizes[7] = int(L * B * NH * T * T)
        model[0].act_sizes[8] = int(L * B * T * C)
        model[0].act_sizes[9] = int(L * B * T * C)
        model[0].act_sizes[10] = int(L * B * T * C)
        model[0].act_sizes[11] = int(L * B * T)
        model[0].act_sizes[12] = int(L * B * T)
        model[0].act_sizes[13] = int(L * B * T * 4 * C)
        model[0].act_sizes[14] = int(L * B * T * 4 * C)
        model[0].act_sizes[15] = int(L * B * T * C)
        model[0].act_sizes[16] = int(L * B * T * C)
        model[0].act_sizes[17] = int(B * T * C)
        model[0].act_sizes[18] = int(B * T)
        model[0].act_sizes[19] = int(B * T)
        model[0].act_sizes[20] = int(B * T * V)
        model[0].act_sizes[21] = int(B * T * V)
        model[0].act_sizes[22] = int(B * T)
        var num_activations = 0
        for i in range(NUM_ACTIVATION_TENSORS):
            num_activations += int(model[0].act_sizes[i])
        print("num_activations: ", num_activations)
        model[0].num_activations = num_activations
        model[0].acts_memory = malloc_and_point_activations(
            Pointer.address_of(model[0].acts), int(model[0].act_sizes)
        )
        # also create memory for caching inputs and targets
        model[0].inputs = Pointer[Int32].alloc(B * T * sizeof[Int32]())
        model[0].targets = Pointer[Int32].alloc(
            B * T * sizeof[Int32]()
        )  # might be unused if we never have targets but it's small
    else:
        # validate B,T is no larger than what was previously allocated
        # in principle, we could re-allocate a larger chunk of memory, for now we just error out
        if B > int(model[0].batch_size) or T > int(model[0].seq_len):
            print("Error: batch size or sequence length is inadequately large")
            print(
                "Model: B= ", model[0].batch_size, "T= ", model[0].seq_len, "Desired: B=", B, "T=",T)
            abort(1)

    # cache the inputs/targets
    memcpy(model[0].inputs, inputs, B * T * sizeof[Int32]())
    if targets:
        memcpy(model[0].targets, targets, B * T * sizeof[Int32]())

    # forward pass
    var params: ParameterTensors = model[0].params  # for brevity
    var acts: ActivationTensors = model[0].acts
    var residual: Pointer[Float32]
    encoder_forward(
        acts.encoded, inputs, params.wte, params.wpe, int(B), int(T), int(C)
    )  # encoding goes into residual[0]
    for l in range(L):
        residual = acts.encoded if l == 0 else acts.residual3 + (l - 1) * B * T * C

        # get the pointers of the weights for this layer
        var l_ln1w: Pointer[Float32] = params.ln1w + l * C
        var l_ln1b: Pointer[Float32] = params.ln1b + l * C
        var l_qkvw: Pointer[Float32] = params.qkvw + l * 3 * C * C
        var l_qkvb: Pointer[Float32] = params.qkvb + l * 3 * C
        var l_attprojw: Pointer[Float32] = params.attprojw + l * C * C
        var l_attprojb: Pointer[Float32] = params.attprojb + l * C
        var l_ln2w: Pointer[Float32] = params.ln2w + l * C
        var l_ln2b: Pointer[Float32] = params.ln2b + l * C
        var l_fcw: Pointer[Float32] = params.fcw + l * 4 * C * C
        var l_fcb: Pointer[Float32] = params.fcb + l * 4 * C
        var l_fcprojw: Pointer[Float32] = params.fcprojw + l * C * 4 * C
        var l_fcprojb: Pointer[Float32] = params.fcprojb + l * C
        var l_ln1: Pointer[Float32] = acts.ln1 + l * B * T * C
        var l_ln1_mean: Pointer[Float32] = acts.ln1_mean + l * B * T
        var l_ln1_rstd: Pointer[Float32] = acts.ln1_rstd + l * B * T
        var l_qkv: Pointer[Float32] = acts.qkv + l * B * T * 3 * C
        var l_atty: Pointer[Float32] = acts.atty + l * B * T * C
        var l_preatt: Pointer[Float32] = acts.preatt + l * B * NH * T * T
        var l_att: Pointer[Float32] = acts.att + l * B * NH * T * T
        var l_attproj: Pointer[Float32] = acts.attproj + l * B * T * C
        var l_residual2: Pointer[Float32] = acts.residual2 + l * B * T * C
        var l_ln2: Pointer[Float32] = acts.ln2 + l * B * T * C
        var l_ln2_mean: Pointer[Float32] = acts.ln2_mean + l * B * T
        var l_ln2_rstd: Pointer[Float32] = acts.ln2_rstd + l * B * T
        var l_fch: Pointer[Float32] = acts.fch + l * B * T * 4 * C
        var l_fch_gelu: Pointer[Float32] = acts.fch_gelu + l * B * T * 4 * C
        var l_fcproj: Pointer[Float32] = acts.fcproj + l * B * T * C
        var l_residual3: Pointer[Float32] = acts.residual3 + l * B * T * C

        # now do the forward pass
        layernorm_forward(
            l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, int(C)
        )
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C)
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, int(C), int(NH))
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
        residual_forward(l_residual2, residual, l_attproj, int(B * T * C))
        layernorm_forward(
            l_ln2,
            l_ln2_mean,
            l_ln2_rstd,
            l_residual2,
            l_ln2w,
            l_ln2b,
            B,
            int(T),
            int(C),
        )
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C)
        gelu_forward(l_fch_gelu, l_fch, int(B * T * 4 * C))
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C)
        residual_forward(l_residual3, l_residual2, l_fcproj, int(B * T * C))
    residual = acts.residual3 + (L - 1) * B * T * C  # last residual is in residual3
    layernorm_forward(
        acts.lnf,
        acts.lnf_mean,
        acts.lnf_rstd,
        residual,
        params.lnfw,
        params.lnfb,
        B,
        int(T),
        int(C),
    )
    matmul_forward(
        acts.logits, acts.lnf, params.wte, Pointer[Float32].get_null(), B, T, C, V
    )
    softmax_forward(acts.probs, acts.logits, B, T, int(V))

    # also forward the cross-entropy loss function if we have the targets
    if targets:
        crossentropy_forward(
            model[0].acts.losses, model[0].acts.probs, targets, int(B), int(T), int(V)
        )
        # for convenience also evaluate the mean loss
        var mean_loss: Float32 = 0.0
        for i in range(B * T):
            mean_loss += model[0].acts.losses[i]
        mean_loss /= B * T
        model[0].mean_loss = mean_loss
    else:
        # if we don't have targets, we don't have a loss
        model[0].mean_loss = -1.0


fn gpt2_zero_grad(model: Pointer[GPT2]) raises -> None:
    if model[0].grads_memory:
        memset(
            model[0].grads_memory, 0, int(model[0].num_parameters * sizeof[Float32]())
        )
    if model[0].grads_acts_memory:
        memset(
            model[0].grads_acts_memory,
            0,
            int(model[0].num_activations * sizeof[Float32]()),
        )


fn gpt2_backward(model: Pointer[GPT2]) raises -> None:
    # double check we forwarded previously, with targets
    if model[0].mean_loss == -1.0:
        print("Error: must forward with targets before backward")
        abort(1)

    # lazily allocate the memory for gradients of the weights and activations, if needed
    if not model[0].grads_memory:
        model[0].grads_memory = malloc_and_point_parameters(
            Pointer.address_of(model[0].grads), model[0].param_sizes
        )
        model[0].grads_acts_memory = malloc_and_point_activations(
            Pointer.address_of(model[0].grads_acts), int(model[0].act_sizes)
        )
        gpt2_zero_grad(model)

    # convenience shortcuts
    var B: Int32 = model[0].batch_size
    var T = model[0].seq_len
    var V = model[0].config.vocab_size
    var L = model[0].config.num_layers
    var NH = model[0].config.num_heads
    var C = model[0].config.channels

    # backward pass
    var params: ParameterTensors = model[0].params  # for brevity
    var grads: ParameterTensors = model[0].grads
    var acts: ActivationTensors = model[0].acts
    var grads_acts: ActivationTensors = model[0].grads_acts

    # we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
    var dloss_mean = 1.00 / (B * T)
    for i in range(B * T):
        grads_acts.losses[i] = dloss_mean.cast[DType.float32]()

    crossentropy_softmax_backward(
        grads_acts.logits,
        grads_acts.losses,
        acts.probs,
        model[0].targets,
        int(B),
        int(T),
        int(V),
    )
    matmul_backward(
        grads_acts.lnf,
        grads.wte,
        Pointer[Float32].get_null(),
        grads_acts.logits,
        acts.lnf,
        params.wte,
        int(B),
        int(T),
        int(C),
        int(V),
    )
    var residual: Pointer[Float32] = acts.residual3 + (
        L - 1
    ) * B * T * C  # last layer's residual
    var dresidual: Pointer[Float32] = grads_acts.residual3 + (
        L - 1
    ) * B * T * C  # write to last layer's residual
    layernorm_backward(
        dresidual,
        grads.lnfw,
        grads.lnfb,
        grads_acts.lnf,
        residual,
        params.lnfw,
        acts.lnf_mean,
        acts.lnf_rstd,
        int(B),
        int(T),
        int(C),
    )

    for l in range(L - 1, -1, -1):
        residual = acts.encoded if l == 0 else acts.residual3 + (l - 1) * B * T * C
        dresidual = (
            grads_acts.encoded if l == 0 else grads_acts.residual3 + (l - 1) * B * T * C
        )

        # get the pointers of the weights for this layer
        var l_ln1w = params.ln1w + l * C
        var l_qkvw = params.qkvw + l * 3 * C * C
        var l_attprojw = params.attprojw + l * C * C
        var l_ln2w = params.ln2w + l * C
        var l_fcw = params.fcw + l * 4 * C * C
        var l_fcprojw = params.fcprojw + l * C * 4 * C
        # get the pointers of the gradients of the weights for this layer
        var dl_ln1w = grads.ln1w + l * C
        var dl_ln1b = grads.ln1b + l * C
        var dl_qkvw = grads.qkvw + l * 3 * C * C
        var dl_qkvb = grads.qkvb + l * 3 * C
        var dl_attprojw = grads.attprojw + l * C * C
        var dl_attprojb = grads.attprojb + l * C
        var dl_ln2w = grads.ln2w + l * C
        var dl_ln2b = grads.ln2b + l * C
        var dl_fcw = grads.fcw + l * 4 * C * C
        var dl_fcb = grads.fcb + l * 4 * C
        var dl_fcprojw = grads.fcprojw + l * C * 4 * C
        var dl_fcprojb = grads.fcprojb + l * C
        # get the pointers of the activations for this layer
        var l_ln1 = acts.ln1 + l * B * T * C
        var l_ln1_mean = acts.ln1_mean + l * B * T
        var l_ln1_rstd = acts.ln1_rstd + l * B * T
        var l_qkv = acts.qkv + l * B * T * 3 * C
        var l_atty = acts.atty + l * B * T * C
        var l_att = acts.att + l * B * NH * T * T
        var l_residual2 = acts.residual2 + l * B * T * C
        var l_ln2 = acts.ln2 + l * B * T * C
        var l_ln2_mean = acts.ln2_mean + l * B * T
        var l_ln2_rstd = acts.ln2_rstd + l * B * T
        var l_fch = acts.fch + l * B * T * 4 * C
        var l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C
        # get the pointers of the gradients of the activations for this layer
        var dl_ln1 = grads_acts.ln1 + l * B * T * C
        var dl_qkv = grads_acts.qkv + l * B * T * 3 * C
        var dl_atty = grads_acts.atty + l * B * T * C
        var dl_preatt = grads_acts.preatt + l * B * NH * T * T
        var dl_att = grads_acts.att + l * B * NH * T * T
        var dl_attproj = grads_acts.attproj + l * B * T * C
        var dl_residual2 = grads_acts.residual2 + l * B * T * C
        var dl_ln2 = grads_acts.ln2 + l * B * T * C
        var dl_fch = grads_acts.fch + l * B * T * 4 * C
        var dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C
        var dl_fcproj = grads_acts.fcproj + l * B * T * C
        var dl_residual3 = grads_acts.residual3 + l * B * T * C
        # backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, int(B * T * C))
        matmul_backward(
            dl_fch_gelu,
            dl_fcprojw,
            dl_fcprojb,
            dl_fcproj,
            l_fch_gelu,
            l_fcprojw,
            int(B),
            int(T),
            int(4 * C),
            int(C),
        )
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, int(B * T * 4 * C))
        matmul_backward(
            dl_ln2,
            dl_fcw,
            dl_fcb,
            dl_fch,
            l_ln2,
            l_fcw,
            int(B),
            int(T),
            int(C),
            int(4 * C),
        )
        layernorm_backward(
            dl_residual2,
            dl_ln2w,
            dl_ln2b,
            dl_ln2,
            l_residual2,
            l_ln2w,
            l_ln2_mean,
            l_ln2_rstd,
            int(B),
            int(T),
            int(C),
        )
        residual_backward(dresidual, dl_attproj, dl_residual2, int(B * T * C))
        matmul_backward(
            dl_atty,
            dl_attprojw,
            dl_attprojb,
            dl_attproj,
            l_atty,
            l_attprojw,
            int(B),
            int(T),
            int(C),
            int(C),
        )
        attention_backward(
            dl_qkv,
            dl_preatt,
            dl_att,
            dl_atty,
            l_qkv,
            l_att,
            int(B),
            int(T),
            int(C),
            int(NH),
        )
        matmul_backward(
            dl_ln1,
            dl_qkvw,
            dl_qkvb,
            dl_qkv,
            l_ln1,
            l_qkvw,
            int(B),
            int(T),
            int(C),
            int(3 * C),
        )
        layernorm_backward(
            dresidual,
            dl_ln1w,
            dl_ln1b,
            dl_ln1,
            residual,
            l_ln1w,
            l_ln1_mean,
            l_ln1_rstd,
            int(B),
            int(T),
            int(C),
        )
    encoder_backward(
        grads.wte,
        grads.wpe,
        grads_acts.encoded,
        model[0].inputs,
        int(B),
        int(T),
        int(C),
    )


fn gpt2_update(
    model: Pointer[GPT2],
    learning_rate: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    t: Int32,
) raises -> None:
    # reference: https:#pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    # lazily allocate the memory for m_memory and v_memory
    if not model[0].m_memory:
        model[0].m_memory = Pointer[Float32].alloc(
            model[0].num_parameters * sizeof[Float32]()
        )
        model[0].v_memory = Pointer[Float32].alloc(
            model[0].num_parameters * sizeof[Float32]()
        )

    for i in range(model[0].num_parameters):
        var param = model[0].params_memory[i]
        var grad = model[0].grads_memory[i]

        # update the first moment (momentum)
        var m = beta1 * model[0].m_memory[i] + (1.0 - beta1) * grad
        # update the second moment (RMSprop)
        var v = beta2 * model[0].v_memory[i] + (1.0 - beta2) * grad * grad
        # bias-correct both moments
        var m_hat = m / (1.0 - math.pow(beta1, t))  # FIXME:
        var v_hat = v / (1.0 - math.pow(beta2, t))  # FIXME:

        # update
        model[0].m_memory[i] = m
        model[0].v_memory[i] = v
        model[0].params_memory[i] -= learning_rate * (
            m_hat / (math.sqrt(v_hat) + eps) + weight_decay * param
        )  # FIXME


fn gpt2_free(model: Pointer[GPT2]) raises -> None:
    model[0].params_memory.free()
    model[0].grads_memory.free()
    model[0].m_memory.free()
    model[0].v_memory.free()
    model[0].acts_memory.free()
    model[0].grads_acts_memory.free()
    model[0].inputs.free()
    model[0].targets.free()

@value
@register_passable
struct DataLoader:
    # hyperparameters
    var B: Int32
    var T: Int32
    var file: Pointer[FileHandle]
    var file_Path: StringLiteral
    # input handling and its state
    var file_size: Int64
    var current_position: Int64
    # output memory
    var batch: Pointer[Int32]
    var inputs: Pointer[Int32]
    var targets: Pointer[Int32]
    # convenience variables
    var num_batches: Int32

    fn __init__(inout self: DataLoader):
        self.B = 64
        self.T = 256
        self.file = Pointer[FileHandle].alloc(4)
        self.file_Path = ""
        self.file_size = 0
        self.current_position = 0
        self.batch = Pointer[Int32].alloc(4)
        self.inputs = Pointer[Int32].alloc(4)
        self.targets = Pointer[Int32].alloc(4)
        self.num_batches = 0

           
fn dataloader_init(loader: Pointer[DataLoader], filename: StringLiteral, B: Int, T: Int) raises -> None:
    loader[0].B = B
    loader[0].T = T
    loader[0].file_Path = filename
    var fd = open(filename, "rb")
    loader[0].file =Pointer.address_of(fd)
    # open the input file for reading
    if not loader[0].file:
        print("Error opening tokens file\n")
        abort(1 )
    # determine the file size
    loader[0].file_size = len(fd.read())
    _ = fd.seek(0)
    if (loader[0].file_size < (B * T + 1) * sizeof[Int32]()):
        print("Error: file size is too small for the batch size and sequence length\n")
        abort(1)
    loader[0].current_position = 0 # start at the beginning

    # allocate space for B*T + 1 integers to store the inputs and targets
    loader[0].batch = Pointer[Int32].alloc((B * T + 1) * sizeof[Int32]())
    loader[0].inputs = loader[0].batch
    loader[0].targets = loader[0].batch + 1 # targets are shifted by one
    loader[0].num_batches = int(loader[0].file_size / (B * T * sizeof[Int32]()))

fn dataloader_reset(loader: Pointer[DataLoader]) raises -> None:
    loader[0].current_position = 0

fn dataloader_next_batch(loader: Pointer[DataLoader]) raises -> None:
    var B = loader[0].B
    var T = loader[0].T
    # if we are at the end of the file, loop back to the beginning
    if (int(loader[0].current_position) + (B*T+1) * sizeof[Int32]() > int(loader[0].file_size)):
        loader[0].current_position = 0;
    var fd = open(loader[0].file_Path, "rb")
    var x = fd.read(int(B*T+1)*sizeof[Int32]())
    # read the B*T+1 integers from the file into batch
    # loader[0].file.offset(0).a, loader[0].current_position, SEEK_SET);
    storeToMem(loader[0].batch, x._steal_ptr().bitcast[DType.uint8](), int(B*T+1))
    #advance the current position by B*T integers
    loader[0].current_position += int(B*T * sizeof[Int32]())


fn dataloader_free(loader: Pointer[DataLoader]) raises -> None:
    loader[0].batch.free()

fn random_u32(state: DTypePointer[DType.uint64]) raises -> UInt32:
    # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state[0] ^= state[0] >> 12
    state[0] ^= state[0] << 25
    state[0] ^= state[0] >> 27
    return int((state[0] * 0x2545F4914F6CDD1D) >> 32)

fn random_f32(state: DTypePointer[DType.uint64]) raises -> Float32:
    var f: Float32 = 16777216.0
    return 0.5

fn sample_mult(probabilities: Pointer[Float32], n: Int32, coin: Float32) raises -> Int32:
    # sample index from probabilities (they must sum to 1!)
    # coin is a random number in [0, 1), usually from random_f32()
    var cdf: Float32 = 0.0
    for  i in range(n):
        cdf += probabilities[i]
        if (coin < cdf):
            return i
    return n - 1 # in case of rounding errors


fn main() raises -> None:

    # build the GPT-2 model from a checkpoint
    var model: GPT2 = GPT2()
    gpt2_build_from_checkpoint(Pointer.address_of(model),"data/gpt2_124M.bin")

    # build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    var tiny_stories_train = path.Path(path.cwd().joinpath("data/TinyStories_train.bin"))
    var tiny_stories_val = path.Path(path.cwd().joinpath("data/TinyStories_val.bin"))
    var tiny_shakespeare_train = path.Path(path.cwd().joinpath("data/tiny_shakespeare_train.bin"))
    var tiny_shakespeare_val = path.Path(path.cwd().joinpath("data/tiny_shakespeare_val.bin"))
    var train_tokens =  tiny_shakespeare_train
    var val_tokens = tiny_shakespeare_val
    var B = 4
    var T = 64
    var train_loader: DataLoader = DataLoader()
    dataloader_init(Pointer.address_of(train_loader), "/Users/zhoujing/codes/llm.c/data/TinyStories_train.bin", B, T)
    print("train dataset num_batches: ", train_loader.num_batches)
    var val_loader : DataLoader = DataLoader()
    dataloader_init(Pointer.address_of(val_loader), "/Users/zhoujing/codes/llm.c/data/TinyStories_val.bin", B, T)
    print("val dataset num_batches: ", val_loader.num_batches)
    var val_num_batches = 10

    # some memory for generating samples from the model
    var rng_state: UInt64 = 1337
    var gen_max_length = 64
    var gen_tokens: Pointer[Int32] = Pointer[Int32].alloc(gen_max_length)

    # train
    # struct timespec start, end #FIXME:
    for step in range(41):

        # once in a while estimate the validation loss
        if (step % 10 == 0):
            var val_loss: Float32 = 0.0
            dataloader_reset(Pointer.address_of(val_loader))
            for i in range(val_num_batches):
                dataloader_next_batch(Pointer.address_of(val_loader))
                gpt2_forward(Pointer.address_of(model), val_loader.inputs, val_loader.targets, B, T)
                val_loss += model.mean_loss
            val_loss /= val_num_batches
            print("val loss ", val_loss)

        # once in a while do model inference to print generated text
        if (step > 0 and step % 20 == 0):
            gen_tokens[0] = GPT2_EOT # the GPT-2 EOT token kicks off the generation
            for t in range(1,gen_max_length):
                # note that inference is wasteful here because
                # for each t, we re-compute all activations between 0 and t
                # leaving this alone because you want separate code for inference anyway
                # the inference here is just for sanity checking purposes
                gpt2_forward(Pointer.address_of(model), gen_tokens, Pointer[Int32].get_null(), 1, t)
                var probs = model.acts.probs + (t-1) * model.config.vocab_size
                var coin = random_f32(DTypePointer.address_of(rng_state))
                var next_token = sample_mult(probs, model.config.vocab_size, coin)
                gen_tokens[t] = next_token
            print("generated: ")
            for  t in range(gen_max_length):
                print(gen_tokens[t])
            print("")

        # do a training step
        # clock_gettime(CLOCK_MONOTONIC, &start)
        dataloader_next_batch(Pointer.address_of(train_loader))
        gpt2_forward(Pointer.address_of(model), train_loader.inputs, train_loader.targets, B, T)
        gpt2_zero_grad(Pointer.address_of(model))
        gpt2_backward(Pointer.address_of(model))
        gpt2_update(Pointer.address_of(model), 1e-4, 0.9, 0.999, 1e-8, 0.0, step+1)
        # clock_gettime(CLOCK_MONOTONIC, &end)
        # var time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9
        # printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000)

    # free
    dataloader_free(Pointer.address_of(train_loader))
    dataloader_free(Pointer.address_of(val_loader))
    gpt2_free(Pointer.address_of(model))
#endif
