# ===----------------------------------------------------------------------=== #
# Copyright (C) 2020-2024 Terapines Technology (Wuhan) Co., Ltd
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import math

alias M_PI: Float32 = 3.14159265358979323846264338327950288

# ----------------------------------------------------------------------------
# all the individual layers' forward and backward passes


fn encoder_forward(
    out: Pointer[Float32],
    inp: Pointer[Float32],
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
    inp: Pointer[Float32],
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
                        # out_bth[i] += att_bth[t2] * value_t2[i];
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
                        # preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        # so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale


fn gelu_forward(out: Pointer[Float32], inp: Pointer[Float32], N: Int) 
    raises -> None:
    var s = math.sqrt(2.0 / M_PI)  # FIXME: sqrtf
    for i in range(N):
        var x = inp[i]
        var cube = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1.0 + math.tanh(s * (x + cube)))  # FIXME:tanhf


fn gelu_backward(
    dinp: Pointer[Float32],
    inp: Pointer[Float32],
    dout: Pointer[Float32],
    N: Pointer[Float32],
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
