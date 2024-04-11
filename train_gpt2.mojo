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
