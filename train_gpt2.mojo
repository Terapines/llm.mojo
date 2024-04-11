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

import layernorm

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
