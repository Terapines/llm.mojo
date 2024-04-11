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
import pathlib.path as path

alias eps = 1e-5
alias B = 2
alias C = 4
alias T = 3
alias float_size = sizeof[DType.float32]()

@always_inline("nodebug")
fn layernorm_forward(
    out: Pointer[Float32],
    mean: Pointer[Float32],
    rstd: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    bias: Pointer[Float32],
) raises -> None:
    for b in range(B):
        for t in range(T):
            var x = inp + b * T * C + t * C
            var m: Float32 = 0.0
            for i in range(C):
                m += x[i]
            m = m / C
            var v: Float32 = 0.0
            for i in range(C):
                var xsgift = x[i] - m
                v += xsgift * xsgift
            v = v / C
            # Calculate the rstd
            var tmp: Float32 = math.sqrt(eps + v)
            var s: Float32 = 1.0 / tmp
            # seek to the output position in out[b,t,:]
            var out_bt = out + b * T * C + t * C
            for i in range(C):
                var n = (s * (x[i] - m))  # normalized output
                var o = n * weight[i] + bias[i]  # scale and shift it
                out_bt[i] = o
            mean[b * T + t] = m
            rstd[b * T + t] = s

@always_inline("nodebug")
fn layernorm_backward(
    dinp: Pointer[Float32],
    dweight: Pointer[Float32],
    dbias: Pointer[Float32],
    dout: Pointer[Float32],
    inp: Pointer[Float32],
    weight: Pointer[Float32],
    mean: Pointer[Float32],
    rstd: Pointer[Float32],
) raises -> None:
    for b in range(B):
        for t in range(T):
            var dout_bt = dout + b * T * C + t * C
            var inp_bt = inp + b * T * C + t * C
            var dinp_bt = dinp + b * T * C + t * C
            var mean_bt = mean[b * T + t]
            var rstd_bt = rstd[b * T + t]
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

@always_inline("nodebug")
fn check_tensor(
    a: Pointer[Float32], b: Pointer[Float32], n: Int, lable: String
) raises -> None:
    print(lable)
    for i in range(n):
        if math.abs(a[i] - b[i]) <= 1e-5:
            print("OK ")
        else:
            print("NOT OK")
        print(a[i], b[i])

@always_inline("nodebug")
fn storeToMem(mem: Pointer[Float32], base: DTypePointer, size: Int):
    for i in range(size):
        var tmp = base.offset(0).bitcast[DType.float32]().load(i)
        mem.store(i, tmp)


fn main() raises -> None:
    var x: Pointer[Float32] = Pointer[Float32].alloc(B * T * C * float_size)
    var w: Pointer[Float32] = Pointer[Float32].alloc(C * float_size)
    var b: Pointer[Float32] = Pointer[Float32].alloc(C * float_size)
    var out: Pointer[Float32] = Pointer[Float32].alloc(B * C * T * float_size)
    var mean: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)
    var rstd: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)
    var dout: Pointer[Float32] = Pointer[Float32].alloc(B * T * C * float_size)
    var dx: Pointer[Float32] = Pointer[Float32].alloc(B * T * C * float_size)
    var dw: Pointer[Float32] = Pointer[Float32].alloc(C * float_size)
    var db: Pointer[Float32] = Pointer[Float32].alloc(C * float_size)
    var reference = path.Path(path.cwd().joinpath("data/ln.bin"))
    var fd = open(reference, "rb")
    var _x = fd.read(B * T * C * float_size)
    var _w = fd.read(C * float_size)
    var _b = fd.read(C * float_size)
    var _out = fd.read(B * T * C * float_size)
    var _mean = fd.read(B * T * float_size)
    var _rstd = fd.read(B * T * float_size)
    var _dout = fd.read(B * T * C * float_size)
    var _dx = fd.read(B * T * C * float_size)
    var _dw = fd.read(C * float_size)
    var _db = fd.read(C * float_size)

    storeToMem(x, _x._steal_ptr().bitcast[DType.uint8](), B * T * C)
    storeToMem(w, _w._steal_ptr().bitcast[DType.uint8](), C)
    storeToMem(b, _b._steal_ptr().bitcast[DType.uint8](), C)
    storeToMem(out, _out._steal_ptr().bitcast[DType.uint8](), B * T * C)
    storeToMem(mean, _mean._steal_ptr().bitcast[DType.uint8](), B * T)
    storeToMem(rstd, _rstd._steal_ptr().bitcast[DType.uint8](), B * T)
    storeToMem(dout, _dout._steal_ptr().bitcast[DType.uint8](), B * T * C)
    storeToMem(dx, _dx._steal_ptr().bitcast[DType.uint8](), B * T * C)
    storeToMem(dw, _dw._steal_ptr().bitcast[DType.uint8](), C)
    storeToMem(db, _db._steal_ptr().bitcast[DType.uint8](), C)
    var c_out: Pointer[Float32] = Pointer[Float32].alloc(B * T * C * float_size)
    var c_mean: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)
    var c_rstd: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)

    layernorm_forward(c_out, c_mean, c_rstd, x, w, b)
    check_tensor(out, c_out, B * T * C, "out")
    check_tensor(mean, c_mean, B * T, "mean")
    check_tensor(rstd, c_rstd, B * T, "rstd")
    var c_dx: Pointer[Float32] = Pointer[Float32].alloc(B * T * C * float_size)
    var c_dw: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)
    var c_db: Pointer[Float32] = Pointer[Float32].alloc(B * T * float_size)
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd)
    check_tensor(c_dx, dx, B * T * C, "dx")
    check_tensor(c_dw, dw, C, "dw")
    check_tensor(c_db, db, C, "db")
    x.free()
    w.free()
    b.free()
    out.free()
    mean.free()
    rstd.free()
    dout.free()
    dx.free()
    dw.free()
    db.free()
    c_out.free()
    c_mean.free()
    c_rstd.free()
    c_dx.free()
    c_dw.free()
    c_db.free()
