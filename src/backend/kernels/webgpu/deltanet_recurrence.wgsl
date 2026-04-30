// DeltaNet SSM recurrence kernel — one workgroup per v-head.
// State update: S = decay * S + beta * delta * outer(k, k); out = S @ q * q_scale
// Where delta = beta * (v - S @ k)
// Sequential across v_dim (inherent data dependency).

struct Params {
    num_v_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    q_scale: f32,
    rms_eps: f32,
}

@group(0) @binding(0) var<storage, read> q_ptr: array<f32>;
@group(0) @binding(1) var<storage, read> k_ptr: array<f32>;
@group(0) @binding(2) var<storage, read> v_ptr: array<f32>;
@group(0) @binding(3) var<storage, read> gate_vals: array<f32>;
@group(0) @binding(4) var<storage, read> beta_vals: array<f32>;
@group(0) @binding(5) var<storage, read> z_buf: array<f32>;
@group(0) @binding(6) var<storage, read> ssm_norm_w: array<f32>;
@group(0) @binding(7) var<storage, read_write> ssm_state: array<f32>;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wgid: vec3<u32>) {
    let h = wgid.x;
    if (h >= params.num_v_heads) { return; }

    let hkd = params.head_k_dim;
    let hvd = params.head_v_dim;
    let nkh = params.num_k_heads;
    let nvh = params.num_v_heads;

    let decay = exp(gate_vals[h]);
    let beta_h = beta_vals[h];
    let kh = select(h % nkh, h * nkh / nvh, nkh == nvh);
    let s_off = h * hvd * hkd;
    let k_base = kh * hkd;

    // dot(K, Q)
    var kq: f32 = 0.0;
    for (var ki = 0u; ki < hkd; ki = ki + 1u) {
        kq = kq + k_ptr[k_base + ki] * q_ptr[k_base + ki];
    }

    // Recurrence per v-dim element
    for (var vi = 0u; vi < hvd; vi = vi + 1u) {
        let row_off = s_off + vi * hkd;

        var sk: f32 = 0.0;
        var sq_dec: f32 = 0.0;
        for (var ki = 0u; ki < hkd; ki = ki + 1u) {
            let s_dec = ssm_state[row_off + ki] * decay;
            ssm_state[row_off + ki] = s_dec;
            sk = sk + s_dec * k_ptr[k_base + ki];
            sq_dec = sq_dec + s_dec * q_ptr[k_base + ki];
        }

        let delta = beta_h * (v_ptr[h * hvd + vi] - sk);
        output[h * hvd + vi] = (sq_dec + delta * kq) * params.q_scale;

        // Update state: S += delta * k
        for (var ki = 0u; ki < hkd; ki = ki + 1u) {
            ssm_state[row_off + ki] = ssm_state[row_off + ki] + k_ptr[k_base + ki] * delta;
        }
    }

    // Gated output: RMSNorm + SiLU gate
    let off = h * hvd;
    var ss: f32 = 0.0;
    for (var vi = 0u; vi < hvd; vi = vi + 1u) {
        ss = ss + output[off + vi] * output[off + vi];
    }
    let inv_rms = inverseSqrt(ss / f32(hvd) + params.rms_eps);
    for (var vi = 0u; vi < hvd; vi = vi + 1u) {
        let normed = output[off + vi] * ssm_norm_w[vi] * inv_rms;
        let z = z_buf[off + vi];
        let silu_z = z / (1.0 + exp(-z));
        output[off + vi] = normed * silu_z;
    }
}
