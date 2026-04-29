// Fused GELU-multiply: o[i] = gelu(a[i]) * b[i]

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const COEFF: f32 = 0.044715;

struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let x = a[i];
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    let clamped = clamp(inner, -10.0, 10.0);
    let e2 = exp(2.0 * clamped);
    let tanh_val = (e2 - 1.0) / (e2 + 1.0);
    o[i] = 0.5 * x * (1.0 + tanh_val) * b[i];
}
