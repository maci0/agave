struct Params {
    conv_ch: u32,
    d_conv: u32,
    has_bias: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_buf: array<f32>;
@group(0) @binding(1) var<storage, read> state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<storage, read> conv_b: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ch = gid.x;
    if (ch >= params.conv_ch) { return; }

    var sum: f32 = 0.0;
    for (var i = 0u; i < params.d_conv - 1u; i = i + 1u) {
        sum = sum + state[i * params.conv_ch + ch] * conv_w[ch * params.d_conv + i];
    }
    sum = sum + input_buf[ch] * conv_w[ch * params.d_conv + (params.d_conv - 1u)];

    if (params.has_bias != 0u) {
        sum = sum + conv_b[ch];
    }

    let sigmoid_val = 1.0 / (1.0 + exp(-sum));
    out_buf[ch] = sum * sigmoid_val;
}
