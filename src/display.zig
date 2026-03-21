//! Display module — types and output formatting for the Agave inference engine.
//! Provides TTY-aware output (banner, stats, progress) and JSON serialization.
//! Single source of truth for all display-related types used across main.zig and server.zig.

const std = @import("std");
const vaxis = @import("vaxis");

/// ANSI escape: erase from cursor to end of line (EL0).
const erase_line = "\x1b[K";

/// Engine version string, shared across all output modes.
pub const version = "0.1.0";

/// Bits per byte, used for bits-per-weight (bpw) calculation.
const bits_per_byte: f32 = 8.0;
/// Show available memory when available is less than this percentage of total.
const avail_display_pct: usize = 98;

// ── Public Types ─────────────────────────────────────────────────

/// Metadata about the loaded model, displayed in banner and JSON output.
pub const ModelInfo = struct {
    name: []const u8,
    arch_name: []const u8,
    quant: []const u8,
    be_name: []const u8,
    n_layers: u32,
    n_embed: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    ff_dim: u32,
    vocab_size: u32,
    ctx_size: u32,
    rope_theta: f32,
    n_params: u64,
    n_experts: u32,
    n_experts_used: u32,
    file_size_bytes: usize,
    load_ms: u64,
    warmup_ms: u64,
};

/// Re-export BackendInfo from backend.zig for display consumers.
pub const BackendInfo = @import("backend/backend.zig").BackendInfo;

/// Technical details about the model loading process.
pub const LoadInfo = struct {
    /// Number of tensors in the model file.
    n_tensors: u64 = 0,
    /// Tokenizer type name (e.g., "bpe", "spm", "spm_no_dummy").
    tok_kind: []const u8 = "",
    /// Vocabulary size.
    vocab_size: u32 = 0,
    /// EOS token ID.
    eos_id: u32 = 0,
    /// BOS token ID.
    bos_id: u32 = 0,
    /// Number of additional EOG token IDs (beyond EOS).
    n_eog: usize = 0,
    /// Chat template name (e.g., "chatml", "gemma", "qwen35", "gpt_oss").
    template_name: []const u8 = "",
    /// Model format ("GGUF v3", "SafeTensors").
    format_name: []const u8 = "",
    /// Model init time in milliseconds.
    init_ms: u64 = 0,
};

/// Generation statistics collected after a generation run.
pub const GenStats = struct {
    token_count: u32,
    gen_ms: u64,
    prefill_token_count: u32,
    prefill_ms: u64,

    /// Tokens per second during decode (generation phase).
    /// Returns 0.0 if gen_ms is zero to avoid division by zero.
    pub fn tokPerSec(self: GenStats) f32 {
        if (self.gen_ms == 0) return 0.0;
        return @as(f32, @floatFromInt(self.token_count)) / (@as(f32, @floatFromInt(self.gen_ms)) / 1000.0);
    }

    /// Tokens per second during prefill.
    /// Returns 0.0 if prefill_ms is zero to avoid division by zero.
    pub fn prefillTokPerSec(self: GenStats) f32 {
        if (self.prefill_ms == 0) return 0.0;
        return @as(f32, @floatFromInt(self.prefill_token_count)) / (@as(f32, @floatFromInt(self.prefill_ms)) / 1000.0);
    }
};

/// Single benchmark result for a compute operation.
pub const BenchResult = struct {
    op: []const u8,
    dimensions: []const u8,
    ms: f64,
    gbps: f64,
};

/// Output mode for all display operations.
pub const OutputMode = enum {
    tty,
    plain,
    json,
};

/// Human-friendly file size with unit string.
pub const FormattedSize = struct {
    val: f64,
    unit: []const u8,
};

// ── Public Functions ─────────────────────────────────────────────

/// Formats a byte count into a human-readable size with appropriate unit.
/// Uses binary (1024-based) thresholds: returns "GB" for >= 1 GiB, "MB" for >= 1 MiB, "KB" otherwise.
pub fn formatSize(size: usize) FormattedSize {
    const gib: usize = 1024 * 1024 * 1024;
    const mib: usize = 1024 * 1024;
    if (size >= gib) return .{ .val = @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(gib)), .unit = "GB" };
    if (size >= mib) return .{ .val = @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(mib)), .unit = "MB" };
    return .{ .val = @as(f64, @floatFromInt(size)) / 1024.0, .unit = "KB" };
}

/// Formats a large number compactly: 152064 → "152K", 1000000 → "1M", 9200000000 → "9.2B".
fn fmtCompact(buf: *[16]u8, n: u64) []const u8 {
    if (n >= 1_000_000_000) {
        const b = n / 1_000_000_000;
        const frac = (n % 1_000_000_000) / 100_000_000;
        if (frac > 0) return std.fmt.bufPrint(buf, "{d}.{d}B", .{ b, frac }) catch "";
        return std.fmt.bufPrint(buf, "{d}B", .{b}) catch "";
    }
    if (n >= 1_000_000) {
        const m = n / 1_000_000;
        const frac = (n % 1_000_000) / 100_000;
        if (frac > 0) return std.fmt.bufPrint(buf, "{d}.{d}M", .{ m, frac }) catch "";
        return std.fmt.bufPrint(buf, "{d}M", .{m}) catch "";
    }
    if (n >= 1_000) {
        return std.fmt.bufPrint(buf, "{d}K", .{ n / 1_000 }) catch "";
    }
    return std.fmt.bufPrint(buf, "{d}", .{n}) catch "";
}

/// Formats milliseconds as duration: 21515 → "21.5s", 800 → "800ms".
fn fmtDuration(buf: *[16]u8, ms: u64) []const u8 {
    if (ms >= 10_000) {
        return std.fmt.bufPrint(buf, "{d}s", .{ms / 1000}) catch "";
    }
    if (ms >= 1_000) {
        return std.fmt.bufPrint(buf, "{d}.{d}s", .{ ms / 1000, (ms % 1000) / 100 }) catch "";
    }
    return std.fmt.bufPrint(buf, "{d}ms", .{ms}) catch "";
}

// ── Internal Helpers ─────────────────────────────────────────────

/// Calculates the display width (terminal columns) of a UTF-8 string.
/// Uses libvaxis grapheme-aware width calculation for correct handling
/// of multi-byte characters, combining marks, and wide (CJK) glyphs.
fn displayWidth(s: []const u8) usize {
    return @as(usize, vaxis.gwidth.gwidth(s, .unicode));
}

/// Returns a byte-slice prefix of `s` whose display width does not exceed `max_cols`.
/// Cuts on UTF-8 codepoint boundaries to avoid invalid sequences.
fn truncateToWidth(s: []const u8, max_cols: usize) []const u8 {
    var cols: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const cp_len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        const end = @min(i + cp_len, s.len);
        const w = displayWidth(s[i..end]);
        if (cols + w > max_cols) break;
        cols += w;
        i = end;
    }
    return s[0..i];
}

// ── Display Struct ───────────────────────────────────────────────

/// Main display controller. Dispatches all output through the selected
/// OutputMode (tty, plain, json). No allocator needed — uses only
/// stack-buffered writes to stderr/stdout.
pub const Display = struct {
    mode: OutputMode,
    verbose: bool,

    const stderr = std.fs.File.stderr();
    const stdout = std.fs.File.stdout();

    /// Buffer size for formatted output.
    const out_buf_size: usize = 8192;

    /// Initialize a Display with the given output mode and verbosity.
    /// No allocator required — all output is stack-buffered.
    pub fn init(mode: OutputMode, verbose: bool) Display {
        return .{ .mode = mode, .verbose = verbose };
    }

    // ── Banner ───────────────────────────────────────────────

    /// Print the model banner, dispatching to TTY or plain format.
    pub fn printBanner(self: Display, info: ModelInfo) void {
        switch (self.mode) {
            .tty => self.printBannerTty(info),
            .plain => self.printBannerPlain(info),
            .json => {}, // JSON mode prints nothing for banner
        }
    }

    /// Plain-text single-line banner written to stderr.
    /// Verbose mode appends full model dimensions and timing.
    pub fn printBannerPlain(self: Display, info: ModelInfo) void {
        const fsize = formatSize(info.file_size_bytes);
        var buf: [out_buf_size]u8 = undefined;
        const text = if (self.verbose)
            std.fmt.bufPrint(&buf, "agave {s} \xc2\xb7 {s} \xc2\xb7 {s} \xc2\xb7 {d:.1}{s} \xc2\xb7 {s} \xc2\xb7 {d}L/{d}E/{d}FFN/{d}H/{d}KV/hd{d} ({d}+{d}ms)\n", .{
                info.name,
                info.arch_name,
                info.quant,
                fsize.val,
                fsize.unit,
                info.be_name,
                info.n_layers,
                info.n_embed,
                info.ff_dim,
                info.n_heads,
                info.n_kv_heads,
                info.head_dim,
                info.load_ms,
                info.warmup_ms,
            }) catch return
        else
            std.fmt.bufPrint(&buf, "agave {s} \xc2\xb7 {s} \xc2\xb7 {s} \xc2\xb7 {d:.1}{s} \xc2\xb7 {s}\n", .{
                info.name,
                info.arch_name,
                info.quant,
                fsize.val,
                fsize.unit,
                info.be_name,
            }) catch return;
        stderr.writeAll(text) catch {};
    }

    /// TTY banner with box drawing characters and ANSI colors.
    /// Compact 3-line box: model name, quant/size/arch, backend/timing.
    pub fn printBannerTty(_: Display, info: ModelInfo) void {
        const fsize = formatSize(info.file_size_bytes);

        const ctl = vaxis.ctlseqs;
        const cyan = comptime std.fmt.comptimePrint(ctl.fg_base, .{6});
        const bold = ctl.bold_set;
        const dim = ctl.dim_set;
        const reset = ctl.sgr_reset;

        // Box drawing
        const tl = "\xe2\x95\xad"; // ╭
        const tr = "\xe2\x95\xae"; // ╮
        const bl = "\xe2\x95\xb0"; // ╰
        const br = "\xe2\x95\xaf"; // ╯
        const hz = "\xe2\x94\x80"; // ─
        const vt = "\xe2\x94\x82"; // │

        // Build content lines
        const max_content_lines = 6;
        var line_bufs: [max_content_lines][256]u8 = undefined;
        var lines: [max_content_lines][]const u8 = undefined;
        var n_lines: usize = 0;

        // Line 0: model name
        lines[n_lines] = info.name;
        n_lines += 1;

        // Line 1: arch · quant · params · size · bpw
        if (info.n_params > 0) {
            var pb: [16]u8 = undefined;
            const ps = fmtCompact(&pb, info.n_params);
            const bpw: f32 = @as(f32, @floatFromInt(info.file_size_bytes)) * bits_per_byte / @as(f32, @floatFromInt(info.n_params));
            lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{s} \xc2\xb7 {s} \xc2\xb7 {s} \xc2\xb7 {d:.1}{s} \xc2\xb7 {d:.1} bpw", .{
                info.arch_name, info.quant, ps, fsize.val, fsize.unit, bpw,
            }) catch "";
        } else {
            lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{s} \xc2\xb7 {s} \xc2\xb7 {d:.1}{s}", .{
                info.arch_name, info.quant, fsize.val, fsize.unit,
            }) catch "";
        }
        n_lines += 1;

        // Line 2: layers · embed [· ffn] · heads · head_dim [· experts]
        {
            const has_ff = info.ff_dim > 0;
            const has_exp = info.n_experts > 0;
            if (has_ff and has_exp) {
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{d}L \xc2\xb7 {d}E \xc2\xb7 {d}FFN \xc2\xb7 {d}h/{d}kv \xc2\xb7 hd{d} \xc2\xb7 {d}/{d}exp", .{
                    info.n_layers, info.n_embed, info.ff_dim, info.n_heads, info.n_kv_heads, info.head_dim, info.n_experts_used, info.n_experts,
                }) catch "";
            } else if (has_ff) {
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{d}L \xc2\xb7 {d}E \xc2\xb7 {d}FFN \xc2\xb7 {d}h/{d}kv \xc2\xb7 hd{d}", .{
                    info.n_layers, info.n_embed, info.ff_dim, info.n_heads, info.n_kv_heads, info.head_dim,
                }) catch "";
            } else if (has_exp) {
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{d}L \xc2\xb7 {d}E \xc2\xb7 {d}h/{d}kv \xc2\xb7 hd{d} \xc2\xb7 {d}/{d}exp", .{
                    info.n_layers, info.n_embed, info.n_heads, info.n_kv_heads, info.head_dim, info.n_experts_used, info.n_experts,
                }) catch "";
            } else {
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{d}L \xc2\xb7 {d}E \xc2\xb7 {d}h/{d}kv \xc2\xb7 hd{d}", .{
                    info.n_layers, info.n_embed, info.n_heads, info.n_kv_heads, info.head_dim,
                }) catch "";
            }
        }
        n_lines += 1;

        // Line 3: vocab · ctx · θ (only if any value is present)
        {
            var p: usize = 0;
            const sep: []const u8 = " \xc2\xb7 ";
            if (info.vocab_size > 0) {
                var nb: [16]u8 = undefined;
                const ns = fmtCompact(&nb, info.vocab_size);
                const s = std.fmt.bufPrint(line_bufs[n_lines][p..], "{s} vocab", .{ns}) catch "";
                p += s.len;
            }
            if (info.ctx_size > 0) {
                if (p > 0) {
                    @memcpy(line_bufs[n_lines][p..][0..sep.len], sep);
                    p += sep.len;
                }
                var nb: [16]u8 = undefined;
                const ns = fmtCompact(&nb, info.ctx_size);
                const s = std.fmt.bufPrint(line_bufs[n_lines][p..], "{s} ctx", .{ns}) catch "";
                p += s.len;
            }
            if (info.rope_theta > 0) {
                if (p > 0) {
                    @memcpy(line_bufs[n_lines][p..][0..sep.len], sep);
                    p += sep.len;
                }
                var nb: [16]u8 = undefined;
                const ns = fmtCompact(&nb, @as(u64, @intFromFloat(info.rope_theta)));
                const s = std.fmt.bufPrint(line_bufs[n_lines][p..], "\xce\xb8 {s}", .{ns}) catch "";
                p += s.len;
            }
            if (p > 0) {
                lines[n_lines] = line_bufs[n_lines][0..p];
                n_lines += 1;
            }
        }

        // Line 4: backend · timing
        {
            if (info.warmup_ms > 0) {
                var db: [16]u8 = undefined;
                const ds = fmtDuration(&db, info.warmup_ms);
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{s} \xc2\xb7 {d}ms load + {s} warmup", .{
                    info.be_name, info.load_ms, ds,
                }) catch "";
            } else {
                lines[n_lines] = std.fmt.bufPrint(&line_bufs[n_lines], "{s} \xc2\xb7 loaded in {d}ms", .{
                    info.be_name, info.load_ms,
                }) catch "";
            }
            n_lines += 1;
        }

        // Find max display width, capped to terminal width
        var max_width: usize = 0;
        for (lines[0..n_lines]) |line| {
            const w = displayWidth(line);
            if (w > max_width) max_width = w;
        }

        // Query terminal width, default to 80 if unavailable
        const term_cols: usize = blk: {
            var ws: std.posix.winsize = undefined;
            const rc = std.posix.system.ioctl(stderr.handle, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
            if (rc == 0 and ws.col > 0) break :blk @as(usize, ws.col);
            break :blk 80;
        };
        if (max_width + 4 > term_cols) max_width = term_cols - 4;

        const box_w = max_width + 2; // 1 char padding each side

        var out_buf: [out_buf_size]u8 = undefined;
        var out_pos: usize = 0;

        const append = struct {
            fn f(buf: *[out_buf_size]u8, pos: *usize, s: []const u8) void {
                const n = @min(s.len, buf.len - pos.*);
                @memcpy(buf[pos.*..][0..n], s[0..n]);
                pos.* += n;
            }
        }.f;

        // ╭─ 🌵 agave v0.1.0 ──────╮
        const green = comptime std.fmt.comptimePrint(ctl.fg_base, .{2});
        const cactus = "\xf0\x9f\x8c\xb5"; // 🌵
        const title = " agave v" ++ version ++ " ";
        const title_w = displayWidth(cactus) + displayWidth(title); // 🌵 + title text
        const rule_after = if (box_w > title_w + 1) box_w - title_w - 1 else 0;

        append(&out_buf, &out_pos, cyan);
        append(&out_buf, &out_pos, tl);
        append(&out_buf, &out_pos, hz);
        append(&out_buf, &out_pos, reset);
        append(&out_buf, &out_pos, green);
        append(&out_buf, &out_pos, cactus);
        append(&out_buf, &out_pos, reset);
        append(&out_buf, &out_pos, bold ++ cyan);
        append(&out_buf, &out_pos, title);
        append(&out_buf, &out_pos, reset ++ cyan);
        for (0..rule_after) |_| append(&out_buf, &out_pos, hz);
        append(&out_buf, &out_pos, tr);
        append(&out_buf, &out_pos, reset);
        append(&out_buf, &out_pos, "\n");

        // Content lines
        for (lines[0..n_lines], 0..) |line, i| {
            const lw = displayWidth(line);
            const cw = @min(lw, max_width);
            const pad = box_w - cw;

            const content = if (lw > max_width) truncateToWidth(line, max_width) else line;

            append(&out_buf, &out_pos, cyan);
            append(&out_buf, &out_pos, vt);
            append(&out_buf, &out_pos, reset);
            append(&out_buf, &out_pos, " ");

            if (i == 0) {
                append(&out_buf, &out_pos, bold);
            } else {
                append(&out_buf, &out_pos, dim);
            }
            append(&out_buf, &out_pos, content);
            append(&out_buf, &out_pos, reset);

            for (0..pad - 1) |_| append(&out_buf, &out_pos, " ");

            append(&out_buf, &out_pos, cyan);
            append(&out_buf, &out_pos, vt);
            append(&out_buf, &out_pos, reset);
            append(&out_buf, &out_pos, "\n");
        }

        // ╰──────────────────────╯
        append(&out_buf, &out_pos, cyan);
        append(&out_buf, &out_pos, bl);
        for (0..box_w) |_| append(&out_buf, &out_pos, hz);
        append(&out_buf, &out_pos, br);
        append(&out_buf, &out_pos, reset);
        append(&out_buf, &out_pos, "\n");

        stderr.writeAll(out_buf[0..out_pos]) catch {};
    }

    // ── System Info ──────────────────────────────────────────

    /// Print system/backend information after the banner.
    /// Shows platform, device, VRAM, loaded libraries, kernel count, and threads.
    pub fn printSystemInfo(self: Display, info: BackendInfo) void {
        if (self.mode == .json) return;
        const ctl = vaxis.ctlseqs;
        const is_tty = self.mode == .tty;
        var buf: [out_buf_size]u8 = undefined;
        var pos: usize = 0;
        const w = struct {
            fn f(b: []u8, p: *usize, comptime fmt_str: []const u8, args: anytype) void {
                const s = std.fmt.bufPrint(b[p.*..], fmt_str, args) catch return;
                p.* += s.len;
            }
        }.f;

        if (is_tty) w(&buf, &pos, "{s}", .{ctl.dim_set});
        w(&buf, &pos, "system: {s}/{s}", .{ info.arch, info.os });
        w(&buf, &pos, " \xc2\xb7 {s}", .{info.name});
        if (info.device_name.len > 0) w(&buf, &pos, " \xc2\xb7 {s}", .{info.device_name});
        if (info.compute_cap.len > 0) w(&buf, &pos, " ({s})", .{info.compute_cap});
        if (info.total_mem > 0) {
            const total = formatSize(info.total_mem);
            const is_gpu = info.n_gpu_kernels > 0;
            const label: []const u8 = if (info.is_uma) "unified" else if (is_gpu) "VRAM" else "RAM";
            // Show "avail/total" when available differs meaningfully from total (>2%)
            if (info.avail_mem > 0 and info.avail_mem < info.total_mem * avail_display_pct / 100) {
                const avail = formatSize(info.avail_mem);
                w(&buf, &pos, " \xc2\xb7 {d:.1}/{d:.1}{s} {s}", .{ avail.val, total.val, total.unit, label });
            } else {
                w(&buf, &pos, " \xc2\xb7 {d:.1}{s} {s}", .{ total.val, total.unit, label });
            }
        }
        // Show system RAM separately for discrete GPUs (VRAM differs from system RAM)
        if (info.n_gpu_kernels > 0 and !info.is_uma and info.system_mem > 0) {
            const sys_total = formatSize(info.system_mem);
            if (info.system_avail > 0 and info.system_avail < info.system_mem * avail_display_pct / 100) {
                const sys_avail = formatSize(info.system_avail);
                w(&buf, &pos, " \xc2\xb7 {d:.1}/{d:.1}{s} RAM", .{ sys_avail.val, sys_total.val, sys_total.unit });
            } else {
                w(&buf, &pos, " \xc2\xb7 {d:.1}{s} RAM", .{ sys_total.val, sys_total.unit });
            }
        }
        // CPU cache sizes
        if (info.l2_cache > 0) {
            if (info.l1_cache > 0) {
                const l1 = formatSize(info.l1_cache);
                w(&buf, &pos, " \xc2\xb7 L1 {d:.0}{s}", .{ l1.val, l1.unit });
            }
            const l2 = formatSize(info.l2_cache);
            w(&buf, &pos, " \xc2\xb7 L2 {d:.0}{s}", .{ l2.val, l2.unit });
            if (info.l3_cache > 0) {
                const l3 = formatSize(info.l3_cache);
                w(&buf, &pos, " \xc2\xb7 L3 {d:.0}{s}", .{ l3.val, l3.unit });
            }
        }
        if (info.lib_name.len > 0) w(&buf, &pos, " \xc2\xb7 {s}", .{info.lib_name});
        if (info.driver_version.len > 0) w(&buf, &pos, " \xc2\xb7 {s}", .{info.driver_version});
        if (info.n_gpu_kernels > 0) w(&buf, &pos, " \xc2\xb7 {d} {s} kernels", .{ info.n_gpu_kernels, info.kernel_type });
        if (info.n_threads > 0) w(&buf, &pos, " \xc2\xb7 {d} threads", .{info.n_threads});
        if (is_tty) w(&buf, &pos, "{s}", .{ctl.sgr_reset});
        w(&buf, &pos, "\n", .{});

        stderr.writeAll(buf[0..pos]) catch {};
    }

    // ── Load Info ────────────────────────────────────────────

    /// Print technical loading details (tensors, tokenizer, template, init time).
    pub fn printLoadInfo(self: Display, info: LoadInfo) void {
        if (self.mode == .json) return;
        const ctl = vaxis.ctlseqs;
        const is_tty = self.mode == .tty;
        var buf: [out_buf_size]u8 = undefined;
        var pos: usize = 0;
        const w = struct {
            fn f(b: []u8, p: *usize, comptime fmt_str: []const u8, args: anytype) void {
                const s = std.fmt.bufPrint(b[p.*..], fmt_str, args) catch return;
                p.* += s.len;
            }
        }.f;

        if (is_tty) w(&buf, &pos, "{s}", .{ctl.dim_set});
        w(&buf, &pos, "loaded:", .{});
        if (info.format_name.len > 0) w(&buf, &pos, " {s}", .{info.format_name});
        if (info.n_tensors > 0) w(&buf, &pos, " \xc2\xb7 {d} tensors", .{info.n_tensors});
        if (info.tok_kind.len > 0) w(&buf, &pos, " \xc2\xb7 {s} tokenizer", .{info.tok_kind});
        if (info.vocab_size > 0) {
            var nb: [16]u8 = undefined;
            const ns = fmtCompact(&nb, info.vocab_size);
            w(&buf, &pos, " \xc2\xb7 {s} vocab", .{ns});
        }
        w(&buf, &pos, " \xc2\xb7 eos={d} bos={d}", .{ info.eos_id, info.bos_id });
        if (info.n_eog > 1) w(&buf, &pos, " (+{d} eog)", .{info.n_eog - 1});
        if (info.template_name.len > 0) w(&buf, &pos, " \xc2\xb7 {s} template", .{info.template_name});
        if (info.init_ms > 0) w(&buf, &pos, " \xc2\xb7 init {d}ms", .{info.init_ms});
        if (is_tty) w(&buf, &pos, "{s}", .{ctl.sgr_reset});
        w(&buf, &pos, "\n", .{});

        stderr.writeAll(buf[0..pos]) catch {};
    }

    // ── Prefill Progress ─────────────────────────────────────

    /// Show a prefill-in-progress message on stderr.
    pub fn showPrefillStart(self: Display, token_count: usize) void {
        if (self.mode == .json) return;
        var buf: [out_buf_size]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "prefill {d} tokens...", .{token_count}) catch return;
        stderr.writeAll(text) catch {};
    }

    /// Clear the prefill progress message (carriage return + erase line).
    pub fn clearPrefillProgress(_: Display) void {
        stderr.writeAll("\r" ++ erase_line) catch {};
    }

    // ── Generation Progress ──────────────────────────────────

    /// Update the generation progress display with a live progress bar.
    /// Uses Unicode block characters to show progress visually in TTY mode.
    pub fn updateProgress(self: Display, tokens_generated: u32, max_tokens: u32, elapsed_ms: u64) void {
        if (self.mode != .tty) return;
        const ctl = vaxis.ctlseqs;
        const tps: f32 = if (elapsed_ms > 0)
            @as(f32, @floatFromInt(tokens_generated)) / (@as(f32, @floatFromInt(elapsed_ms)) / 1000.0)
        else
            0.0;

        const bar_width: u32 = 30;
        const filled: u32 = if (max_tokens > 0)
            @intCast(@min(@as(u64, tokens_generated) * bar_width / max_tokens, bar_width))
        else
            0;

        var buf: [256]u8 = undefined;
        var pos: usize = 0;
        const append = struct {
            fn f(b: []u8, p: *usize, s: []const u8) void {
                const n = @min(s.len, b.len - p.*);
                @memcpy(b[p.*..][0..n], s[0..n]);
                p.* += n;
            }
        }.f;

        append(&buf, &pos, "\r" ++ erase_line ++ ctl.dim_set ++ "\xe2\x96\x90"); // CR, clear, dim, ▐
        for (0..bar_width) |i| {
            if (i < filled) {
                append(&buf, &pos, "\xe2\x96\x88"); // █
            } else {
                append(&buf, &pos, "\xe2\x96\x91"); // ░
            }
        }
        append(&buf, &pos, "\xe2\x96\x8c "); // ▌ + space

        // Append text stats
        const text = std.fmt.bufPrint(buf[pos..], "{d}/{d} tok \xc2\xb7 {d:.1} tok/s" ++ ctl.sgr_reset, .{
            tokens_generated, max_tokens, tps,
        }) catch "";
        pos += text.len;

        stderr.writeAll(buf[0..pos]) catch {};
    }

    /// Clear the progress line.
    pub fn clearProgress(_: Display) void {
        stderr.writeAll("\r" ++ erase_line) catch {};
    }

    // ── Stats ────────────────────────────────────────────────

    /// Print generation statistics, dispatching to TTY or plain format.
    pub fn printStats(self: Display, stats: GenStats) void {
        switch (self.mode) {
            .tty => self.printStatsTty(stats),
            .plain => self.printStatsPlain(stats),
            .json => {}, // JSON stats are printed via printJsonPrompt
        }
    }

    /// Plain-text stats written to stderr.
    pub fn printStatsPlain(self: Display, stats: GenStats) void {
        var buf: [out_buf_size]u8 = undefined;
        const text = if (self.verbose)
            std.fmt.bufPrint(&buf, "{d} tok \xc2\xb7 {d:.1} tok/s \xc2\xb7 prefill {d} tok in {d}ms ({d:.0} tok/s)\n", .{
                stats.token_count,
                stats.tokPerSec(),
                stats.prefill_token_count,
                stats.prefill_ms,
                stats.prefillTokPerSec(),
            }) catch return
        else
            std.fmt.bufPrint(&buf, "{d} tok \xc2\xb7 {d:.1} tok/s \xc2\xb7 {d}ms prefill\n", .{
                stats.token_count,
                stats.tokPerSec(),
                stats.prefill_ms,
            }) catch return;
        stderr.writeAll(text) catch {};
    }

    /// TTY stats display (currently delegates to plain-text output).
    pub fn printStatsTty(self: Display, stats: GenStats) void {
        self.printStatsPlain(stats);
    }

    // ── JSON Output ──────────────────────────────────────────

    /// Print full JSON output for a prompt response (model info + generated text + stats).
    pub fn printJsonPrompt(_: Display, info: ModelInfo, output_text: []const u8, stats: GenStats) void {
        var buf: [out_buf_size]u8 = undefined;
        var writer = std.io.Writer.fixed(&buf);
        var jw: std.json.Stringify = .{ .writer = &writer };
        jw.beginObject() catch return;

        // Model info
        jw.objectField("model") catch return;
        jw.write(info.name) catch return;
        jw.objectField("arch") catch return;
        jw.write(info.arch_name) catch return;
        jw.objectField("quant") catch return;
        jw.write(info.quant) catch return;
        jw.objectField("backend") catch return;
        jw.write(info.be_name) catch return;

        // Output
        jw.objectField("output") catch return;
        jw.write(output_text) catch return;

        // Stats
        jw.objectField("tokens") catch return;
        jw.write(stats.token_count) catch return;
        jw.objectField("tok_per_sec") catch return;
        jw.write(stats.tokPerSec()) catch return;
        jw.objectField("prefill_tokens") catch return;
        jw.write(stats.prefill_token_count) catch return;
        jw.objectField("prefill_ms") catch return;
        jw.write(stats.prefill_ms) catch return;
        jw.objectField("gen_ms") catch return;
        jw.write(stats.gen_ms) catch return;

        jw.endObject() catch return;

        const written = writer.buffer[0..writer.end];
        stdout.writeAll(written) catch {};
        stdout.writeAll("\n") catch {};
    }

    /// Print JSON model info (for --model-info --json).
    pub fn printJsonModelInfo(_: Display, info: ModelInfo) void {
        const fsize = formatSize(info.file_size_bytes);
        var buf: [out_buf_size]u8 = undefined;
        var writer = std.io.Writer.fixed(&buf);
        var jw: std.json.Stringify = .{ .writer = &writer };
        jw.beginObject() catch return;

        jw.objectField("name") catch return;
        jw.write(info.name) catch return;
        jw.objectField("arch") catch return;
        jw.write(info.arch_name) catch return;
        jw.objectField("quant") catch return;
        jw.write(info.quant) catch return;
        jw.objectField("backend") catch return;
        jw.write(info.be_name) catch return;
        jw.objectField("layers") catch return;
        jw.write(info.n_layers) catch return;
        jw.objectField("embed") catch return;
        jw.write(info.n_embed) catch return;
        jw.objectField("heads") catch return;
        jw.write(info.n_heads) catch return;
        jw.objectField("kv_heads") catch return;
        jw.write(info.n_kv_heads) catch return;
        jw.objectField("head_dim") catch return;
        jw.write(info.head_dim) catch return;
        jw.objectField("ff_dim") catch return;
        jw.write(info.ff_dim) catch return;
        jw.objectField("vocab_size") catch return;
        jw.write(info.vocab_size) catch return;
        jw.objectField("ctx_size") catch return;
        jw.write(info.ctx_size) catch return;
        jw.objectField("rope_theta") catch return;
        jw.write(info.rope_theta) catch return;
        jw.objectField("n_params") catch return;
        jw.write(info.n_params) catch return;
        if (info.n_params > 0) {
            jw.objectField("bpw") catch return;
            jw.write(@as(f32, @floatFromInt(info.file_size_bytes)) * bits_per_byte / @as(f32, @floatFromInt(info.n_params))) catch return;
        }
        if (info.n_experts > 0) {
            jw.objectField("n_experts") catch return;
            jw.write(info.n_experts) catch return;
            jw.objectField("n_experts_used") catch return;
            jw.write(info.n_experts_used) catch return;
        }
        jw.objectField("file_size") catch return;
        jw.write(info.file_size_bytes) catch return;
        jw.objectField("file_size_human") catch return;
        // Format as "1.2GB"
        {
            var sbuf: [32]u8 = undefined;
            const stxt = std.fmt.bufPrint(&sbuf, "{d:.1}{s}", .{ fsize.val, fsize.unit }) catch "?";
            jw.write(stxt) catch return;
        }
        jw.objectField("load_ms") catch return;
        jw.write(info.load_ms) catch return;
        jw.objectField("warmup_ms") catch return;
        jw.write(info.warmup_ms) catch return;

        jw.endObject() catch return;

        const written = writer.buffer[0..writer.end];
        stdout.writeAll(written) catch {};
        stdout.writeAll("\n") catch {};
    }

    /// Print JSON benchmark results.
    pub fn printJsonBench(_: Display, be_name: []const u8, results: []const BenchResult) void {
        var buf: [out_buf_size]u8 = undefined;
        var writer = std.io.Writer.fixed(&buf);
        var jw: std.json.Stringify = .{ .writer = &writer };
        jw.beginObject() catch return;

        jw.objectField("backend") catch return;
        jw.write(be_name) catch return;

        jw.objectField("results") catch return;
        jw.beginArray() catch return;
        for (results) |r| {
            jw.beginObject() catch return;
            jw.objectField("op") catch return;
            jw.write(r.op) catch return;
            jw.objectField("dimensions") catch return;
            jw.write(r.dimensions) catch return;
            jw.objectField("ms") catch return;
            jw.write(r.ms) catch return;
            jw.objectField("gbps") catch return;
            jw.write(r.gbps) catch return;
            jw.endObject() catch return;
        }
        jw.endArray() catch return;

        jw.endObject() catch return;

        const written = writer.buffer[0..writer.end];
        stdout.writeAll(written) catch {};
        stdout.writeAll("\n") catch {};
    }

    // ── Human-Readable Model Info ────────────────────────────

    /// Print human-readable model information (for --model-info and REPL /model).
    pub fn printModelInfo(_: Display, info: ModelInfo) void {
        const fsize = formatSize(info.file_size_bytes);
        var buf: [out_buf_size]u8 = undefined;
        var pos: usize = 0;
        const w = struct {
            fn f(b: []u8, p: *usize, comptime fmt: []const u8, args: anytype) void {
                const s = std.fmt.bufPrint(b[p.*..], fmt, args) catch return;
                p.* += s.len;
            }
        }.f;

        w(&buf, &pos, "  Model:    {s}\n", .{info.name});
        w(&buf, &pos, "  Arch:     {s}\n", .{info.arch_name});
        w(&buf, &pos, "  Quant:    {s}\n", .{info.quant});
        if (info.n_params > 0) {
            var pb: [16]u8 = undefined;
            const ps = fmtCompact(&pb, info.n_params);
            const bpw: f32 = @as(f32, @floatFromInt(info.file_size_bytes)) * bits_per_byte / @as(f32, @floatFromInt(info.n_params));
            w(&buf, &pos, "  Params:   {d} ({s}, {d:.2} bpw)\n", .{ info.n_params, ps, bpw });
        }
        w(&buf, &pos, "  Backend:  {s}\n", .{info.be_name});
        w(&buf, &pos, "  Layers:   {d}\n", .{info.n_layers});
        w(&buf, &pos, "  Embed:    {d}\n", .{info.n_embed});
        if (info.ff_dim > 0) w(&buf, &pos, "  FFN:      {d}\n", .{info.ff_dim});
        w(&buf, &pos, "  Heads:    {d} ({d} KV, GQA {d}:1)\n", .{ info.n_heads, info.n_kv_heads, if (info.n_kv_heads > 0) info.n_heads / info.n_kv_heads else 0 });
        w(&buf, &pos, "  Head dim: {d}\n", .{info.head_dim});
        if (info.n_experts > 0) w(&buf, &pos, "  Experts:  {d} used / {d} total\n", .{ info.n_experts_used, info.n_experts });
        if (info.vocab_size > 0) w(&buf, &pos, "  Vocab:    {d}\n", .{info.vocab_size});
        if (info.ctx_size > 0) w(&buf, &pos, "  Context:  {d}\n", .{info.ctx_size});
        if (info.rope_theta > 0) w(&buf, &pos, "  RoPE:     {d}\n", .{@as(u64, @intFromFloat(info.rope_theta))});
        w(&buf, &pos, "  Size:     {d:.1} {s}\n", .{ fsize.val, fsize.unit });
        w(&buf, &pos, "  Loaded:   {d}ms\n", .{info.load_ms});
        if (info.warmup_ms > 0) w(&buf, &pos, "  Warmup:   {d}ms\n", .{info.warmup_ms});

        stdout.writeAll(buf[0..pos]) catch {};
    }
};

// ── Tests ────────────────────────────────────────────────────────

test "GenStats tok/s calculation" {
    const stats = GenStats{
        .token_count = 100,
        .gen_ms = 2000,
        .prefill_token_count = 50,
        .prefill_ms = 500,
    };
    // 100 tokens / 2.0 seconds = 50.0 tok/s
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), stats.tokPerSec(), 0.01);
    // 50 tokens / 0.5 seconds = 100.0 tok/s
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), stats.prefillTokPerSec(), 0.01);
}

test "GenStats zero ms" {
    const stats = GenStats{
        .token_count = 10,
        .gen_ms = 0,
        .prefill_token_count = 5,
        .prefill_ms = 0,
    };
    try std.testing.expectEqual(@as(f32, 0.0), stats.tokPerSec());
    try std.testing.expectEqual(@as(f32, 0.0), stats.prefillTokPerSec());
}

test "formatSize" {
    const kb = formatSize(512);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), kb.val, 0.01);
    try std.testing.expectEqualStrings("KB", kb.unit);

    const mb = formatSize(5 * 1024 * 1024);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), mb.val, 0.01);
    try std.testing.expectEqualStrings("MB", mb.unit);

    const gb = formatSize(3 * 1024 * 1024 * 1024);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), gb.val, 0.01);
    try std.testing.expectEqualStrings("GB", gb.unit);
}

test "displayWidth ascii" {
    try std.testing.expectEqual(@as(usize, 5), displayWidth("hello"));
}

test "displayWidth middot" {
    // "a · b" — the middot (U+00B7) is 2 bytes in UTF-8 but occupies 1 terminal column
    try std.testing.expectEqual(@as(usize, 5), displayWidth("a \xc2\xb7 b"));
}
