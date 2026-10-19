//! `POST /v1/systemone`: typed decision questions answered by a single logit
//! read instead of generated text.
//!
//! One question becomes one prompt: the state, the question's instructions,
//! its options under single-token labels, and an instruction to answer with
//! one label. The answer is the softmax over just those label tokens in the
//! first position's logits, so nothing is sampled and nothing is parsed out of
//! prose. Each read is a prefill; questions sharing a state are read in the
//! order they arrive so a causal model reuses the common prefix.
//!
//! The three question types differ only in how labels map back to values:
//!
//!   * `noul`   yes/no, the answer is P(yes).
//!   * `choice` labels A, B, C ... in request order, the answer is the
//!              highest-probability option's name.
//!   * `score`  labels 0, 1, 2 ... lowest level first, the answer is the
//!              probability-weighted mean of the level indices, so it lands
//!              between levels rather than snapping to one.
//!
//! This is the logit-read backend, the same one vLLM ships. A model trained
//! with a decision head would answer in one pass for all questions; agave has
//! no such head, so the cost is one prefill per question and the caller sees
//! `output_tokens: 0` either way.

const std = @import("std");
const json = @import("json.zig");

/// Upper bounds, matching the MVP limits the other implementations enforce.
/// A request past any of them is a 400 rather than a slow success.
pub const max_questions: usize = 64;
pub const max_choices: usize = 26;
pub const max_score_levels: usize = 10;
pub const min_choices: usize = 2;
pub const min_score_levels: usize = 2;
/// Longest question name echoed back in `answers`.
pub const max_name_len: usize = 64;
/// Longest option label text (the value for a choice, the level for a score).
pub const max_option_len: usize = 128;

/// Labels a choice question's options take, in request order. Each has to be
/// a single token for the read to work, which rules out longer alphabets.
const choice_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
/// Labels a score question's levels take, lowest first.
const score_labels = "0123456789";
/// Labels a noul question's two outcomes. Index 0 is the true case.
const noul_labels = [_][]const u8{ "yes", "no" };

pub const QuestionType = enum { noul, choice, score };

pub const ParseError = error{
    /// Body is not an object, or `questions` is missing or not an object.
    MalformedRequest,
    /// Zero questions, or more than `max_questions`.
    QuestionCount,
    /// A question's `type` is absent or not one of the three.
    UnknownQuestionType,
    /// A choice outside [min_choices, max_choices], or a score outside
    /// [min_score_levels, max_score_levels].
    OptionCount,
    /// Two options in one question share a value. Their probabilities would
    /// be indistinguishable in the answer.
    DuplicateOption,
    /// A name or option longer than this module stores.
    TooLong,
    OutOfMemory,
};

/// One option of a choice or score question: the text the caller used, and
/// the description that goes in the prompt beside it.
pub const Option = struct {
    value: []const u8,
    description: []const u8 = "",
};

pub const Question = struct {
    name: []const u8,
    kind: QuestionType,
    instructions: []const u8 = "",
    /// Choice options or score levels, in request order. Empty for `noul`,
    /// whose two outcomes are fixed.
    options: []Option = &.{},

    /// Number of labels this question reads, which is the width of its
    /// probability vector.
    pub fn labelCount(self: Question) usize {
        return switch (self.kind) {
            .noul => noul_labels.len,
            .choice, .score => self.options.len,
        };
    }

    /// Text of label `i`, as it appears in the prompt and is matched against
    /// the tokenizer.
    pub fn label(self: Question, i: usize) []const u8 {
        return switch (self.kind) {
            .noul => noul_labels[i],
            .choice => choice_labels[i .. i + 1],
            .score => score_labels[i .. i + 1],
        };
    }
};

pub const Request = struct {
    /// The thing being judged. Raw JSON when the caller sent an object or
    /// array, so a model sees the structure rather than a flattened string.
    state: []const u8,
    questions: []Question,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Request) void {
        self.arena.deinit();
    }
};

/// Parse a `/v1/systemone` body. Every slice in the result is owned by the
/// returned arena, so the caller frees once with `deinit`.
///
/// `body` is untrusted. Every count, length and duplicate check happens here
/// so the read path downstream can work from the types.
pub fn parseRequest(allocator: std.mem.Allocator, body: []const u8) ParseError!Request {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();

    const questions_obj = json.extractObjectField(body, "questions") orelse return error.MalformedRequest;
    if (questions_obj.len == 0 or questions_obj[0] != '{') return error.MalformedRequest;

    const state = try dupeState(a, body);

    var list = try std.ArrayList(Question).initCapacity(a, 0);
    var it = json.ObjectIterator.init(questions_obj);
    while (it.next()) |member| {
        if (list.items.len >= max_questions) return error.QuestionCount;
        if (member.key.len > max_name_len) return error.TooLong;
        if (member.value.len == 0 or member.value[0] != '{') return error.MalformedRequest;
        try list.append(a, try parseQuestion(a, member.key, member.value));
    }
    if (list.items.len == 0) return error.QuestionCount;

    return .{ .state = state, .questions = try list.toOwnedSlice(a), .arena = arena };
}

/// Copy the `state` field, keeping an object or array as raw JSON and
/// unwrapping a string to its text.
fn dupeState(a: std.mem.Allocator, body: []const u8) ParseError![]const u8 {
    if (json.extractObjectField(body, "state")) |structured|
        return a.dupe(u8, structured);
    const text = json.extractField(body, "state") orelse return error.MalformedRequest;
    return json.jsonUnescape(a, text) catch error.OutOfMemory;
}

fn parseQuestion(a: std.mem.Allocator, name: []const u8, obj: []const u8) ParseError!Question {
    const type_text = json.extractField(obj, "type") orelse return error.UnknownQuestionType;
    const kind: QuestionType = if (std.mem.eql(u8, type_text, "noul"))
        .noul
    else if (std.mem.eql(u8, type_text, "choice"))
        .choice
    else if (std.mem.eql(u8, type_text, "score"))
        .score
    else
        return error.UnknownQuestionType;

    const instructions = if (json.extractField(obj, "instructions")) |raw|
        json.jsonUnescape(a, raw) catch return error.OutOfMemory
    else
        "";

    var q = Question{
        .name = try a.dupe(u8, name),
        .kind = kind,
        .instructions = instructions,
    };
    if (kind == .noul) return q;

    const criteria = json.extractObjectField(obj, "criteria") orelse return error.OptionCount;
    q.options = switch (kind) {
        .choice => try parseChoiceCriteria(a, criteria),
        .score => try parseScoreCriteria(a, criteria),
        .noul => unreachable,
    };
    const n = q.options.len;
    const lo: usize = if (kind == .choice) min_choices else min_score_levels;
    const hi: usize = if (kind == .choice) max_choices else max_score_levels;
    if (n < lo or n > hi) return error.OptionCount;
    try rejectDuplicates(q.options);
    return q;
}

/// `criteria` for a choice is an object mapping label to description. Keys
/// are the values the answer reports, so their order is the label order.
fn parseChoiceCriteria(a: std.mem.Allocator, criteria: []const u8) ParseError![]Option {
    if (criteria.len == 0 or criteria[0] != '{') return error.OptionCount;
    var list = try std.ArrayList(Option).initCapacity(a, 0);
    var it = json.ObjectIterator.init(criteria);
    while (it.next()) |member| {
        if (list.items.len >= max_choices) return error.OptionCount;
        if (member.key.len > max_option_len) return error.TooLong;
        try list.append(a, .{
            .value = try a.dupe(u8, member.key),
            .description = try dupeJsonString(a, member.value),
        });
    }
    return list.toOwnedSlice(a) catch error.OutOfMemory;
}

/// `criteria` for a score is an array of levels, lowest first. An element is
/// either a bare string or an object with `label` and optional `description`.
fn parseScoreCriteria(a: std.mem.Allocator, criteria: []const u8) ParseError![]Option {
    if (criteria.len == 0 or criteria[0] != '[') return error.OptionCount;
    var list = try std.ArrayList(Option).initCapacity(a, 0);
    var it = json.ArrayIterator.init(criteria);
    while (it.next()) |element| {
        if (list.items.len >= max_score_levels) return error.OptionCount;
        if (element.len > 0 and element[0] == '{') {
            const label_text = json.extractField(element, "label") orelse return error.MalformedRequest;
            if (label_text.len > max_option_len) return error.TooLong;
            try list.append(a, .{
                .value = json.jsonUnescape(a, label_text) catch return error.OutOfMemory,
                .description = if (json.extractField(element, "description")) |d|
                    json.jsonUnescape(a, d) catch return error.OutOfMemory
                else
                    "",
            });
        } else {
            try list.append(a, .{ .value = try dupeJsonString(a, element) });
        }
    }
    return list.toOwnedSlice(a) catch error.OutOfMemory;
}

/// Unescape a raw JSON value that should be a string, yielding "" for
/// anything else so a non-string description does not fail the request.
fn dupeJsonString(a: std.mem.Allocator, raw: []const u8) ParseError![]const u8 {
    if (raw.len < 2 or raw[0] != '"') return "";
    if (raw.len - 2 > max_option_len) return error.TooLong;
    return json.jsonUnescape(a, raw[1 .. raw.len - 1]) catch error.OutOfMemory;
}

/// Two options with the same value would produce two probabilities the caller
/// cannot tell apart, and for a choice the answer could name either.
fn rejectDuplicates(options: []const Option) ParseError!void {
    for (options, 0..) |opt, i| {
        for (options[i + 1 ..]) |other| {
            if (std.mem.eql(u8, opt.value, other.value)) return error.DuplicateOption;
        }
    }
}

/// Render the prompt for one question. Returns the slice of `buf` used, or
/// null when the question does not fit, which the caller reports rather than
/// truncating: a clipped state changes the answer silently.
pub fn renderPrompt(buf: []u8, state: []const u8, q: Question) ?[]const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.writeAll(state) catch return null;
    w.writeAll("\n\n") catch return null;
    if (q.instructions.len > 0) {
        w.writeAll(q.instructions) catch return null;
        w.writeAll("\n") catch return null;
    }
    switch (q.kind) {
        .noul => w.writeAll("Answer yes or no.\nAnswer:") catch return null,
        .choice, .score => {
            for (q.options, 0..) |opt, i| {
                w.writeAll(q.label(i)) catch return null;
                w.writeAll(") ") catch return null;
                w.writeAll(opt.value) catch return null;
                if (opt.description.len > 0) {
                    w.writeAll(": ") catch return null;
                    w.writeAll(opt.description) catch return null;
                }
                w.writeAll("\n") catch return null;
            }
            w.writeAll("Answer with one label.\nAnswer:") catch return null;
        },
    }
    return w.buffered();
}

/// Resolve the single token a model emits for `label` right after `Answer:`.
///
/// Tries the space-prefixed form first: a BPE tokenizer mid-sentence produces
/// ` A`, not `A`, so reading the bare token's logit would read a position the
/// model never reaches. Returns null when neither form is one token, which
/// the caller reports as a 400: the read looks at a single position, so a
/// label spanning two tokens has no probability to report.
pub fn resolveLabelToken(tokenizer: anytype, allocator: std.mem.Allocator, label: []const u8) ?u32 {
    var buf: [max_option_len + 1]u8 = undefined;
    if (label.len + 1 <= buf.len) {
        buf[0] = ' ';
        @memcpy(buf[1 .. label.len + 1], label);
        if (singleToken(tokenizer, allocator, buf[0 .. label.len + 1])) |id| return id;
    }
    return singleToken(tokenizer, allocator, label);
}

fn singleToken(tokenizer: anytype, allocator: std.mem.Allocator, text: []const u8) ?u32 {
    const ids = tokenizer.encode(text) catch return null;
    defer allocator.free(ids);
    return if (ids.len == 1) ids[0] else null;
}

/// Softmax over just the label logits, written into `out`.
///
/// Normalising over the labels rather than the vocabulary is the point: the
/// model's mass on unrelated tokens is not evidence about this question, and
/// including it would make every confidence a function of how chatty the
/// model is. `out` and `label_logits` have the same length.
pub fn labelProbabilities(label_logits: []const f32, out: []f32) void {
    std.debug.assert(label_logits.len == out.len);
    if (out.len == 0) return;
    var max_logit: f32 = label_logits[0];
    for (label_logits[1..]) |v| max_logit = @max(max_logit, v);
    var sum: f32 = 0;
    for (label_logits, out) |v, *o| {
        o.* = @exp(v - max_logit);
        sum += o.*;
    }
    if (sum > 0 and std.math.isFinite(sum)) {
        const inv = 1.0 / sum;
        for (out) |*o| o.* *= inv;
    } else {
        // Every label underflowed or a logit was not finite. A uniform answer
        // is the honest one: the read carries no information.
        const uniform = 1.0 / @as(f32, @floatFromInt(out.len));
        for (out) |*o| o.* = uniform;
    }
}

/// Probability-weighted mean of the level indices, which is what `score`
/// reports. Lands between levels, so a model split between "calm" and
/// "annoyed" scores near 0.5 rather than snapping to one of them.
pub fn expectedScore(probabilities: []const f32) f32 {
    var acc: f32 = 0;
    for (probabilities, 0..) |p, i| acc += p * @as(f32, @floatFromInt(i));
    return acc;
}

/// Index of the most likely label, and how much mass it holds.
pub const Peak = struct { index: usize, probability: f32 };

pub fn peak(probabilities: []const f32) Peak {
    var best: Peak = .{ .index = 0, .probability = if (probabilities.len > 0) probabilities[0] else 0 };
    for (probabilities, 0..) |p, i| {
        if (p > best.probability) best = .{ .index = i, .probability = p };
    }
    return best;
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "parses the three question types with their criteria shapes" {
    const body =
        \\{"state":"the printer is on fire","questions":{
        \\  "urgent":{"type":"noul","instructions":"Is this urgent?"},
        \\  "tier":{"type":"choice","instructions":"Pick a tier.","criteria":{"low":"can wait","high":"page someone"}},
        \\  "mood":{"type":"score","criteria":["calm","annoyed",{"label":"furious","description":"all caps"}]}
        \\}}
    ;
    var req = try parseRequest(std.testing.allocator, body);
    defer req.deinit();

    try std.testing.expectEqualStrings("the printer is on fire", req.state);
    try std.testing.expectEqual(@as(usize, 3), req.questions.len);

    // Question order follows the request, so a causal model reuses the prefix.
    try std.testing.expectEqualStrings("urgent", req.questions[0].name);
    try std.testing.expectEqual(QuestionType.noul, req.questions[0].kind);
    try std.testing.expectEqual(@as(usize, 2), req.questions[0].labelCount());
    try std.testing.expectEqualStrings("yes", req.questions[0].label(0));

    const tier = req.questions[1];
    try std.testing.expectEqual(QuestionType.choice, tier.kind);
    try std.testing.expectEqualStrings("low", tier.options[0].value);
    try std.testing.expectEqualStrings("can wait", tier.options[0].description);
    try std.testing.expectEqualStrings("high", tier.options[1].value);
    try std.testing.expectEqualStrings("A", tier.label(0));
    try std.testing.expectEqualStrings("B", tier.label(1));

    const mood = req.questions[2];
    try std.testing.expectEqual(QuestionType.score, mood.kind);
    try std.testing.expectEqual(@as(usize, 3), mood.options.len);
    try std.testing.expectEqualStrings("calm", mood.options[0].value);
    try std.testing.expectEqualStrings("furious", mood.options[2].value);
    try std.testing.expectEqualStrings("all caps", mood.options[2].description);
    try std.testing.expectEqualStrings("0", mood.label(0));
    try std.testing.expectEqualStrings("2", mood.label(2));
}

test "keeps a structured state as raw JSON" {
    const body =
        \\{"state":{"ticket":12,"body":"help"},"questions":{"q":{"type":"noul"}}}
    ;
    var req = try parseRequest(std.testing.allocator, body);
    defer req.deinit();
    try std.testing.expectEqualStrings("{\"ticket\":12,\"body\":\"help\"}", req.state);
}

test "rejects the bodies that would otherwise answer nonsense" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, want: ParseError }{
        .{ .body = "{\"state\":\"s\"}", .want = error.MalformedRequest },
        .{ .body = "{\"state\":\"s\",\"questions\":{}}", .want = error.QuestionCount },
        .{ .body = "{\"questions\":{\"q\":{\"type\":\"noul\"}}}", .want = error.MalformedRequest },
        .{
            .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"vibes\"}}}",
            .want = error.UnknownQuestionType,
        },
        .{
            .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{\"only\":\"\"}}}}",
            .want = error.OptionCount,
        },
        .{
            .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[\"a\"]}}}",
            .want = error.OptionCount,
        },
        .{
            .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\"}}}",
            .want = error.OptionCount,
        },
        .{
            .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[\"a\",\"a\"]}}}",
            .want = error.DuplicateOption,
        },
    };
    for (cases) |c| {
        try std.testing.expectError(c.want, parseRequest(allocator, c.body));
    }
}

test "rejects more score levels and choices than a label alphabet holds" {
    const allocator = std.testing.allocator;
    // 11 levels, one past the ten digits available as labels.
    const too_many_levels =
        \\{"state":"s","questions":{"q":{"type":"score","criteria":
        \\["a","b","c","d","e","f","g","h","i","j","k"]}}}
    ;
    try std.testing.expectError(error.OptionCount, parseRequest(allocator, too_many_levels));

    // 27 choices, one past the alphabet.
    var buf = try std.ArrayList(u8).initCapacity(allocator, 0);
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{");
    var name_buf: [16]u8 = undefined;
    for (0..27) |i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&name_buf, "\"v{d}\":\"\"", .{i}));
    }
    try buf.appendSlice(allocator, "}}}}");
    try std.testing.expectError(error.OptionCount, parseRequest(allocator, buf.items));
}

test "renderPrompt lists every option under its label" {
    var options = [_]Option{
        .{ .value = "low", .description = "can wait" },
        .{ .value = "high" },
    };
    const q = Question{
        .name = "tier",
        .kind = .choice,
        .instructions = "Pick a tier.",
        .options = &options,
    };
    var buf: [512]u8 = undefined;
    const prompt = renderPrompt(&buf, "printer on fire", q) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, prompt, "printer on fire") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "A) low: can wait") != null);
    // No description, so no trailing colon.
    try std.testing.expect(std.mem.indexOf(u8, prompt, "B) high\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, prompt, "Answer:"));
}

test "renderPrompt refuses to truncate" {
    // A clipped state would change the answer with nothing to show for it.
    const q = Question{ .name = "q", .kind = .noul };
    var small: [8]u8 = undefined;
    try std.testing.expect(renderPrompt(&small, "a state far longer than the buffer", q) == null);
}

test "labelProbabilities normalises over the labels alone" {
    // The same label logits must give the same answer whatever the rest of
    // the vocabulary is doing, which is why the softmax is not over it.
    const logits = [_]f32{ 2.0, 1.0, 0.0 };
    var probs: [3]f32 = undefined;
    labelProbabilities(&logits, &probs);
    var sum: f32 = 0;
    for (probs) |p| sum += p;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
    try std.testing.expect(probs[0] > probs[1]);
    try std.testing.expect(probs[1] > probs[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.665241), probs[0], 1e-4);
}

test "labelProbabilities falls back to uniform rather than NaN" {
    const broken = [_]f32{ std.math.nan(f32), std.math.nan(f32) };
    var probs: [2]f32 = undefined;
    labelProbabilities(&broken, &probs);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), probs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), probs[1], 1e-6);
}

test "expectedScore lands between levels" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), expectedScore(&[_]f32{ 1.0, 0.0, 0.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), expectedScore(&[_]f32{ 0.0, 0.0, 1.0 }), 1e-6);
    // Split between the bottom two levels, so halfway between them.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), expectedScore(&[_]f32{ 0.5, 0.5, 0.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), expectedScore(&[_]f32{ 0.07, 0.91, 0.02 }), 1e-5);
}

test "peak reports the winning label and its mass" {
    const p = peak(&[_]f32{ 0.1, 0.7, 0.2 });
    try std.testing.expectEqual(@as(usize, 1), p.index);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), p.probability, 1e-6);
    // A flat distribution resolves to the first label, not the last.
    try std.testing.expectEqual(@as(usize, 0), peak(&[_]f32{ 0.5, 0.5 }).index);
    try std.testing.expectEqual(@as(usize, 0), peak(&.{}).index);
}

/// Tokenizer stub: every entry in `table` is one token, anything else splits.
const StubTokenizer = struct {
    table: []const struct { text: []const u8, id: u32 },
    allocator: std.mem.Allocator,

    fn encode(self: StubTokenizer, text: []const u8) ![]u32 {
        for (self.table) |e| {
            if (std.mem.eql(u8, e.text, text)) {
                const out = try self.allocator.alloc(u32, 1);
                out[0] = e.id;
                return out;
            }
        }
        // Not in the table: split per byte, which is what a BPE tokenizer
        // does with an unfamiliar string.
        const out = try self.allocator.alloc(u32, @max(text.len, 2));
        @memset(out, 0);
        return out;
    }
};

test "resolveLabelToken prefers the space-prefixed token" {
    const a = std.testing.allocator;
    // Both forms exist. After "Answer:" the model emits " A", so that is the
    // logit the read has to look at.
    const tok = StubTokenizer{
        .allocator = a,
        .table = &.{ .{ .text = " A", .id = 71 }, .{ .text = "A", .id = 32 } },
    };
    try std.testing.expectEqual(@as(?u32, 71), resolveLabelToken(tok, a, "A"));
}

test "resolveLabelToken falls back to the bare token" {
    const a = std.testing.allocator;
    const tok = StubTokenizer{ .allocator = a, .table = &.{.{ .text = "0", .id = 15 }} };
    try std.testing.expectEqual(@as(?u32, 15), resolveLabelToken(tok, a, "0"));
}

test "resolveLabelToken reports a multi-token label rather than guessing" {
    const a = std.testing.allocator;
    // A label the tokenizer splits has no single position to read, so the
    // request has to be refused instead of reading an arbitrary piece of it.
    const tok = StubTokenizer{ .allocator = a, .table = &.{} };
    try std.testing.expectEqual(@as(?u32, null), resolveLabelToken(tok, a, "maybe"));
}
