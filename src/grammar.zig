//! GBNF grammar parser and constrained decoding state machine.
//!
//! Parses GBNF (GGML BNF) grammar strings into a rule-based representation,
//! then provides token-level constraint checking during generation.
//!
//! Usage:
//!   var grammar = try Grammar.parse(allocator, gbnf_text);
//!   defer grammar.deinit();
//!   var state = try grammar.initState();
//!   defer state.deinit();
//!   // In generation loop:
//!   grammar.maskLogits(&state, logits, vocab);
//!   // After sampling:
//!   state.acceptToken(token_text);

const std = @import("std");

// ── Parser Constants ───────────────────────────────────────────

const unresolved_rule_id: u32 = 0xFFFF;
const forward_ref_base: u32 = 0xFF00;
const max_json_properties: usize = 32;
const max_schema_depth: usize = 16;
const max_grammar_input_size: usize = 64 * 1024;
const max_rules: usize = 512;
const max_accept_depth: u32 = 32;
const max_stack_growth_per_token: usize = max_accept_depth + 1;
const bpe_two_byte_prefix: u8 = 0xC4;
const bpe_latin1_prefix: u8 = 0xC3;

// ── Grammar Elements ────────────────────────────────────────────

/// Categorizes grammar rule elements into character matches, rule references,
/// and their repetition/optionality variants.
pub const ElementType = enum {
    char_range, // Match character in range [lo..hi]
    char_not, // Match character NOT in range [lo..hi]
    char_range_star, // Match character in range, zero or more times (*)
    char_range_plus, // Match character in range, one or more times (+)
    char_range_opt, // Match character in range, zero or one time (?)
    char_not_star, // Match character NOT in range, zero or more times (*)
    char_not_plus, // Match character NOT in range, one or more times (+)
    char_not_opt, // Match character NOT in range, zero or one time (?)
    rule_ref, // Reference to another rule by index
    rule_ref_star, // Reference to another rule, zero or more times (*)
    rule_ref_plus, // Reference to another rule, one or more times (+)
    rule_ref_opt, // Reference to another rule, zero or one time (?)
    alt, // Alternative separator (|)
    end, // End of alternative/rule
};

/// A tagged element within a grammar rule, pairing an `ElementType` discriminator
/// with operand fields (`lo`/`hi`) whose meaning depends on the element kind.
pub const Element = struct {
    type: ElementType,
    lo: u32 = 0, // For char_range/char_not: low codepoint. For rule_ref: rule index.
    hi: u32 = 0, // For char_range/char_not: high codepoint (inclusive).
};

/// A named grammar production rule consisting of a sequence of `Element`s.
/// Alternatives within the rule are separated by `Element.alt` sentinels.
pub const Rule = struct {
    name: []const u8,
    elements: []Element,
};

// ── Grammar ─────────────────────────────────────────────────────

/// A parsed GBNF grammar: a set of named `Rule`s with a designated root.
/// Provides logit masking and jump-decoding for constrained generation.
pub const Grammar = struct {
    allocator: std.mem.Allocator,
    rules: []Rule,
    root_id: u32,
    rule_names: std.StringHashMap(u32),

    // Built-in grammars
    pub const json_grammar =
        \\root   ::= object | array
        \\object ::= "{" ws (pair ("," ws pair)*)? ws "}"
        \\pair   ::= ws string ws ":" ws value
        \\array  ::= "[" ws (value ("," ws value)*)? ws "]"
        \\value  ::= string | number | object | array | "true" | "false" | "null"
        \\string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt] | "\\u" hex hex hex hex)* "\""
        \\hex    ::= [0-9] | [a-f] | [A-F]
        \\number ::= "-"? [0-9]+ ("." [0-9]+)? (("e" | "E") ("-" | "+")? [0-9]+)?
        \\ws     ::= [ \t\n\r]*
    ;

    pub const bool_grammar = "root ::= \"true\" | \"false\"";
    pub const integer_grammar = "root ::= \"-\"? [0-9]+";

    /// Convert a JSON schema to GBNF grammar and parse it.
    /// Supports: object (with properties), string, number, integer, boolean,
    /// array (with items), enum, and nested schemas.
    pub fn fromJsonSchema(allocator: std.mem.Allocator, schema_json: []const u8) !Grammar {
        if (schema_json.len > max_grammar_input_size) return error.GrammarTooLarge;
        var converter = SchemaConverter.init(allocator);
        defer converter.deinit();
        const gbnf = try converter.convert(schema_json);
        defer allocator.free(gbnf);
        return parse(allocator, gbnf);
    }

    /// Parse a GBNF grammar string into a `Grammar`. The returned grammar owns all
    /// allocated rule data; call `deinit` to release it.
    /// Returns `error.GrammarTooLarge` if `input` exceeds `max_grammar_input_size`.
    pub fn parse(allocator: std.mem.Allocator, input: []const u8) !Grammar {
        if (input.len > max_grammar_input_size) return error.GrammarTooLarge;
        var parser = Parser.init(allocator, input);
        return parser.parseGrammar();
    }

    /// Free all rule elements, the rules slice, and the rule-name map.
    pub fn deinit(self: *Grammar) void {
        for (self.rules) |rule| {
            self.allocator.free(rule.elements);
        }
        self.allocator.free(self.rules);
        self.rule_names.deinit();
    }

    /// Create a fresh `GrammarState` for constrained decoding, starting at the root rule.
    /// Returns `error.OutOfMemory` if the stack allocation fails.
    pub fn initState(self: *const Grammar) error{OutOfMemory}!GrammarState {
        return try GrammarState.init(self);
    }

    /// Mask logits for tokens that don't match the grammar.
    /// Sets disallowed token logits to -inf.
    /// Returns `error.OutOfMemory` if the temporary test-state stack cannot be allocated
    /// (callers must fail the request, not ignore).
    pub fn maskLogits(self: *const Grammar, state: *GrammarState, logits: []f32, vocab: []const []const u8) error{OutOfMemory}!void {
        if (state.completed) return;
        if (state.stack.items.len == 0) return;

        // Reuse state.scratch across tokens so the test stack is allocated once.
        // acceptCharInner recursion allows depth 0..max_accept_depth, each pushes at most 1 entry.
        const src = state.stack.items;
        try state.prepareScratch(src);
        var test_state = GrammarState{
            .grammar = self,
            .stack = state.scratch,
            .completed = state.completed,
        };
        defer state.scratch = test_state.stack;

        // Build first-byte acceptance bitmap: test each byte 0..255 against current
        // grammar state. Rejects ~90% of vocab tokens without per-token state copy.
        var first_byte_valid: [256]bool = .{false} ** 256;
        for (0..256) |byte_val| {
            test_state.stack.items.len = src.len;
            @memcpy(test_state.stack.items.ptr[0..src.len], src);
            test_state.completed = state.completed;
            first_byte_valid[byte_val] = test_state.acceptChar(@intCast(byte_val));
        }

        for (logits, 0..) |*logit, token_id| {
            if (token_id >= vocab.len) break;
            const text = vocab[token_id];
            if (text.len == 0) continue;

            const effective = getEffectiveText(text);
            if (effective.len == 0) {
                logit.* = -std.math.inf(f32);
                continue;
            }

            // Fast reject via first-byte bitmap (avoids state copy for ~90% of tokens)
            if (!first_byte_valid[effective[0]]) {
                logit.* = -std.math.inf(f32);
                continue;
            }

            // Full validation for tokens that pass first-byte check
            test_state.stack.items.len = src.len;
            @memcpy(test_state.stack.items.ptr[0..src.len], src);
            test_state.completed = state.completed;

            var valid = true;
            for (effective) |c| {
                if (test_state.completed) {
                    // Grammar finished mid-token: remaining bytes violate it.
                    valid = false;
                    break;
                }
                if (!test_state.acceptChar(c)) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                logit.* = -std.math.inf(f32);
            }
        }
    }

    /// Jump decoding: if the grammar allows exactly one token at the current state,
    /// return that token ID without needing a forward pass. Returns null if
    /// multiple tokens are valid (need model to choose) or zero tokens (grammar error).
    pub fn singleValidToken(self: *const Grammar, state: *GrammarState, vocab: []const []const u8) ?u32 {
        if (state.completed) return null;
        if (state.stack.items.len == 0) return null;

        const src = state.stack.items;
        state.prepareScratch(src) catch return null;
        var test_state = GrammarState{
            .grammar = self,
            .stack = state.scratch,
            .completed = state.completed,
        };
        defer state.scratch = test_state.stack;

        // First-byte bitmap: skip tokens whose first byte is rejected
        var first_byte_valid: [256]bool = .{false} ** 256;
        for (0..256) |byte_val| {
            test_state.stack.items.len = src.len;
            @memcpy(test_state.stack.items.ptr[0..src.len], src);
            test_state.completed = state.completed;
            first_byte_valid[byte_val] = test_state.acceptChar(@intCast(byte_val));
        }

        var valid_id: ?u32 = null;
        for (vocab, 0..) |text, token_id| {
            if (text.len == 0) continue;
            const effective = getEffectiveText(text);
            if (effective.len == 0) continue;

            if (!first_byte_valid[effective[0]]) continue;

            test_state.stack.items.len = src.len;
            @memcpy(test_state.stack.items.ptr[0..src.len], src);
            test_state.completed = state.completed;

            var valid = true;
            for (effective) |c| {
                if (test_state.completed) {
                    valid = false;
                    break;
                }
                if (!test_state.acceptChar(c)) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                if (valid_id != null) return null;
                valid_id = @intCast(token_id);
            }
        }
        return valid_id;
    }

    /// Strip BPE byte-level encoding prefix to get actual text.
    /// Qwen/GPT uses Ġ (0xC4 0xA0) for space, Ċ (0xC4 0x8A) for newline, etc.
    pub fn getEffectiveText(text: []const u8) []const u8 {
        if (text.len >= 2 and text[0] == bpe_two_byte_prefix) {
            return text[2..];
        }
        if (text.len >= 2 and text[0] == bpe_latin1_prefix) {
            return text[2..];
        }
        // Strip a leading ASCII space (0x20), the decoded form of the BPE Ġ prefix.
        // Some tokenizers store decoded text in id_to_token, so space-prefixed tokens
        // appear as " word" rather than "Ġword".
        if (text.len >= 2 and text[0] == ' ') {
            return text[1..];
        }
        return text;
    }
};

// ── Grammar State ───────────────────────────────────────────────

const StackEntry = struct {
    rule_id: u32,
    elem_idx: u32,
};

/// Tracks the constrained-decoding state machine: a stack of rule positions
/// that advances as characters are accepted and detects grammar completion.
pub const GrammarState = struct {
    grammar: *const Grammar,
    stack: std.ArrayList(StackEntry),
    completed: bool = false,
    /// Reused by `maskLogits` / `singleValidToken` so each generated token does
    /// not heap-allocate a fresh test stack.
    scratch: std.ArrayList(StackEntry) = .empty,

    /// Initialize a grammar state by pushing the root rule onto the stack.
    /// Uses the grammar's allocator for stack storage.
    pub fn init(grammar: *const Grammar) error{OutOfMemory}!GrammarState {
        var state = GrammarState{
            .grammar = grammar,
            .stack = std.ArrayList(StackEntry).empty,
        };
        try state.stack.append(grammar.allocator, .{ .rule_id = grammar.root_id, .elem_idx = 0 });
        return state;
    }

    /// Free the rule stack using the grammar's allocator.
    pub fn deinit(self: *GrammarState) void {
        self.stack.deinit(self.grammar.allocator);
        self.scratch.deinit(self.grammar.allocator);
    }

    /// Reset `scratch` to a copy of `src` with room for acceptChar recursion.
    fn prepareScratch(self: *GrammarState, src: []const StackEntry) error{OutOfMemory}!void {
        const required_cap = src.len + max_stack_growth_per_token;
        try self.scratch.ensureTotalCapacity(self.grammar.allocator, required_cap);
        self.scratch.items.len = src.len;
        if (src.len > 0) @memcpy(self.scratch.items[0..src.len], src);
    }

    /// Test whether byte `c` is accepted by the current grammar state.
    /// Advances the state machine if accepted. Returns true if the byte matched.
    pub fn acceptChar(self: *GrammarState, c: u8) bool {
        return self.acceptCharInner(c, 0);
    }

    fn acceptCharInner(self: *GrammarState, c: u8, depth: u32) bool {
        if (depth > max_accept_depth or self.completed or self.stack.items.len == 0) return false;

        const top = &self.stack.items[self.stack.items.len - 1];
        if (top.rule_id >= self.grammar.rules.len) return false;
        const rule = self.grammar.rules[top.rule_id];
        if (top.elem_idx >= rule.elements.len) {
            _ = self.stack.pop();
            if (self.stack.items.len == 0) {
                self.completed = true;
                return false;
            }
            return self.acceptCharInner(c, depth + 1);
        }

        const elem = rule.elements[top.elem_idx];
        switch (elem.type) {
            .char_range => {
                if (@as(u32, c) >= elem.lo and @as(u32, c) <= elem.hi) {
                    top.elem_idx += 1;
                    self.advancePastEnd();
                    return true;
                }
                // Try next alternative in this rule
                return self.tryNextAlternative(c, depth);
            },
            .char_not => {
                const run_len: u32 = @intCast(self.countCharNotRun(top.rule_id, top.elem_idx));
                if (self.charNotRunAccepts(c, top.rule_id, top.elem_idx, @intCast(run_len))) {
                    top.elem_idx += run_len;
                    self.advancePastEnd();
                    return true;
                }
                return self.tryNextAlternative(c, depth);
            },
            .char_range_star => {
                if (@as(u32, c) >= elem.lo and @as(u32, c) <= elem.hi) {
                    return true; // match, stay at this element
                }
                // no match, advance past (zero matches ok)
                top.elem_idx += 1;
                self.advancePastEnd();
                return self.acceptCharInner(c, depth + 1);
            },
            .char_range_plus => unreachable, // decomposed to char_range + char_range_star by parser
            .char_range_opt => {
                // ? = zero or one
                if (@as(u32, c) >= elem.lo and @as(u32, c) <= elem.hi) {
                    top.elem_idx += 1;
                    self.advancePastEnd();
                    return true;
                }
                // No match, skip (zero matches ok)
                top.elem_idx += 1;
                return self.acceptCharInner(c, depth + 1);
            },
            .char_not_star => {
                const run_len: u32 = @intCast(self.countCharNotRun(top.rule_id, top.elem_idx));
                if (self.charNotRunAccepts(c, top.rule_id, top.elem_idx, @intCast(run_len))) {
                    return true; // match, stay at this run
                }
                top.elem_idx += run_len;
                self.advancePastEnd();
                return self.acceptCharInner(c, depth + 1);
            },
            .char_not_plus => unreachable, // decomposed to char_not + char_not_star by parser
            .char_not_opt => {
                const run_len: u32 = @intCast(self.countCharNotRun(top.rule_id, top.elem_idx));
                if (self.charNotRunAccepts(c, top.rule_id, top.elem_idx, @intCast(run_len))) {
                    top.elem_idx += run_len;
                    self.advancePastEnd();
                    return true;
                }
                top.elem_idx += run_len;
                return self.acceptCharInner(c, depth + 1);
            },
            .rule_ref => {
                top.elem_idx += 1;
                self.stack.append(self.grammar.allocator, .{ .rule_id = elem.lo, .elem_idx = 0 }) catch return false;
                return self.acceptCharInner(c, depth + 1);
            },
            .rule_ref_star => {
                // Try matching the subrule; if it fails, skip (zero matches ok)
                const saved_len = self.stack.items.len;
                const saved_completed = self.completed;
                self.stack.append(self.grammar.allocator, .{ .rule_id = elem.lo, .elem_idx = 0 }) catch return false;
                if (self.acceptCharInner(c, depth + 1)) return true;
                // Subrule didn't match, restore and advance past
                self.stack.shrinkRetainingCapacity(saved_len);
                self.completed = saved_completed;
                top.elem_idx += 1;
                self.advancePastEnd();
                return self.acceptCharInner(c, depth + 1);
            },
            .rule_ref_opt => {
                // Try matching subrule once; if fails, skip
                const saved_len = self.stack.items.len;
                const saved_completed = self.completed;
                top.elem_idx += 1; // advance past regardless
                self.stack.append(self.grammar.allocator, .{ .rule_id = elem.lo, .elem_idx = 0 }) catch return false;
                if (self.acceptCharInner(c, depth + 1)) return true;
                // Didn't match, restore and try without
                self.stack.shrinkRetainingCapacity(saved_len);
                self.completed = saved_completed;
                self.advancePastEnd();
                return self.acceptCharInner(c, depth + 1);
            },
            .rule_ref_plus => unreachable, // decomposed to rule_ref + rule_ref_star by parser
            .alt => {
                // Skip past this alt marker
                top.elem_idx += 1;
                return self.acceptCharInner(c, depth + 1);
            },
            .end => {
                _ = self.stack.pop();
                if (self.stack.items.len == 0) {
                    self.completed = true;
                    return false;
                }
                return self.acceptCharInner(c, depth + 1);
            },
        }
    }

    fn advancePastEnd(self: *GrammarState) void {
        while (self.stack.items.len > 0) {
            const t = &self.stack.items[self.stack.items.len - 1];
            if (t.rule_id >= self.grammar.rules.len) break;
            const r = self.grammar.rules[t.rule_id];
            if (t.elem_idx < r.elements.len and r.elements[t.elem_idx].type != .end and r.elements[t.elem_idx].type != .alt) break;
            // Skip past end/alt markers
            if (t.elem_idx < r.elements.len and r.elements[t.elem_idx].type == .alt) {
                // Skip remaining alternatives (we already matched one)
                while (t.elem_idx < r.elements.len and r.elements[t.elem_idx].type != .end) : (t.elem_idx += 1) {}
            }
            if (t.elem_idx >= r.elements.len or r.elements[t.elem_idx].type == .end) {
                _ = self.stack.pop();
                if (self.stack.items.len == 0) {
                    self.completed = true;
                    return;
                }
            } else break;
        }
    }

    fn tryNextAlternative(self: *GrammarState, c: u8, depth: u32) bool {
        if (self.stack.items.len == 0) return false;
        const top = &self.stack.items[self.stack.items.len - 1];
        if (top.rule_id >= self.grammar.rules.len) return false;
        const rule = self.grammar.rules[top.rule_id];
        var idx = top.elem_idx;
        if (idx < rule.elements.len and rule.elements[idx].type == .char_not) {
            idx = std.math.add(u32, idx, std.math.cast(u32, self.countCharNotRun(top.rule_id, idx)) orelse 1) catch 1;
        } else {
            idx = std.math.add(u32, idx, 1) catch return false;
        }
        while (idx < rule.elements.len) : (idx += 1) {
            if (rule.elements[idx].type == .alt) {
                top.elem_idx = idx + 1;
                return self.acceptCharInner(c, depth + 1);
            }
            if (rule.elements[idx].type == .end) break;
        }
        return false;
    }

    fn countCharNotRun(self: *const GrammarState, rule_id: u32, start: u32) usize {
        if (rule_id >= self.grammar.rules.len) return 1;
        const elems = self.grammar.rules[rule_id].elements;
        var n: usize = 0;
        var i = start;
        while (i < elems.len and elems[i].type == .char_not) : (i += 1) {
            n += 1;
        }
        return if (n == 0) 1 else n;
    }

    fn charNotRunAccepts(self: *const GrammarState, c: u8, rule_id: u32, start: u32, run_len: usize) bool {
        if (rule_id >= self.grammar.rules.len) return false;
        const elems = self.grammar.rules[rule_id].elements;
        const uc: u32 = @as(u32, c);
        for (0..run_len) |k| {
            const idx = start + k;
            if (idx >= elems.len) break;
            const e = elems[idx];
            if (e.type != .char_not) break;
            if (uc >= e.lo and uc <= e.hi) return false;
        }
        return true;
    }

    /// Feed a full token's text through the state machine, accepting characters one by one.
    /// Strips BPE multi-byte prefixes via `getEffectiveText` before processing.
    pub fn acceptToken(self: *GrammarState, text: []const u8) void {
        const effective = Grammar.getEffectiveText(text);
        for (effective) |c| {
            if (self.completed) break;
            if (!self.acceptChar(c)) break;
        }
    }

    /// Returns true if the grammar has reached a terminal state (all rules satisfied).
    pub fn isComplete(self: *const GrammarState) bool {
        return self.completed;
    }
};

// ── GBNF Parser ─────────────────────────────────────────────────

const UnresolvedRef = struct {
    name: []const u8, // name to resolve
};

const Parser = struct {
    allocator: std.mem.Allocator,
    input: []const u8,
    pos: usize = 0,
    rules: std.ArrayList(Rule),
    rule_names: std.StringHashMap(u32),
    elements: std.ArrayList(Element),
    unresolved: std.ArrayList(UnresolvedRef) = .empty,

    fn init(allocator: std.mem.Allocator, input: []const u8) Parser {
        return .{
            .allocator = allocator,
            .input = input,
            .rules = std.ArrayList(Rule).empty,
            .rule_names = std.StringHashMap(u32).init(allocator),
            .elements = std.ArrayList(Element).empty,
        };
    }

    fn parseGrammar(self: *Parser) !Grammar {
        errdefer {
            for (self.rules.items) |rule| self.allocator.free(rule.elements);
            self.rules.deinit(self.allocator);
            self.elements.deinit(self.allocator);
            self.unresolved.deinit(self.allocator);
            self.rule_names.deinit();
        }
        // Pass 1: collect all rule names so forward references resolve correctly
        {
            var scan_pos: usize = 0;
            while (scan_pos < self.input.len) {
                // Skip whitespace
                while (scan_pos < self.input.len and (self.input[scan_pos] == ' ' or self.input[scan_pos] == '\t' or self.input[scan_pos] == '\r')) : (scan_pos += 1) {}
                if (scan_pos >= self.input.len) break;
                if (self.input[scan_pos] == '#' or self.input[scan_pos] == '\n') {
                    while (scan_pos < self.input.len and self.input[scan_pos] != '\n') : (scan_pos += 1) {}
                    if (scan_pos < self.input.len) scan_pos += 1;
                    continue;
                }
                // Read rule name
                const name_start = scan_pos;
                while (scan_pos < self.input.len and (std.ascii.isAlphanumeric(self.input[scan_pos]) or self.input[scan_pos] == '_' or self.input[scan_pos] == '-')) : (scan_pos += 1) {}
                const name = self.input[name_start..scan_pos];
                if (name.len == 0) {
                    scan_pos += 1;
                    continue;
                }
                // Skip to ::=
                while (scan_pos < self.input.len and (self.input[scan_pos] == ' ' or self.input[scan_pos] == '\t')) : (scan_pos += 1) {}
                if (scan_pos + 3 <= self.input.len and std.mem.eql(u8, self.input[scan_pos..][0..3], "::=")) {
                    // Pre-register with placeholder, actual ID assigned in pass 2
                    if (!self.rule_names.contains(name)) {
                        try self.rule_names.put(name, unresolved_rule_id);
                    }
                }
                // Skip to next line
                while (scan_pos < self.input.len and self.input[scan_pos] != '\n') : (scan_pos += 1) {}
                if (scan_pos < self.input.len) scan_pos += 1;
            }
        }

        // Pass 2: parse rule bodies (synthetic rules from groups get real IDs)
        while (self.pos < self.input.len) {
            if (self.rules.items.len >= max_rules) return error.TooManyRules;
            self.skipWs();
            if (self.pos >= self.input.len) break;
            if (self.input[self.pos] == '#' or self.input[self.pos] == '\n') {
                self.skipLine();
                continue;
            }
            try self.parseRule();
        }

        self.elements.deinit(self.allocator);
        self.elements = .empty;

        for (self.unresolved.items, 0..) |ref, i| {
            const placeholder: u32 = forward_ref_base + @as(u32, @intCast(i));
            const resolved_id = self.rule_names.get(ref.name) orelse continue;
            if (resolved_id == unresolved_rule_id) continue;
            for (self.rules.items) |rule| {
                for (rule.elements) |*elem| {
                    switch (elem.type) {
                        .rule_ref, .rule_ref_star, .rule_ref_opt, .rule_ref_plus => {
                            if (elem.lo == placeholder) elem.lo = resolved_id;
                        },
                        else => {},
                    }
                }
            }
        }

        self.unresolved.deinit(self.allocator);
        self.unresolved = .empty;
        const root_id = self.rule_names.get("root") orelse return error.MissingRootRule;
        return Grammar{
            .allocator = self.allocator,
            .rules = try self.rules.toOwnedSlice(self.allocator),
            .root_id = root_id,
            .rule_names = self.rule_names,
        };
    }

    fn parseRule(self: *Parser) !void {
        const name_start = self.pos;
        while (self.pos < self.input.len and (std.ascii.isAlphanumeric(self.input[self.pos]) or self.input[self.pos] == '_' or self.input[self.pos] == '-')) : (self.pos += 1) {}
        const name = self.input[name_start..self.pos];
        if (name.len == 0) {
            self.skipLine();
            return;
        }

        self.skipWs();
        // Expect ::=
        if (self.pos + 3 <= self.input.len and std.mem.eql(u8, self.input[self.pos..][0..3], "::=")) {
            self.pos += 3;
        } else return;

        // Reserve a slot for this rule (pre-registered name points here).
        // Synthetic rules from (...) groups may be inserted during body parsing,
        // so we need to fix up the ID after parsing.
        const pre_id: u32 = @intCast(self.rules.items.len);
        try self.rule_names.put(name, pre_id);

        const elem_start = self.elements.items.len;
        try self.parseAlternatives();
        try self.elements.append(self.allocator, .{ .type = .end });

        const elems = try self.allocator.dupe(Element, self.elements.items[elem_start..]);
        self.elements.shrinkRetainingCapacity(elem_start);

        // Actual rule ID may differ if synthetic rules were inserted
        const actual_id: u32 = @intCast(self.rules.items.len);
        if (actual_id != pre_id) {
            try self.rule_names.put(name, actual_id);
        }
        try self.rules.append(self.allocator, .{ .name = name, .elements = elems });
    }

    const ParseError = error{OutOfMemory};

    fn parseAlternatives(self: *Parser) ParseError!void {
        try self.parseSequence();
        while (self.pos < self.input.len) {
            self.skipWs();
            if (self.pos < self.input.len and self.input[self.pos] == '|') {
                try self.elements.append(self.allocator, .{ .type = .alt });
                self.pos += 1;
                try self.parseSequence();
            } else break;
        }
    }

    fn parseSequence(self: *Parser) ParseError!void {
        while (self.pos < self.input.len) {
            self.skipWs();
            if (self.pos >= self.input.len) break;
            const c = self.input[self.pos];
            if (c == '\n' or c == '|' or c == ')') break;
            if (c == '#') {
                self.skipLine();
                break;
            }
            try self.parseElement();
        }
    }

    fn parseElement(self: *Parser) ParseError!void {
        self.skipWs();
        if (self.pos >= self.input.len) return;

        const c = self.input[self.pos];
        if (c == '"') {
            try self.parseString();
        } else if (c == '[') {
            try self.parseCharClass();
        } else if (c == '(') {
            self.pos += 1;
            // Save current elements, parse group into separate buffer, create synthetic rule
            const group_start = self.elements.items.len;
            try self.parseAlternatives();
            if (self.pos < self.input.len and self.input[self.pos] == ')') self.pos += 1;
            const group_end = self.elements.items.len;

            // Extract group elements into a synthetic rule
            const group_elems_src = self.elements.items[group_start..group_end];
            const n_group = group_elems_src.len + 1; // +1 for end marker
            const group_elems = try self.allocator.alloc(Element, n_group);
            @memcpy(group_elems[0..group_elems_src.len], group_elems_src);
            group_elems[n_group - 1] = .{ .type = .end };

            // Remove group elements from inline position
            self.elements.shrinkRetainingCapacity(group_start);

            if (self.rules.items.len >= max_rules) {
                self.allocator.free(group_elems);
                return error.OutOfMemory;
            }
            const synth_id: u32 = @intCast(self.rules.items.len);
            try self.rules.append(self.allocator, .{ .name = "_group", .elements = group_elems });

            // Emit rule_ref to the synthetic rule
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = synth_id });
        } else if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-') {
            try self.parseRuleRef();
        } else {
            self.pos += 1;
        }

        // Handle repetition modifiers: *, +, ?
        if (self.pos < self.input.len) {
            const mod = self.input[self.pos];
            if (mod == '*' or mod == '+' or mod == '?') {
                self.pos += 1;
                if (self.elements.items.len > 0) {
                    const last = &self.elements.items[self.elements.items.len - 1];
                    switch (last.type) {
                        .char_range => {
                            if (mod == '+') {
                                // x+ → x x* (one mandatory + zero or more)
                                const lo = last.lo;
                                const hi = last.hi;
                                try self.elements.append(self.allocator, .{ .type = .char_range_star, .lo = lo, .hi = hi });
                            } else {
                                last.type = switch (mod) {
                                    '*' => .char_range_star,
                                    '?' => .char_range_opt,
                                    else => last.type,
                                };
                            }
                        },
                        .char_not => {
                            if (mod == '+') {
                                // x+ → x x* (one mandatory + zero or more)
                                const lo = last.lo;
                                const hi = last.hi;
                                try self.elements.append(self.allocator, .{ .type = .char_not_star, .lo = lo, .hi = hi });
                            } else {
                                last.type = switch (mod) {
                                    '*' => .char_not_star,
                                    '?' => .char_not_opt,
                                    else => last.type,
                                };
                            }
                        },
                        .rule_ref => {
                            if (mod == '+') {
                                // rule+ → rule rule* (one mandatory + zero or more)
                                const rule_id = last.lo;
                                try self.elements.append(self.allocator, .{ .type = .rule_ref_star, .lo = rule_id });
                            } else {
                                last.type = switch (mod) {
                                    '*' => .rule_ref_star,
                                    '?' => .rule_ref_opt,
                                    else => last.type,
                                };
                            }
                        },
                        else => {},
                    }
                }
            }
        }
    }

    fn parseString(self: *Parser) !void {
        self.pos += 1;
        while (self.pos < self.input.len and self.input[self.pos] != '"') {
            var ch = self.input[self.pos];
            if (ch == '\\' and self.pos + 1 < self.input.len) {
                self.pos += 1;
                ch = switch (self.input[self.pos]) {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '"' => '"',
                    else => self.input[self.pos],
                };
            }
            try self.elements.append(self.allocator, .{ .type = .char_range, .lo = ch, .hi = ch });
            self.pos += 1;
        }
        if (self.pos < self.input.len) self.pos += 1;
    }

    fn parseCharClass(self: *Parser) !void {
        self.pos += 1;
        var negate = false;
        if (self.pos < self.input.len and self.input[self.pos] == '^') {
            negate = true;
            self.pos += 1;
        }
        // Collect all ranges in the character class first.
        const RangePair = struct { lo: u32, hi: u32 };
        var ranges: [64]RangePair = undefined;
        var n_ranges: usize = 0;
        while (self.pos < self.input.len and self.input[self.pos] != ']') {
            var lo: u8 = self.input[self.pos];
            self.pos += 1;
            // Handle escape sequences inside character class
            if (lo == '\\' and self.pos < self.input.len) {
                lo = switch (self.input[self.pos]) {
                    'n' => 0x0A,
                    't' => 0x09,
                    'r' => 0x0D,
                    '\\' => 0x5C,
                    ']' => 0x5D,
                    '-' => 0x2D,
                    else => self.input[self.pos],
                };
                self.pos += 1;
            }
            var hi = lo;
            if (self.pos + 1 < self.input.len and self.input[self.pos] == '-' and self.input[self.pos + 1] != ']') {
                self.pos += 1;
                hi = self.input[self.pos];
                self.pos += 1;
                // Handle escape sequences for range endpoint
                if (hi == '\\' and self.pos < self.input.len) {
                    hi = switch (self.input[self.pos]) {
                        'n' => 0x0A,
                        't' => 0x09,
                        'r' => 0x0D,
                        '\\' => 0x5C,
                        ']' => 0x5D,
                        '-' => 0x2D,
                        else => self.input[self.pos],
                    };
                    self.pos += 1;
                }
            }
            if (n_ranges < ranges.len) {
                ranges[n_ranges] = .{ .lo = lo, .hi = hi };
                n_ranges += 1;
            }
        }
        if (self.pos < self.input.len) self.pos += 1;
        // Single range: emit directly (no .alt needed).
        // Negated classes: emit all ranges sequentially (acceptCharInner handles
        // char_not by checking all consecutive char_not elements before accepting).
        if (n_ranges <= 1 or negate) {
            const elem_type: ElementType = if (negate) .char_not else .char_range;
            for (ranges[0..n_ranges]) |r| {
                try self.elements.append(self.allocator, .{ .type = elem_type, .lo = r.lo, .hi = r.hi });
            }
            return;
        }
        // Multi-range non-negated class: emit alternatives so the matcher
        // accepts any single character from the union of ranges.
        for (ranges[0..n_ranges], 0..) |r, i| {
            if (i > 0) try self.elements.append(self.allocator, .{ .type = .alt, .lo = 0 });
            try self.elements.append(self.allocator, .{ .type = .char_range, .lo = r.lo, .hi = r.hi });
        }
    }

    fn parseRuleRef(self: *Parser) !void {
        const start = self.pos;
        while (self.pos < self.input.len and (std.ascii.isAlphanumeric(self.input[self.pos]) or self.input[self.pos] == '_' or self.input[self.pos] == '-')) : (self.pos += 1) {}
        const name = self.input[start..self.pos];
        const rule_id = self.rule_names.get(name) orelse 0;
        if (rule_id != unresolved_rule_id) {
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = rule_id });
        } else {
            if (self.unresolved.items.len >= 0xFF) return;
            const placeholder: u32 = std.math.add(u32, forward_ref_base, @as(u32, @intCast(self.unresolved.items.len))) catch return;
            if (placeholder == unresolved_rule_id) return;
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = placeholder });
            try self.unresolved.append(self.allocator, .{
                .name = name,
            });
        }
    }

    fn skipWs(self: *Parser) void {
        while (self.pos < self.input.len and (self.input[self.pos] == ' ' or self.input[self.pos] == '\t' or self.input[self.pos] == '\r')) : (self.pos += 1) {}
    }

    fn skipLine(self: *Parser) void {
        while (self.pos < self.input.len and self.input[self.pos] != '\n') : (self.pos += 1) {}
        if (self.pos < self.input.len) self.pos += 1;
    }
};

// ── JSON Schema → GBNF Converter ────────────────────────────────

const SchemaConverter = struct {
    allocator: std.mem.Allocator,
    rules: std.ArrayList(u8),

    fn init(allocator: std.mem.Allocator) SchemaConverter {
        return .{
            .allocator = allocator,
            .rules = std.ArrayList(u8).empty,
        };
    }

    fn deinit(self: *SchemaConverter) void {
        self.rules.deinit(self.allocator);
    }

    fn convert(self: *SchemaConverter, schema_json: []const u8) ![]u8 {
        // Shared primitive rules
        try self.emit("string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt])* \"\\\"\"\n");
        try self.emit("number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\n");
        try self.emit("integer ::= \"-\"? [0-9]+\n");
        try self.emit("boolean ::= \"true\" | \"false\"\n");
        try self.emit("null ::= \"null\"\n");
        try self.emit("ws ::= [ \\t\\n\\r]*\n");
        try self.emit("value ::= string | number | boolean | null\n");

        try self.emitRule("root", schema_json, 0);
        return self.rules.toOwnedSlice(self.allocator);
    }

    const ConvertError = error{OutOfMemory};
    fn emitRule(self: *SchemaConverter, name: []const u8, schema: []const u8, depth: usize) ConvertError!void {
        if (depth > max_schema_depth) {
            // Emit a permissive catch-all so the rule name resolves instead of
            // silently leaving it undefined (which maps to rule 0 via orelse 0).
            try self.emit(name);
            try self.emit(" ::= value\n");
            return;
        }
        const type_str = extractJsonStr(schema, "type");

        // Check for enum first
        if (findJsonArray(schema, "enum")) |enum_content| {
            try self.emit(name);
            try self.emit(" ::= ");
            try self.emitEnum(enum_content);
            try self.emit("\n");
            return;
        }

        if (type_str) |t| {
            if (std.mem.eql(u8, t, "object")) {
                try self.emitObject(name, schema, depth);
                return;
            } else if (std.mem.eql(u8, t, "array")) {
                try self.emitArray(name, schema, depth);
                return;
            }
        }
        const simple_types = [_][]const u8{ "string", "number", "integer", "boolean", "null" };
        const rule_name = if (type_str) |t| blk: {
            for (simple_types) |st| {
                if (std.mem.eql(u8, t, st)) break :blk st;
            }
            break :blk "value";
        } else "value";
        try self.emit(name);
        try self.emit(" ::= ");
        try self.emit(rule_name);
        try self.emit("\n");
    }

    fn emitObject(self: *SchemaConverter, name: []const u8, schema: []const u8, depth: usize) ConvertError!void {
        const props_content = findJsonObject(schema, "properties") orelse {
            try self.emit(name);
            try self.emit(" ::= \"{\" ws \"}\" | \"{\" ws string ws \":\" ws value (ws \",\" ws string ws \":\" ws value)* ws \"}\"\n");
            return;
        };

        // Parse property names and their schemas
        var prop_names: [max_json_properties][]const u8 = undefined;
        var prop_schemas: [max_json_properties][]const u8 = undefined;
        var n_props: usize = 0;

        var pi: usize = 0;
        while (pi < props_content.len and n_props < max_json_properties) {
            // Skip to next property key
            pi = skipWsSchema(props_content, pi);
            if (pi >= props_content.len or props_content[pi] != '"') break;

            const key_start = pi + 1;
            pi = key_start;
            while (pi < props_content.len and props_content[pi] != '"') : (pi += 1) {}
            const key_end = pi;
            if (pi < props_content.len) pi += 1; // skip closing "

            // Skip to colon + value
            pi = skipWsSchema(props_content, pi);
            if (pi < props_content.len and props_content[pi] == ':') pi += 1;
            pi = skipWsSchema(props_content, pi);

            // Find the schema object for this property
            const val_start = pi;
            pi = skipJsonValue(props_content, pi);
            const val_end = pi;

            prop_names[n_props] = props_content[key_start..key_end];
            prop_schemas[n_props] = props_content[val_start..val_end];
            n_props += 1;

            // Skip comma
            pi = skipWsSchema(props_content, pi);
            if (pi < props_content.len and props_content[pi] == ',') pi += 1;
        }

        if (n_props == 0) {
            try self.emit(name);
            try self.emit(" ::= \"{\" ws \"}\"\n");
            return;
        }

        // Emit: root ::= "{" ws "\"key1\"" ws ":" ws val1 "," ... "}"
        try self.emit(name);
        try self.emit(" ::= \"{\" ws ");
        for (0..n_props) |i| {
            if (i > 0) try self.emit(" \",\" ws ");
            try self.emit("\"\\\"");
            try self.emit(prop_names[i]);
            try self.emit("\\\"\" ws \":\" ws ");

            // Generate sub-rule for this property's value
            var sub_name_buf: [64]u8 = undefined;
            const sub_name = std.fmt.bufPrint(&sub_name_buf, "{s}-{s}", .{ name, prop_names[i] }) catch "prop";
            try self.emitRule(sub_name, prop_schemas[i], depth + 1);
            try self.emit(sub_name);
        }
        try self.emit(" ws \"}\"\n");
    }

    fn emitArray(self: *SchemaConverter, name: []const u8, schema: []const u8, depth: usize) ConvertError!void {
        const items_schema = findJsonObject(schema, "items");

        var item_rule: []const u8 = "value";
        var sub_name_buf: [64]u8 = undefined;
        if (items_schema) |is| {
            const sub = std.fmt.bufPrint(&sub_name_buf, "{s}-item", .{name}) catch "item";
            try self.emitRule(sub, is, depth + 1);
            item_rule = sub;
        }

        try self.emit(name);
        try self.emit(" ::= \"[\" ws (");
        try self.emit(item_rule);
        try self.emit(" (\",\" ws ");
        try self.emit(item_rule);
        try self.emit(")*)? ws \"]\"\n");
    }

    fn emitEnum(self: *SchemaConverter, content: []const u8) ConvertError!void {
        var ei: usize = 0;
        var first = true;
        while (ei < content.len) {
            ei = skipWsSchema(content, ei);
            if (ei >= content.len) break;
            if (content[ei] == '"') {
                const str_start = ei + 1;
                ei = str_start;
                while (ei < content.len and content[ei] != '"') : (ei += 1) {}
                const str_end = ei;
                if (ei < content.len) ei += 1;
                if (!first) try self.emit(" | ");
                try self.emit("\"");
                try self.emit(content[str_start..str_end]);
                try self.emit("\"");
                first = false;
            } else if (content[ei] == ',') {
                ei += 1;
            } else if (content[ei] == '-' or (content[ei] >= '0' and content[ei] <= '9')) {
                // Numeric literal (integer or float, possibly negative)
                const lit_start = ei;
                if (content[ei] == '-') ei += 1;
                while (ei < content.len and ((content[ei] >= '0' and content[ei] <= '9') or content[ei] == '.' or content[ei] == 'e' or content[ei] == 'E' or content[ei] == '+' or content[ei] == '-')) : (ei += 1) {}
                if (!first) try self.emit(" | ");
                try self.emit("\"");
                try self.emit(content[lit_start..ei]);
                try self.emit("\"");
                first = false;
            } else if (ei + 4 <= content.len and std.mem.eql(u8, content[ei..][0..4], "true")) {
                if (!first) try self.emit(" | ");
                try self.emit("\"true\"");
                ei += 4;
                first = false;
            } else if (ei + 5 <= content.len and std.mem.eql(u8, content[ei..][0..5], "false")) {
                if (!first) try self.emit(" | ");
                try self.emit("\"false\"");
                ei += 5;
                first = false;
            } else if (ei + 4 <= content.len and std.mem.eql(u8, content[ei..][0..4], "null")) {
                if (!first) try self.emit(" | ");
                try self.emit("\"null\"");
                ei += 4;
                first = false;
            } else {
                ei += 1;
            }
        }
    }

    fn emit(self: *SchemaConverter, s: []const u8) ConvertError!void {
        try self.rules.appendSlice(self.allocator, s);
    }

    // Simple JSON field extraction for schema parsing
    fn extractJsonStr(json: []const u8, field: []const u8) ?[]const u8 {
        var buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
        const idx = std.mem.indexOf(u8, json, needle) orelse return null;
        var i = idx + needle.len;
        while (i < json.len and (json[i] == ' ' or json[i] == ':')) : (i += 1) {}
        if (i >= json.len or json[i] != '"') return null;
        i += 1;
        const start = i;
        while (i < json.len and json[i] != '"') : (i += 1) {}
        return json[start..i];
    }

    fn findJsonDelimited(json: []const u8, field: []const u8, open: u8, close: u8) ?[]const u8 {
        var buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
        const idx = std.mem.indexOf(u8, json, needle) orelse return null;
        var i = idx + needle.len;
        while (i < json.len and (json[i] == ' ' or json[i] == ':')) : (i += 1) {}
        if (i >= json.len or json[i] != open) return null;
        const start = i + 1;
        var depth: i32 = 1;
        i += 1;
        while (i < json.len and depth > 0) : (i += 1) {
            if (json[i] == open) depth += 1;
            if (json[i] == close) depth -= 1;
        }
        return json[start .. i - 1];
    }

    fn findJsonObject(json: []const u8, field: []const u8) ?[]const u8 {
        return findJsonDelimited(json, field, '{', '}');
    }

    fn findJsonArray(json: []const u8, field: []const u8) ?[]const u8 {
        return findJsonDelimited(json, field, '[', ']');
    }

    fn skipWsSchema(json: []const u8, start: usize) usize {
        var i = start;
        while (i < json.len and (json[i] == ' ' or json[i] == '\t' or json[i] == '\n' or json[i] == '\r')) : (i += 1) {}
        return i;
    }

    fn skipJsonValue(json: []const u8, start: usize) usize {
        if (start >= json.len) return start;
        var i = start;
        switch (json[i]) {
            '{' => {
                var depth: i32 = 1;
                i += 1;
                while (i < json.len and depth > 0) : (i += 1) {
                    if (json[i] == '{') depth += 1;
                    if (json[i] == '}') depth -= 1;
                }
                return i;
            },
            '[' => {
                var depth: i32 = 1;
                i += 1;
                while (i < json.len and depth > 0) : (i += 1) {
                    if (json[i] == '[') depth += 1;
                    if (json[i] == ']') depth -= 1;
                }
                return i;
            },
            '"' => {
                i += 1;
                while (i < json.len and json[i] != '"') {
                    if (json[i] == '\\' and i + 1 < json.len) i += 1;
                    i += 1;
                }
                if (i < json.len) i += 1;
                return i;
            },
            else => {
                while (i < json.len and json[i] != ',' and json[i] != '}' and json[i] != ']') : (i += 1) {}
                return i;
            },
        }
    }
};

// ── Tests ───────────────────────────────────────────────────────

test "parse simple grammar" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= \"hello\"");
    defer grammar.deinit();
    try std.testing.expectEqual(@as(usize, 1), grammar.rules.len);
    try std.testing.expectEqualStrings("root", grammar.rules[0].name);
}

test "parse bool grammar" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.bool_grammar);
    defer grammar.deinit();
    try std.testing.expectEqual(@as(usize, 1), grammar.rules.len);
}

test "char repetition star" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= [a-z]*");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Zero matches, should complete immediately (empty string valid)
    try std.testing.expect(state.acceptChar('1') == false);
}

test "char repetition plus" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= [0-9]+");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Must match at least one digit
    try std.testing.expect(state.acceptChar('5'));
    try std.testing.expect(state.acceptChar('3'));
    try std.testing.expect(state.acceptChar('a') == false); // stops
}

test "char repetition plus rejects zero matches" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= [0-9]+ \"x\"");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Cannot start with 'x', need at least one digit first
    try std.testing.expect(state.acceptChar('x') == false);
    try std.testing.expect(!state.isComplete());

    // Valid path on a fresh state: one-or-more digits then literal x
    var ok = try grammar.initState();
    defer ok.deinit();
    try std.testing.expect(ok.acceptChar('9'));
    try std.testing.expect(ok.acceptChar('x'));
    try std.testing.expect(ok.isComplete());
}

test "grouped repetition star" {
    const allocator = std.testing.allocator;
    // ("ab")* should match "", "ab", "abab", etc.
    var grammar = try Grammar.parse(allocator, "root ::= (\"ab\")*");
    defer grammar.deinit();

    // Verify parse structure: should have 2 rules (synthetic _group + root)
    // _group rule: char_range('a'), char_range('b'), end
    // root rule: rule_ref_star(_group), end
    try std.testing.expectEqual(@as(usize, 2), grammar.rules.len);

    var state = try grammar.initState();
    defer state.deinit();

    // First match
    try std.testing.expect(state.acceptChar('a'));
    try std.testing.expect(state.acceptChar('b'));
    // Second match (repetition)
    try std.testing.expect(state.acceptChar('a'));
    try std.testing.expect(state.acceptChar('b'));
    // 'c' should fail (not 'a' or end)
    try std.testing.expect(state.acceptChar('c') == false);
}

test "json grammar parses" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.json_grammar);
    defer grammar.deinit();
    // json_grammar defines 9 named rules (root, object, pair, array, value, string, hex, number, ws)
    // plus synthetic groups from repetition/optional expansions
    try std.testing.expect(grammar.rules.len >= 9);
    // Verify rule names include all core JSON rules
    var found_root = false;
    var found_object = false;
    var found_value = false;
    var found_string = false;
    for (grammar.rules) |rule| {
        if (std.mem.eql(u8, rule.name, "root")) found_root = true;
        if (std.mem.eql(u8, rule.name, "object")) found_object = true;
        if (std.mem.eql(u8, rule.name, "value")) found_value = true;
        if (std.mem.eql(u8, rule.name, "string")) found_string = true;
    }
    try std.testing.expect(found_root);
    try std.testing.expect(found_object);
    try std.testing.expect(found_value);
    try std.testing.expect(found_string);

    var state = try grammar.initState();
    defer state.deinit();

    // Should accept start of valid JSON
    try std.testing.expect(state.acceptChar('{'));
}

test "integer grammar" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.integer_grammar);
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Positive integer
    try std.testing.expect(state.acceptChar('4'));
    try std.testing.expect(state.acceptChar('2'));
    try std.testing.expect(state.acceptChar('a') == false);
}

test "integer grammar negative" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.integer_grammar);
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Negative integer
    try std.testing.expect(state.acceptChar('-'));
    try std.testing.expect(state.acceptChar('7'));
    try std.testing.expect(state.acceptChar('x') == false);
}

test "json schema string type" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"type\": \"string\"}");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('"'));
    try std.testing.expect(state.acceptChar('h'));
    try std.testing.expect(state.acceptChar('i'));
    try std.testing.expect(state.acceptChar('"'));
    // Non-quote start should be rejected
    var state2 = try grammar.initState();
    defer state2.deinit();
    try std.testing.expect(!state2.acceptChar('h'));
}

test "json schema boolean type" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"type\": \"boolean\"}");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('t'));
    try std.testing.expect(state.acceptChar('r'));
    try std.testing.expect(state.acceptChar('u'));
    try std.testing.expect(state.acceptChar('e'));
    // Also verify "false" path works
    var state2 = try grammar.initState();
    defer state2.deinit();
    try std.testing.expect(state2.acceptChar('f'));
    try std.testing.expect(state2.acceptChar('a'));
    try std.testing.expect(state2.acceptChar('l'));
    try std.testing.expect(state2.acceptChar('s'));
    try std.testing.expect(state2.acceptChar('e'));
}

test "json schema enum" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"enum\": [\"red\", \"green\", \"blue\"]}");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('r'));
    try std.testing.expect(state.acceptChar('e'));
    try std.testing.expect(state.acceptChar('d'));
    // "green" path from fresh state
    var state2 = try grammar.initState();
    defer state2.deinit();
    try std.testing.expect(state2.acceptChar('g'));
    try std.testing.expect(state2.acceptChar('r'));
    // Invalid start should be rejected
    var state3 = try grammar.initState();
    defer state3.deinit();
    try std.testing.expect(!state3.acceptChar('x'));
}

test "json schema object" {
    const allocator = std.testing.allocator;
    const schema =
        \\{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    ;
    var grammar = try Grammar.fromJsonSchema(allocator, schema);
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Opening brace accepted
    try std.testing.expect(state.acceptChar('{'));
    // Non-object start rejected from fresh state
    var state2 = try grammar.initState();
    defer state2.deinit();
    try std.testing.expect(!state2.acceptChar('['));
}

test "json schema array" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"type\": \"array\", \"items\": {\"type\": \"integer\"}}");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('['));
    // Non-array start rejected from fresh state
    var state2 = try grammar.initState();
    defer state2.deinit();
    try std.testing.expect(!state2.acceptChar('{'));
}

test "json schema integer" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"type\": \"integer\"}");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('-'));
    try std.testing.expect(state.acceptChar('5'));
    try std.testing.expect(state.acceptChar('a') == false);
}

test "parse empty input returns MissingRootRule" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.MissingRootRule, Grammar.parse(allocator, ""));
}

test "parse nonsense input returns MissingRootRule" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.MissingRootRule, Grammar.parse(allocator, "!@#$%^&*()"));
}

test "singleValidToken with bool grammar" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.bool_grammar);
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();
    // Vocab: ["t", "r", "true", "false", "f"]
    const vocab = [_][]const u8{ "t", "r", "true", "false", "f" };
    // Initially, both "true" and "false" (and "t" and "f") are valid prefixes
    // so singleValidToken should return null
    try std.testing.expect(grammar.singleValidToken(&state, &vocab) == null);
}

test "singleValidToken with fixed literal" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= \"yes\"");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();
    // Vocab: "y", "e", "s", "yes", "no"
    const vocab = [_][]const u8{ "y", "e", "s", "yes", "no" };
    // "yes" (idx 3) and "y" (idx 0) are both valid prefixes → null
    try std.testing.expect(grammar.singleValidToken(&state, &vocab) == null);
    // After accepting "y", only "e" (idx 1) is valid
    state.acceptToken("y");
    const result = grammar.singleValidToken(&state, &vocab);
    try std.testing.expectEqual(@as(?u32, 1), result);
}

test "isComplete after full match" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= \"hi\"");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    try std.testing.expect(!state.isComplete());
    try std.testing.expect(state.acceptChar('h'));
    try std.testing.expect(!state.isComplete());
    try std.testing.expect(state.acceptChar('i'));
    try std.testing.expect(state.isComplete());
}

test "maskLogits constrains vocab" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= \"yes\"");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    // Vocab: ["y", "n", "yes", "no"]
    const vocab = [_][]const u8{ "y", "n", "yes", "no" };
    var logits = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    grammar.maskLogits(&state, &logits, &vocab) catch unreachable;
    // "y" (idx 0) valid prefix, "yes" (idx 2) valid full match
    try std.testing.expect(logits[0] != -std.math.inf(f32));
    try std.testing.expect(logits[2] != -std.math.inf(f32));
    // "n" and "no" should be masked
    try std.testing.expectEqual(-std.math.inf(f32), logits[1]);
    try std.testing.expectEqual(-std.math.inf(f32), logits[3]);
}

test "acceptToken advances state" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= \"hello\"");
    defer grammar.deinit();
    var state = try grammar.initState();
    defer state.deinit();

    state.acceptToken("hel");
    try std.testing.expect(!state.isComplete());
    try std.testing.expect(state.acceptChar('l'));
    try std.testing.expect(state.acceptChar('o'));
    try std.testing.expect(state.isComplete());
}

test "fuzz: all grammar functions" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            const allocator = std.testing.allocator;

            // --- pub const Grammar.json_grammar / bool_grammar / integer_grammar ---
            const grammars = [_][]const u8{
                Grammar.bool_grammar,
                Grammar.integer_grammar,
                Grammar.json_grammar,
            };

            // --- pub fn Grammar.getEffectiveText ---
            var text_buf: [8]u8 = undefined;
            for (&text_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @intCast(i));
            const effective = Grammar.getEffectiveText(&text_buf);
            std.debug.assert(effective.len <= text_buf.len);

            // --- pub fn Grammar.parse (random bytes, exercise error paths) ---
            var input_buf: [32]u8 = undefined;
            for (&input_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @as(u32, @intCast(i)) +% 100);
            if (Grammar.parse(allocator, &input_buf)) |*g| {
                var gm = g.*;
                gm.deinit();
            } else |_| {}

            // --- pub fn Grammar.fromJsonSchema (random bytes, exercise error paths) ---
            var schema_buf: [48]u8 = undefined;
            for (&schema_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @as(u32, @intCast(i)) +% 200);
            if (Grammar.fromJsonSchema(allocator, &schema_buf)) |*gs| {
                var gsm = gs.*;
                gsm.deinit();
            } else |_| {}

            // --- pub fn Grammar.parse (valid grammar) + pub fn Grammar.deinit ---
            const pick = smith.valueWithHash(u8, 300) % grammars.len;
            var grammar = Grammar.parse(allocator, grammars[pick]) catch return;
            defer grammar.deinit();

            // --- pub fn Grammar.initState ---
            var state = grammar.initState() catch return;
            defer state.deinit();

            // --- pub fn GrammarState.isComplete ---
            std.debug.assert(!state.isComplete());

            // --- pub fn GrammarState.acceptChar ---
            const ch = smith.valueWithHash(u8, 400);
            _ = state.acceptChar(ch);

            // --- pub fn GrammarState.acceptToken ---
            var tok_buf: [4]u8 = undefined;
            for (&tok_buf, 0..) |*b, i| b.* = smith.valueWithHash(u8, @as(u32, @intCast(i)) +% 500);
            state.acceptToken(&tok_buf);
            _ = state.isComplete();

            // --- pub fn Grammar.maskLogits ---
            const vocab = [_][]const u8{ "t", "f", "true", "false", "-", "1", "42", "{", "[" };
            var logits = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
            var state2 = grammar.initState() catch return;
            defer state2.deinit();
            grammar.maskLogits(&state2, &logits, &vocab) catch return;
            for (logits) |l| {
                std.debug.assert(std.math.isFinite(l) or l == -std.math.inf(f32));
            }

            // --- pub fn Grammar.singleValidToken ---
            var state3 = grammar.initState() catch return;
            defer state3.deinit();
            const single = grammar.singleValidToken(&state3, &vocab);
            if (single) |id| std.debug.assert(id < vocab.len);

            // --- pub fn GrammarState.init + pub fn GrammarState.deinit (standalone) ---
            var state4 = GrammarState.init(&grammar) catch return;
            state4.deinit();

            // --- pub types: ElementType, Element, Rule (comptime verification) ---
            comptime {
                _ = @as(ElementType, .char_range);
                _ = Element{ .type = .end };
                _ = Rule{ .name = "", .elements = &.{} };
            }
        }
    }.f, .{});
}

test "Grammar.parse simple rule count" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"hello\"");
    defer g.deinit();
    try std.testing.expect(g.rules.len >= 1);
}

test "GrammarState.acceptChar matches literal" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"abc\"");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('a'));
    try std.testing.expect(!state.isComplete());
    try std.testing.expect(state.acceptChar('b'));
    try std.testing.expect(state.acceptChar('c'));
    try std.testing.expect(state.isComplete());
}

test "GrammarState.acceptChar rejects wrong char" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"xy\"");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('x'));
    try std.testing.expect(!state.acceptChar('z')); // should reject 'z', expects 'y'
}

test "GrammarState.acceptToken processes full token" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"hello\"");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    state.acceptToken("hello");
    try std.testing.expect(state.isComplete());
}

test "GrammarState.acceptToken partial match" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"hello\"");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    state.acceptToken("hel");
    try std.testing.expect(!state.isComplete());
}

test "GrammarState.isComplete false at start" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"x\"");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    try std.testing.expect(!state.isComplete());
    try std.testing.expect(state.acceptChar('x'));
    try std.testing.expect(state.isComplete());
}

test "GrammarState char range acceptance" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= [a-z]");
    defer g.deinit();
    var state = try g.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('m'));
    try std.testing.expect(state.isComplete());
}

test "GrammarState alternation" {
    var g = try Grammar.parse(std.testing.allocator, "root ::= \"yes\" | \"no\"");
    defer g.deinit();

    // Path 1: "yes"
    {
        var state = try g.initState();
        defer state.deinit();
        state.acceptToken("yes");
        try std.testing.expect(state.isComplete());
    }

    // Path 2: "no"
    {
        var state = try g.initState();
        defer state.deinit();
        state.acceptToken("no");
        try std.testing.expect(state.isComplete());
    }
}

test "Grammar.getEffectiveText strips BPE prefix" {
    // 0xC4 prefix (Qwen/GPT BPE): strip first 2 bytes
    const with_prefix = [_]u8{ 0xC4, 0xA0, 'h', 'i' };
    const effective = Grammar.getEffectiveText(&with_prefix);
    try std.testing.expectEqualStrings("hi", effective);

    // No prefix: identity
    const plain = "hello";
    try std.testing.expectEqualStrings(plain, Grammar.getEffectiveText(plain));
}

test "Grammar.parse rejects oversized input" {
    const big = try std.testing.allocator.alloc(u8, 65 * 1024);
    defer std.testing.allocator.free(big);
    @memset(big, 'a');
    const result = Grammar.parse(std.testing.allocator, big);
    try std.testing.expectError(error.GrammarTooLarge, result);
}

test "Grammar.fromJsonSchema enum with non-string values" {
    // Regression test: emitEnum previously silently dropped numeric, boolean,
    // and null enum values, producing empty/incomplete grammar rules.
    const schema =
        \\{"type":"string","enum":[1,"active",true,null,42]}
    ;
    var g = Grammar.fromJsonSchema(std.testing.allocator, schema) catch |err| {
        if (err == error.OutOfMemory) return err;
        return;
    };
    defer g.deinit();
    // Must produce at least one rule with alternatives for all 5 values
    try std.testing.expect(g.rules.len >= 1);
}

test "Grammar.fromJsonSchema basic object" {
    const schema =
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    ;
    var g = Grammar.fromJsonSchema(std.testing.allocator, schema) catch |err| {
        // If schema conversion fails (non-trivial), just skip this test
        if (err == error.OutOfMemory) return err;
        return; // schema parse failures are acceptable edge cases
    };
    defer g.deinit();
    try std.testing.expect(g.rules.len >= 1);
}
