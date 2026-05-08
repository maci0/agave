//! GBNF grammar parser and constrained decoding state machine.
//!
//! Parses GBNF (GGML BNF) grammar strings into a rule-based representation,
//! then provides token-level constraint checking during generation.
//!
//! Usage:
//!   const grammar = try Grammar.parse(allocator, gbnf_text);
//!   defer grammar.deinit();
//!   var state = grammar.initState();
//!   // In generation loop:
//!   grammar.maskLogits(&state, logits, tokenizer);
//!   // After sampling:
//!   grammar.acceptToken(&state, token_text);

const std = @import("std");

// ── Grammar Elements ────────────────────────────────────────────

pub const ElementType = enum {
    char_range, // Match character in range [lo..hi]
    char_not, // Match character NOT in range [lo..hi]
    char_range_star, // Match character in range, zero or more times (*)
    char_range_plus, // Match character in range, one or more times (+)
    char_range_opt, // Match character in range, zero or one time (?)
    rule_ref, // Reference to another rule by index
    rule_ref_star, // Reference to another rule, zero or more times (*)
    rule_ref_plus, // Reference to another rule, one or more times (+)
    rule_ref_opt, // Reference to another rule, zero or one time (?)
    alt, // Alternative separator (|)
    end, // End of alternative/rule
};

pub const Element = struct {
    type: ElementType,
    lo: u32 = 0, // For char_range/char_not: low codepoint. For rule_ref: rule index.
    hi: u32 = 0, // For char_range/char_not: high codepoint (inclusive).
};

pub const Rule = struct {
    name: []const u8,
    elements: []Element,
};

// ── Grammar ─────────────────────────────────────────────────────

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
        \\string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
        \\number ::= "-"? [0-9]+ ("." [0-9]+)?
        \\ws     ::= [ \t\n\r]*
    ;

    pub const bool_grammar = "root ::= \"true\" | \"false\"";
    pub const integer_grammar = "root ::= \"-\"? [0-9]+";

    /// Convert a JSON schema to GBNF grammar and parse it.
    /// Supports: object (with properties), string, number, integer, boolean,
    /// array (with items), enum, and nested schemas.
    pub fn fromJsonSchema(allocator: std.mem.Allocator, schema_json: []const u8) !Grammar {
        var converter = SchemaConverter.init(allocator);
        defer converter.deinit();
        const gbnf = try converter.convert(schema_json);
        defer allocator.free(gbnf);
        return parse(allocator, gbnf);
    }

    pub fn parse(allocator: std.mem.Allocator, input: []const u8) !Grammar {
        var parser = Parser.init(allocator, input);
        return parser.parseGrammar();
    }

    pub fn deinit(self: *Grammar) void {
        for (self.rules) |rule| {
            self.allocator.free(rule.elements);
        }
        self.allocator.free(self.rules);
        self.rule_names.deinit();
    }

    pub fn initState(self: *const Grammar) GrammarState {
        return GrammarState.init(self);
    }

    /// Mask logits for tokens that don't match the grammar.
    /// Sets disallowed token logits to -inf.
    pub fn maskLogits(self: *const Grammar, state: *GrammarState, logits: []f32, vocab: []const []const u8) void {
        if (state.completed) return;
        if (state.stack.items.len == 0) return;

        for (logits, 0..) |*logit, token_id| {
            if (token_id >= vocab.len) break;
            const text = vocab[token_id];
            if (text.len == 0) continue;

            const effective = getEffectiveText(text);
            if (effective.len == 0) {
                // BPE prefix-only tokens (spaces, newlines) — mask them too
                logit.* = -std.math.inf(f32);
                continue;
            }

            // Try accepting entire token in a copy of the state
            // If any byte fails, the token is disallowed
            if (!self.isTokenValidPrefix(state, effective)) {
                logit.* = -std.math.inf(f32);
            }
        }
    }

    fn isTokenValidPrefix(self: *const Grammar, state: *const GrammarState, text: []const u8) bool {
        // Clone the state stack to test without modifying the original
        var test_state = GrammarState{
            .grammar = self,
            .stack = std.ArrayList(StackEntry).empty,
            .completed = state.completed,
        };
        for (state.stack.items) |entry| {
            test_state.stack.append(self.allocator, entry) catch return true;
        }
        defer test_state.stack.deinit(self.allocator);

        for (text) |c| {
            if (test_state.completed) return true;
            if (!test_state.acceptChar(c)) return false;
        }
        return true;
    }

    /// Strip BPE byte-level encoding prefix to get actual text.
    /// Qwen/GPT uses Ġ (0xC4 0xA0) for space, Ċ (0xC4 0x8A) for newline, etc.
    pub fn getEffectiveText(text: []const u8) []const u8 {
        if (text.len >= 2 and text[0] == 0xC4) {
            // Ġ = space prefix (0xC4 0xA0), Ċ = newline (0xC4 0x8A)
            return text[2..];
        }
        if (text.len >= 3 and text[0] == 0xC3) {
            // Other byte-level BPE encodings
            return text[2..];
        }
        return text;
    }

    fn collectAllowedChars(self: *const Grammar, state: *const GrammarState, lo: *[256]u32, hi: *[256]u32, count: *usize) void {
        if (state.stack.items.len == 0) return;
        const top = state.stack.items[state.stack.items.len - 1];
        if (top.rule_id >= self.rules.len) return;
        const rule = self.rules[top.rule_id];
        self.collectFromRule(rule.elements, top.elem_idx, lo, hi, count, 0);
    }

    fn collectFromRule(self: *const Grammar, elements: []const Element, start_idx: u32, lo: *[256]u32, hi: *[256]u32, count: *usize, depth: u32) void {
        if (depth > 16 or count.* >= 256) return;
        if (start_idx >= elements.len) return;

        // Collect first chars from current alternative and all alternatives after |
        var idx = start_idx;
        var found_in_alt = false;
        while (idx < elements.len) {
            const elem = elements[idx];
            switch (elem.type) {
                .char_range, .char_range_star, .char_range_plus, .char_range_opt => {
                    if (!found_in_alt and count.* < 256) {
                        lo.*[count.*] = elem.lo;
                        hi.*[count.*] = elem.hi;
                        count.* += 1;
                    }
                    found_in_alt = true;
                    // For * and ?, also collect next element's chars (zero matches possible)
                    if (elem.type == .char_range_star or elem.type == .char_range_opt) {
                        if (idx + 1 < elements.len) {
                            const next_elem = elements[idx + 1];
                            if ((next_elem.type == .char_range or next_elem.type == .char_range_star or next_elem.type == .char_range_plus) and count.* < 256) {
                                lo.*[count.*] = next_elem.lo;
                                hi.*[count.*] = next_elem.hi;
                                count.* += 1;
                            }
                        }
                    }
                    idx += 1;
                    while (idx < elements.len and elements[idx].type != .alt and elements[idx].type != .end) : (idx += 1) {}
                    found_in_alt = false;
                    continue;
                },
                .char_not => {
                    if (!found_in_alt and count.* + 2 <= 256) {
                        if (elem.lo > 0) {
                            lo.*[count.*] = 0;
                            hi.*[count.*] = elem.lo - 1;
                            count.* += 1;
                        }
                        if (elem.hi < 255) {
                            lo.*[count.*] = elem.hi + 1;
                            hi.*[count.*] = 255;
                            count.* += 1;
                        }
                    }
                    found_in_alt = true;
                    idx += 1;
                    while (idx < elements.len and elements[idx].type != .alt and elements[idx].type != .end) : (idx += 1) {}
                    found_in_alt = false;
                    continue;
                },
                .rule_ref, .rule_ref_star, .rule_ref_opt, .rule_ref_plus => {
                    if (!found_in_alt and elem.lo < self.rules.len) {
                        self.collectFromRule(self.rules[elem.lo].elements, 0, lo, hi, count, depth + 1);
                    }
                    found_in_alt = true;
                    // For rule_ref_star/opt, also collect next element (zero matches possible)
                    if (elem.type == .rule_ref_star or elem.type == .rule_ref_opt) {
                        if (idx + 1 < elements.len) {
                            const next_elem = elements[idx + 1];
                            switch (next_elem.type) {
                                .char_range, .char_range_star, .char_range_plus => {
                                    if (count.* < 256) {
                                        lo.*[count.*] = next_elem.lo;
                                        hi.*[count.*] = next_elem.hi;
                                        count.* += 1;
                                    }
                                },
                                .rule_ref, .rule_ref_star, .rule_ref_opt, .rule_ref_plus => {
                                    if (next_elem.lo < self.rules.len) {
                                        self.collectFromRule(self.rules[next_elem.lo].elements, 0, lo, hi, count, depth + 1);
                                    }
                                },
                                else => {},
                            }
                        }
                    }
                    idx += 1;
                    while (idx < elements.len and elements[idx].type != .alt and elements[idx].type != .end) : (idx += 1) {}
                    found_in_alt = false;
                    continue;
                },
                .alt => {
                    idx += 1;
                    found_in_alt = false;
                    continue;
                },
                .end => return,
            }
        }
    }
};

// ── Grammar State ───────────────────────────────────────────────

const StackEntry = struct {
    rule_id: u32,
    elem_idx: u32,
};

pub const GrammarState = struct {
    grammar: *const Grammar,
    stack: std.ArrayList(StackEntry),
    completed: bool = false,

    pub fn init(grammar: *const Grammar) GrammarState {
        var state = GrammarState{
            .grammar = grammar,
            .stack = std.ArrayList(StackEntry).empty,
        };
        state.stack.append(grammar.allocator, .{ .rule_id = grammar.root_id, .elem_idx = 0 }) catch {};
        return state;
    }

    pub fn deinit(self: *GrammarState) void {
        self.stack.deinit(self.grammar.allocator);
    }

    pub fn acceptChar(self: *GrammarState, c: u8) bool {
        return self.acceptCharInner(c, 0);
    }

    fn acceptCharInner(self: *GrammarState, c: u8, depth: u32) bool {
        if (depth > 32 or self.completed or self.stack.items.len == 0) return false;

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
                if (c >= @as(u8, @intCast(elem.lo)) and c <= @as(u8, @intCast(elem.hi))) {
                    top.elem_idx += 1;
                    self.advancePastEnd();
                    return true;
                }
                // Try next alternative in this rule
                return self.tryNextAlternative(c, depth);
            },
            .char_not => {
                if (c < @as(u8, @intCast(elem.lo)) or c > @as(u8, @intCast(elem.hi))) {
                    top.elem_idx += 1;
                    self.advancePastEnd();
                    return true;
                }
                return self.tryNextAlternative(c, depth);
            },
            .char_range_star => {
                if (c >= @as(u8, @intCast(elem.lo)) and c <= @as(u8, @intCast(elem.hi))) {
                    return true; // match — stay at this element
                }
                // no match — advance past (zero matches ok)
                top.elem_idx += 1;
                self.advancePastEnd();
                return self.acceptCharInner(c, depth + 1);
            },
            .char_range_plus => unreachable, // decomposed to char_range + char_range_star by parser
            .char_range_opt => {
                // ? = zero or one
                if (c >= @as(u8, @intCast(elem.lo)) and c <= @as(u8, @intCast(elem.hi))) {
                    top.elem_idx += 1;
                    self.advancePastEnd();
                    return true;
                }
                // No match — skip (zero matches ok)
                top.elem_idx += 1;
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
                // Subrule didn't match — restore and advance past
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
                // Didn't match — restore and try without
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
        // Scan forward to find next | in this rule
        var idx = top.elem_idx;
        while (idx < rule.elements.len) : (idx += 1) {
            if (rule.elements[idx].type == .alt) {
                top.elem_idx = idx + 1;
                return self.acceptCharInner(c, depth + 1);
            }
            if (rule.elements[idx].type == .end) break;
        }
        return false;
    }

    pub fn acceptToken(self: *GrammarState, text: []const u8) void {
        const effective = Grammar.getEffectiveText(text);
        for (effective) |c| {
            if (!self.acceptChar(c)) break;
        }
    }

    pub fn isComplete(self: *const GrammarState) bool {
        return self.completed;
    }
};

// ── GBNF Parser ─────────────────────────────────────────────────

const UnresolvedRef = struct {
    rule_idx: u32, // which rule this ref is in
    elem_idx: u32, // which element within the rule
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
                    // Pre-register with placeholder — actual ID assigned in pass 2
                    if (!self.rule_names.contains(name)) {
                        try self.rule_names.put(name, 0xFFFF);
                    }
                }
                // Skip to next line
                while (scan_pos < self.input.len and self.input[scan_pos] != '\n') : (scan_pos += 1) {}
                if (scan_pos < self.input.len) scan_pos += 1;
            }
        }

        // Pass 2: parse rule bodies (synthetic rules from groups get real IDs)
        while (self.pos < self.input.len) {
            self.skipWs();
            if (self.pos >= self.input.len) break;
            if (self.input[self.pos] == '#' or self.input[self.pos] == '\n') {
                self.skipLine();
                continue;
            }
            try self.parseRule();
        }

        self.elements.deinit(self.allocator);

        // Resolve forward references: each has a unique placeholder 0xFF00+i
        for (self.unresolved.items, 0..) |ref, i| {
            const placeholder: u32 = 0xFF00 + @as(u32, @intCast(i));
            const resolved_id = self.rule_names.get(ref.name) orelse continue;
            if (resolved_id == 0xFFFF) continue;
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
        const root_id = self.rule_names.get("root") orelse 0;
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

            // Add synthetic rule
            const synth_id: u32 = @intCast(self.rules.items.len);
            try self.rules.append(self.allocator, .{ .name = "_group", .elements = group_elems });

            // Emit rule_ref to the synthetic rule
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = synth_id });
        } else if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-') {
            try self.parseRuleRef();
        } else {
            self.pos += 1; // skip unknown
        }

        // Handle repetition modifiers: *, +, ?
        if (self.pos < self.input.len) {
            const mod = self.input[self.pos];
            if (mod == '*' or mod == '+' or mod == '?') {
                self.pos += 1;
                if (self.elements.items.len > 0) {
                    const last = &self.elements.items[self.elements.items.len - 1];
                    switch (last.type) {
                        .char_range, .char_not => {
                            if (mod == '+') {
                                // x+ → x x* (one mandatory + zero or more)
                                const lo = last.lo;
                                const hi = last.hi;
                                const star_type: ElementType = if (last.type == .char_not) .char_range_star else .char_range_star;
                                try self.elements.append(self.allocator, .{ .type = star_type, .lo = lo, .hi = hi });
                            } else {
                                last.type = switch (mod) {
                                    '*' => .char_range_star,
                                    '?' => .char_range_opt,
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
        self.pos += 1; // skip opening "
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
        if (self.pos < self.input.len) self.pos += 1; // skip closing "
    }

    fn parseCharClass(self: *Parser) !void {
        self.pos += 1; // skip [
        var negate = false;
        if (self.pos < self.input.len and self.input[self.pos] == '^') {
            negate = true;
            self.pos += 1;
        }
        while (self.pos < self.input.len and self.input[self.pos] != ']') {
            const lo = self.input[self.pos];
            self.pos += 1;
            var hi = lo;
            if (self.pos + 1 < self.input.len and self.input[self.pos] == '-' and self.input[self.pos + 1] != ']') {
                self.pos += 1; // skip -
                hi = self.input[self.pos];
                self.pos += 1;
            }
            const elem_type: ElementType = if (negate) .char_not else .char_range;
            try self.elements.append(self.allocator, .{ .type = elem_type, .lo = lo, .hi = hi });
        }
        if (self.pos < self.input.len) self.pos += 1; // skip ]
    }

    fn parseRuleRef(self: *Parser) !void {
        const start = self.pos;
        while (self.pos < self.input.len and (std.ascii.isAlphanumeric(self.input[self.pos]) or self.input[self.pos] == '_' or self.input[self.pos] == '-')) : (self.pos += 1) {}
        const name = self.input[start..self.pos];
        const rule_id = self.rule_names.get(name) orelse 0;
        if (rule_id != 0xFFFF) {
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = rule_id });
        } else {
            // Forward reference — unique placeholder per ref
            const placeholder: u32 = 0xFF00 + @as(u32, @intCast(self.unresolved.items.len));
            try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = placeholder });
            try self.unresolved.append(self.allocator, .{
                .rule_idx = 0,
                .elem_idx = 0,
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
    rule_count: u32 = 0,

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

        try self.emitRule("root", schema_json);
        return self.rules.toOwnedSlice(self.allocator);
    }

    const ConvertError = error{OutOfMemory};
    fn emitRule(self: *SchemaConverter, name: []const u8, schema: []const u8) ConvertError!void {
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
                try self.emitObject(name, schema);
            } else if (std.mem.eql(u8, t, "array")) {
                try self.emitArray(name, schema);
            } else if (std.mem.eql(u8, t, "string")) {
                try self.emit(name);
                try self.emit(" ::= string\n");
            } else if (std.mem.eql(u8, t, "number")) {
                try self.emit(name);
                try self.emit(" ::= number\n");
            } else if (std.mem.eql(u8, t, "integer")) {
                try self.emit(name);
                try self.emit(" ::= integer\n");
            } else if (std.mem.eql(u8, t, "boolean")) {
                try self.emit(name);
                try self.emit(" ::= boolean\n");
            } else if (std.mem.eql(u8, t, "null")) {
                try self.emit(name);
                try self.emit(" ::= null\n");
            } else {
                try self.emit(name);
                try self.emit(" ::= value\n");
            }
        } else {
            try self.emit(name);
            try self.emit(" ::= value\n");
        }
    }

    fn emitObject(self: *SchemaConverter, name: []const u8, schema: []const u8) ConvertError!void {
        const props_content = findJsonObject(schema, "properties") orelse {
            try self.emit(name);
            try self.emit(" ::= \"{\" ws \"}\" | \"{\" ws string ws \":\" ws value (ws \",\" ws string ws \":\" ws value)* ws \"}\"\n");
            return;
        };

        // Parse property names and their schemas
        var prop_names: [32][]const u8 = undefined;
        var prop_schemas: [32][]const u8 = undefined;
        var n_props: usize = 0;

        var pi: usize = 0;
        while (pi < props_content.len and n_props < 32) {
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
            try self.emitRule(sub_name, prop_schemas[i]);
            try self.emit(sub_name);
        }
        try self.emit(" ws \"}\"\n");
    }

    fn emitArray(self: *SchemaConverter, name: []const u8, schema: []const u8) ConvertError!void {
        const items_schema = findJsonObject(schema, "items");

        var item_rule: []const u8 = "value";
        var sub_name_buf: [64]u8 = undefined;
        if (items_schema) |is| {
            const sub = std.fmt.bufPrint(&sub_name_buf, "{s}-item", .{name}) catch "item";
            try self.emitRule(sub, is);
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

    fn findJsonObject(json: []const u8, field: []const u8) ?[]const u8 {
        var buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
        const idx = std.mem.indexOf(u8, json, needle) orelse return null;
        var i = idx + needle.len;
        while (i < json.len and (json[i] == ' ' or json[i] == ':')) : (i += 1) {}
        if (i >= json.len or json[i] != '{') return null;
        const start = i + 1;
        var depth: i32 = 1;
        i += 1;
        while (i < json.len and depth > 0) : (i += 1) {
            if (json[i] == '{') depth += 1;
            if (json[i] == '}') depth -= 1;
        }
        return json[start .. i - 1];
    }

    fn findJsonArray(json: []const u8, field: []const u8) ?[]const u8 {
        var buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&buf, "\"{s}\"", .{field}) catch return null;
        const idx = std.mem.indexOf(u8, json, needle) orelse return null;
        var i = idx + needle.len;
        while (i < json.len and (json[i] == ' ' or json[i] == ':')) : (i += 1) {}
        if (i >= json.len or json[i] != '[') return null;
        const start = i + 1;
        var depth: i32 = 1;
        i += 1;
        while (i < json.len and depth > 0) : (i += 1) {
            if (json[i] == '[') depth += 1;
            if (json[i] == ']') depth -= 1;
        }
        return json[start .. i - 1];
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
                    if (json[i] == '\\') i += 1;
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
    try std.testing.expect(grammar.rules.len >= 1);
}

test "char repetition star" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= [a-z]*");
    defer grammar.deinit();
    var state = grammar.initState();
    defer state.deinit();

    // Zero matches — should complete immediately (empty string valid)
    try std.testing.expect(state.acceptChar('1') == false);
}

test "char repetition plus" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, "root ::= [0-9]+");
    defer grammar.deinit();
    var state = grammar.initState();
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
    var state = grammar.initState();
    defer state.deinit();

    // Cannot start with 'x' — need at least one digit first
    try std.testing.expect(state.acceptChar('x') == false);
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

    var state = grammar.initState();
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
    try std.testing.expect(grammar.rules.len >= 6);

    var state = grammar.initState();
    defer state.deinit();

    // Should accept start of valid JSON
    try std.testing.expect(state.acceptChar('{'));
}

test "integer grammar" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.parse(allocator, Grammar.integer_grammar);
    defer grammar.deinit();
    var state = grammar.initState();
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
    var state = grammar.initState();
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
    var state = grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('"'));
    try std.testing.expect(state.acceptChar('h'));
    try std.testing.expect(state.acceptChar('i'));
    try std.testing.expect(state.acceptChar('"'));
}

test "json schema boolean type" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"type\": \"boolean\"}");
    defer grammar.deinit();
    var state = grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('t'));
    try std.testing.expect(state.acceptChar('r'));
    try std.testing.expect(state.acceptChar('u'));
    try std.testing.expect(state.acceptChar('e'));
}

test "json schema enum" {
    const allocator = std.testing.allocator;
    var grammar = try Grammar.fromJsonSchema(allocator, "{\"enum\": [\"red\", \"green\", \"blue\"]}");
    defer grammar.deinit();
    var state = grammar.initState();
    defer state.deinit();

    try std.testing.expect(state.acceptChar('r'));
    try std.testing.expect(state.acceptChar('e'));
    try std.testing.expect(state.acceptChar('d'));
}

test "json schema object" {
    const allocator = std.testing.allocator;
    const schema =
        \\{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    ;
    var grammar = try Grammar.fromJsonSchema(allocator, schema);
    defer grammar.deinit();
    var state = grammar.initState();
    defer state.deinit();

    // Should accept opening brace
    try std.testing.expect(state.acceptChar('{'));
}
