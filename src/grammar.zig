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
    rule_ref, // Reference to another rule by index
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
    elements: []const Element,
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

        // Collect all allowed first-chars from current grammar position
        var allowed_lo: [256]u32 = undefined;
        var allowed_hi: [256]u32 = undefined;
        var n_allowed: usize = 0;
        self.collectAllowedChars(state, &allowed_lo, &allowed_hi, &n_allowed);

        if (n_allowed == 0) return; // No constraints

        for (logits, 0..) |*logit, token_id| {
            if (token_id >= vocab.len) break;
            const text = vocab[token_id];
            if (text.len == 0) continue;
            const first: u32 = text[0];
            var ok = false;
            for (0..n_allowed) |i| {
                if (first >= allowed_lo[i] and first <= allowed_hi[i]) {
                    ok = true;
                    break;
                }
            }
            if (!ok) logit.* = -std.math.inf(f32);
        }
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
                .char_range => {
                    if (!found_in_alt and count.* < 256) {
                        lo.*[count.*] = elem.lo;
                        hi.*[count.*] = elem.hi;
                        count.* += 1;
                    }
                    found_in_alt = true;
                    // Skip rest of this alternative to find next |
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
                .rule_ref => {
                    if (!found_in_alt and elem.lo < self.rules.len) {
                        self.collectFromRule(self.rules[elem.lo].elements, 0, lo, hi, count, depth + 1);
                    }
                    found_in_alt = true;
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
            .rule_ref => {
                self.stack.append(self.grammar.allocator, .{ .rule_id = elem.lo, .elem_idx = 0 }) catch return false;
                top.elem_idx += 1;
                return self.acceptCharInner(c, depth + 1);
            },
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
        for (text) |c| {
            if (!self.acceptChar(c)) break;
        }
    }

    pub fn isComplete(self: *const GrammarState) bool {
        return self.completed;
    }
};

// ── GBNF Parser ─────────────────────────────────────────────────

const Parser = struct {
    allocator: std.mem.Allocator,
    input: []const u8,
    pos: usize = 0,
    rules: std.ArrayList(Rule),
    rule_names: std.StringHashMap(u32),
    elements: std.ArrayList(Element),

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
        while (self.pos < self.input.len) {
            self.skipWs();
            if (self.pos >= self.input.len) break;
            if (self.input[self.pos] == '#') {
                self.skipLine();
                continue;
            }
            try self.parseRule();
        }

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

        const rule_id: u32 = @intCast(self.rules.items.len);
        try self.rule_names.put(name, rule_id);

        const elem_start = self.elements.items.len;
        try self.parseAlternatives();
        try self.elements.append(self.allocator, .{ .type = .end });

        const elems = try self.allocator.dupe(Element, self.elements.items[elem_start..]);
        self.elements.shrinkRetainingCapacity(elem_start);

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
            try self.parseAlternatives();
            if (self.pos < self.input.len and self.input[self.pos] == ')') self.pos += 1;
        } else if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-') {
            try self.parseRuleRef();
        } else {
            self.pos += 1; // skip unknown
        }

        // Handle repetition: *, +, ?
        if (self.pos < self.input.len) {
            const mod = self.input[self.pos];
            if (mod == '*' or mod == '+' or mod == '?') {
                self.pos += 1;
                // Simplified: don't implement repetition tracking in state machine yet
                // Just allow the element (correct for basic grammars)
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
        const rule_id = self.rule_names.get(name) orelse blk: {
            // Forward reference — assign next ID
            const id: u32 = @intCast(self.rules.items.len + self.rule_names.count());
            self.rule_names.put(name, id) catch {};
            break :blk id;
        };
        try self.elements.append(self.allocator, .{ .type = .rule_ref, .lo = rule_id });
    }

    fn skipWs(self: *Parser) void {
        while (self.pos < self.input.len and (self.input[self.pos] == ' ' or self.input[self.pos] == '\t' or self.input[self.pos] == '\r')) : (self.pos += 1) {}
    }

    fn skipLine(self: *Parser) void {
        while (self.pos < self.input.len and self.input[self.pos] != '\n') : (self.pos += 1) {}
        if (self.pos < self.input.len) self.pos += 1;
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
