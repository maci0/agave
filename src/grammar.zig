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
        const allowed = self.getAllowedChars(state);
        for (logits, 0..) |*logit, token_id| {
            if (token_id >= vocab.len) break;
            const text = vocab[token_id];
            if (text.len == 0) continue;
            if (!isTokenAllowed(text, allowed)) {
                logit.* = -std.math.inf(f32);
            }
        }
    }

    fn getAllowedChars(self: *const Grammar, state: *const GrammarState) []const Element {
        if (state.stack.items.len == 0) return &.{};
        const top = state.stack.items[state.stack.items.len - 1];
        const rule = self.rules[top.rule_id];
        if (top.elem_idx >= rule.elements.len) return &.{};

        // Find current alternative's elements starting from elem_idx
        const start = top.elem_idx;
        var end = start;
        while (end < rule.elements.len and rule.elements[end].type != .alt and rule.elements[end].type != .end) : (end += 1) {}

        if (start >= end) return &.{};
        const elem = rule.elements[start];
        if (elem.type == .char_range or elem.type == .char_not) {
            return rule.elements[start..end];
        }
        return &.{};
    }

    fn isTokenAllowed(text: []const u8, allowed: []const Element) bool {
        if (allowed.len == 0) return true; // No constraints = allow all
        if (text.len == 0) return false;

        // Check first byte against allowed char ranges
        const first_char: u32 = text[0];
        for (allowed) |elem| {
            switch (elem.type) {
                .char_range => {
                    if (first_char >= elem.lo and first_char <= elem.hi) return true;
                },
                .char_not => {
                    if (first_char < elem.lo or first_char > elem.hi) return true;
                },
                else => {},
            }
        }
        return false;
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
        if (self.completed or self.stack.items.len == 0) return false;

        const top = &self.stack.items[self.stack.items.len - 1];
        const rule = self.grammar.rules[top.rule_id];
        if (top.elem_idx >= rule.elements.len) {
            self.completed = true;
            return false;
        }

        const elem = rule.elements[top.elem_idx];
        switch (elem.type) {
            .char_range => {
                if (c >= @as(u8, @intCast(elem.lo)) and c <= @as(u8, @intCast(elem.hi))) {
                    top.elem_idx += 1;
                    // Check if we reached end of rule
                    if (top.elem_idx >= rule.elements.len or rule.elements[top.elem_idx].type == .end) {
                        _ = self.stack.pop();
                        if (self.stack.items.len == 0) self.completed = true;
                    }
                    return true;
                }
                return false;
            },
            .rule_ref => {
                // Push referenced rule
                self.stack.append(self.grammar.allocator, .{ .rule_id = elem.lo, .elem_idx = 0 }) catch return false;
                top.elem_idx += 1;
                return self.acceptChar(c);
            },
            else => return false,
        }
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
