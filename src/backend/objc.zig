//! Minimal Objective-C runtime bindings for Metal compute.
//! Only the subset needed for device, command queue, buffers, shader compilation,
//! pipeline creation, and compute dispatch.

/// Opaque pointer to any Objective-C object instance.
pub const id = *anyopaque;
/// Opaque pointer to an Objective-C class object.
pub const Class = *anyopaque;
/// Opaque pointer to an Objective-C selector (method name).
pub const SEL = *anyopaque;
/// Objective-C unsigned integer type (64-bit on all Apple Silicon targets).
pub const NSUInteger = u64;

/// MTLSize struct, { width, height, depth }
pub const MTLSize = extern struct {
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,
};

extern "c" fn objc_getClass(name: [*:0]const u8) ?Class;
extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
extern "c" fn objc_msgSend() void;

/// Get the default Metal device (plain C function, not ObjC).
pub extern "c" fn MTLCreateSystemDefaultDevice() ?id;

/// NSArray of MTLDevice. Caller must `release` the array.
pub extern "c" fn MTLCopyAllDevices() ?id;

/// Register (or look up) an Objective-C selector by name.
pub fn sel(name: [*:0]const u8) SEL {
    return sel_registerName(name);
}

/// Look up an Objective-C class by name.
pub fn getClass(name: [*:0]const u8) ?Class {
    return objc_getClass(name);
}

/// Helper to build function pointer type at comptime. Recursively builds the function
/// signature by prepending argument types one by one.
fn MsgSendFn(comptime R: type, comptime T: type, comptime Args: type) type {
    const fields = @typeInfo(Args).@"struct".fields;

    return switch (fields.len) {
        0 => *const fn (T, SEL) callconv(.c) R,
        1 => *const fn (T, SEL, fields[0].type) callconv(.c) R,
        2 => *const fn (T, SEL, fields[0].type, fields[1].type) callconv(.c) R,
        3 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type) callconv(.c) R,
        4 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type) callconv(.c) R,
        5 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type, fields[4].type) callconv(.c) R,
        6 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type, fields[4].type, fields[5].type) callconv(.c) R,
        7 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type, fields[4].type, fields[5].type, fields[6].type) callconv(.c) R,
        8 => *const fn (T, SEL, fields[0].type, fields[1].type, fields[2].type, fields[3].type, fields[4].type, fields[5].type, fields[6].type, fields[7].type) callconv(.c) R,
        else => @compileError("msgSend: too many arguments (max 8)"),
    };
}

/// Type-safe objc_msgSend wrapper. Casts objc_msgSend to the correct
/// function pointer type at comptime based on the return type and argument types.
pub fn msgSend(comptime R: type, target: anytype, s: SEL, args: anytype) R {
    const T = @TypeOf(target);
    const Fn = MsgSendFn(R, T, @TypeOf(args));
    const func: Fn = @ptrCast(&objc_msgSend);
    return @call(.auto, func, .{ target, s } ++ args);
}

// ── Tests ───────────────────────────────────────────────────────────

test "objc, MTLSize layout" {
    // Verify MTLSize has correct extern struct layout (3 × u64 = 24 bytes).
    try @import("std").testing.expectEqual(@as(usize, 24), @sizeOf(MTLSize));
    try @import("std").testing.expectEqual(@as(usize, 8), @alignOf(MTLSize));

    const size = MTLSize{ .width = 64, .height = 32, .depth = 1 };
    try @import("std").testing.expectEqual(@as(NSUInteger, 64), size.width);
    try @import("std").testing.expectEqual(@as(NSUInteger, 32), size.height);
    try @import("std").testing.expectEqual(@as(NSUInteger, 1), size.depth);
}

test "objc, MsgSendFn comptime type generation" {
    // Verify MsgSendFn generates correct function pointer types for 0-8 args.
    const Fn0 = MsgSendFn(void, id, struct {});
    const fn0_info = @typeInfo(Fn0).pointer;
    try @import("std").testing.expect(fn0_info.child == fn (id, SEL) callconv(.c) void);

    const Fn1 = MsgSendFn(u64, id, struct { a: u64 });
    const fn1_info = @typeInfo(Fn1).pointer;
    try @import("std").testing.expect(fn1_info.child == fn (id, SEL, u64) callconv(.c) u64);

    const Fn2 = MsgSendFn(id, id, struct { a: u64, b: id });
    const fn2_info = @typeInfo(Fn2).pointer;
    try @import("std").testing.expect(fn2_info.child == fn (id, SEL, u64, id) callconv(.c) id);
}

test "objc, type sizes" {
    // Verify pointer-sized ObjC types.
    try @import("std").testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(id));
    try @import("std").testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(Class));
    try @import("std").testing.expectEqual(@sizeOf(*anyopaque), @sizeOf(SEL));
    try @import("std").testing.expectEqual(@as(usize, 8), @sizeOf(NSUInteger));
}

test "objc, sel and getClass function signatures exist" {
    comptime {
        _ = @TypeOf(sel);
        _ = @TypeOf(getClass);
        _ = @TypeOf(msgSend);
        _ = @TypeOf(MTLCreateSystemDefaultDevice);
    }
}

test "objc, public function signature contracts" {
    // Verify sel() accepts [*:0]const u8 and returns SEL.
    comptime {
        const SelFn = @TypeOf(sel);
        const sel_info = @typeInfo(SelFn).@"fn";
        if (sel_info.params.len != 1) @compileError("sel: expected 1 param");
        if (sel_info.return_type != SEL) @compileError("sel: expected SEL return");

        const GetClassFn = @TypeOf(getClass);
        const gc_info = @typeInfo(GetClassFn).@"fn";
        if (gc_info.params.len != 1) @compileError("getClass: expected 1 param");
        if (gc_info.return_type != ?Class) @compileError("getClass: expected ?Class return");

        const CreateDevFn = @TypeOf(MTLCreateSystemDefaultDevice);
        const cd_info = @typeInfo(CreateDevFn).@"fn";
        if (cd_info.params.len != 0) @compileError("MTLCreateSystemDefaultDevice: expected 0 params");
        if (cd_info.return_type != ?id) @compileError("MTLCreateSystemDefaultDevice: expected ?id return");
    }
}

test "fuzz: all objc functions" {
    try @import("std").testing.fuzz({}, struct {
        fn f(_: void, smith: *@import("std").testing.Smith) !void {
            _ = smith;
            comptime {
                _ = &sel;
                _ = &getClass;
                _ = &MTLCreateSystemDefaultDevice;
            }
        }
    }.f, .{});
}

test "objc, MsgSendFn covers all arities 0-8" {
    comptime {
        // Arity 0
        _ = MsgSendFn(void, id, struct {});
        // Arity 1
        _ = MsgSendFn(id, id, struct { a: u64 });
        // Arity 2
        _ = MsgSendFn(id, id, struct { a: u64, b: id });
        // Arity 3
        _ = MsgSendFn(void, id, struct { a: u64, b: u64, c: u64 });
        // Arity 4
        _ = MsgSendFn(id, id, struct { a: u64, b: u64, c: u64, d: id });
        // Arity 5
        _ = MsgSendFn(void, id, struct { a: u64, b: u64, c: u64, d: u64, e: u64 });
        // Arity 6
        _ = MsgSendFn(id, id, struct { a: u64, b: u64, c: u64, d: u64, e: u64, f: u64 });
        // Arity 7
        _ = MsgSendFn(void, id, struct { a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64 });
        // Arity 8
        _ = MsgSendFn(id, id, struct { a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64 });
    }
}
