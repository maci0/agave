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

/// MTLSize struct — { width, height, depth }
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
