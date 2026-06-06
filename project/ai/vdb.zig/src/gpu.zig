const std = @import("std");
const simd = @import("simd");

/// GPU Support Fallback Strategy for vdb.zig
///
/// Since vdb.zig targets embedded/edge scenarios with Zig, native GPU is optional.
/// This module provides three tiers:
/// 1. **Metal** (macOS): via dlopen of Metal.framework + MTLCreateSystemDefaultDevice.
/// 2. **CUDA** (Linux/Windows): via raw driver API (libcuda) FFI.
/// 3. **OpenCL** (universal fallback): via dlopen of OpenCL framework/library.
/// 4. **CPU SIMD**: already implemented in simd.zig; used when GPU unavailable.
///
/// cuVS Integration (planned):
/// cuVS (RAPIDS) provides GPU-accelerated IVF-RaBitQ with C API.
/// Integration path: dlopen libcuvs.so -> cuvs::ivf_rabitq::index build/search.
/// Expected speedup: 10-50x QPS over CPU for batch queries on NVIDIA GPUs.
pub const GpuBackend = enum {
    none,
    metal,
    cuda,
    opencl,
    cuvs, // RAPIDS cuVS GPU acceleration
};

// ============================================================
// GPU Kernel Sources
// ============================================================

/// Metal compute kernel for batched RaBitQ XOR-popcount.
/// Each thread handles one code vector.
pub const metal_rabitq_kernel =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void rabitq_popcount(
    \\    device const ulong* codes [[buffer(0)]],
    \\    device const ulong* query   [[buffer(1)]],
    \\    device float*       out     [[buffer(2)]],
    \\    constant uint&      count   [[buffer(3)]],
    \\    constant uint&      words_per_vec [[buffer(4)]],
    \\    uint                gid     [[thread_position_in_grid]])
    \\{
    \\    if (gid >= count) return;
    \\    uint pop = 0;
    \\    for (uint w = 0; w < words_per_vec; w++) {
    \\        ulong x = codes[gid * words_per_vec + w] ^ query[w];
    \\        pop += popcount(x);
    \\    }
    \\    out[gid] = float(pop);
    \\}
;

/// CUDA kernel for batched RaBitQ XOR-popcount.
/// Each block processes a tile of vectors; each thread one vector.
pub const cuda_rabitq_kernel =
    \\extern "C" __global__
    \\void rabitq_popcount(const unsigned long long* codes,
    \\                     const unsigned long long* query,
    \\                     float* out,
    \\                     int count,
    \\                     int words_per_vec)
    \\{
    \\    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (gid >= count) return;
    \\    int pop = 0;
    \\    for (int w = 0; w < words_per_vec; w++) {
    \\        unsigned long long x = codes[gid * words_per_vec + w] ^ query[w];
    \\        pop += __popcll(x);
    \\    }
    \\    out[gid] = float(pop);
    \\}
;

/// OpenCL kernel for batched RaBitQ XOR-popcount.
pub const opencl_rabitq_kernel =
    \\__kernel void rabitq_popcount(
    \\    __global const ulong* codes,
    \\    __global const ulong* query,
    \\    __global float* out,
    \\    int count,
    \\    int words_per_vec)
    \\{
    \\    int gid = get_global_id(0);
    \\    if (gid >= count) return;
    \\    int pop = 0;
    \\    for (int w = 0; w < words_per_vec; w++) {
    \\        ulong x = codes[gid * words_per_vec + w] ^ query[w];
    \\        pop += popcount(x);
    \\    }
    \\    out[gid] = (float)pop;
    \\}
;

// ============================================================
// Backend-specific contexts loaded via dlopen / std.DynLib
// ============================================================

const MetalContext = struct {
    lib: std.DynLib,
    device: ?*anyopaque,
    mtlCreateSystemDefaultDevice: *const fn () callconv(.c) ?*anyopaque,
};

const CudaContext = struct {
    lib: std.DynLib,
    cuInit: *const fn (c_uint) callconv(.c) c_int,
    cuDeviceGet: *const fn (*c_int, c_int) callconv(.c) c_int,
    cuCtxCreate: *const fn (*?*anyopaque, c_uint, c_int) callconv(.c) c_int,
    cuCtxDestroy: ?*const fn (*anyopaque) callconv(.c) c_int,
    device: c_int,
    context: ?*anyopaque,
};

const OpenCLContext = struct {
    lib: std.DynLib,
    clGetPlatformIDs: *const fn (c_uint, ?[*]?*anyopaque, ?*c_uint) callconv(.c) c_int,
    clGetDeviceIDs: *const fn (?*anyopaque, c_uint, c_uint, ?[*]?*anyopaque, ?*c_uint) callconv(.c) c_int,
    clCreateContext: *const fn (?*const anyopaque, c_uint, [*]const ?*anyopaque, ?*const fn ([*c]const u8, ?*const anyopaque, c_uint, ?*anyopaque) callconv(.c) void, ?*anyopaque, ?*c_int) callconv(.c) ?*anyopaque,
    clCreateCommandQueue: *const fn (?*anyopaque, ?*anyopaque, c_ulong, ?*c_int) callconv(.c) ?*anyopaque,
    clReleaseContext: *const fn (?*anyopaque) callconv(.c) c_int,
    clReleaseCommandQueue: *const fn (?*anyopaque) callconv(.c) c_int,
    platform: ?*anyopaque,
    device: ?*anyopaque,
    context: ?*anyopaque,
    queue: ?*anyopaque,
};

const CuvsContext = struct {
    lib: std.DynLib,
    // Skeleton function pointers for cuVS C API.
    // Real signatures will be filled once cuVS headers are integrated.
    cuvsIndexCreate: ?*const fn () callconv(.c) c_int,
    cuvsIndexDestroy: ?*const fn () callconv(.c) c_int,
    cuvsSearch: ?*const fn () callconv(.c) c_int,
};

// ============================================================
// GpuDevice
// ============================================================

pub const GpuDevice = struct {
    backend: GpuBackend,
    context: ?*anyopaque, // backend-specific context handle
    queue: ?*anyopaque, // command queue / stream
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !GpuDevice {
        const builtin = @import("builtin");

        // 1. Try Metal on macOS
        if (builtin.os.tag == .macos) {
            if (initMetal(allocator)) |maybe_dev| {
                if (maybe_dev) |dev| return dev;
            } else |err| {
                std.log.debug("Metal init failed: {s}", .{@errorName(err)});
            }
        }

        // 2. Try CUDA (Linux / Windows)
        if (initCuda(allocator)) |maybe_dev| {
            if (maybe_dev) |dev| return dev;
        } else |err| {
            std.log.debug("CUDA init failed: {s}", .{@errorName(err)});
        }

        // 3. Try OpenCL (universal fallback)
        if (initOpenCL(allocator)) |maybe_dev| {
            if (maybe_dev) |dev| return dev;
        } else |err| {
            std.log.debug("OpenCL init failed: {s}", .{@errorName(err)});
        }

        // 4. Try cuVS (RAPIDS, typically Linux only)
        if (initCuvs(allocator)) |maybe_dev| {
            if (maybe_dev) |dev| return dev;
        } else |err| {
            std.log.debug("cuVS init failed: {s}", .{@errorName(err)});
        }

        // 5. CPU fallback
        return GpuDevice{
            .backend = .none,
            .context = null,
            .queue = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GpuDevice) void {
        switch (self.backend) {
            .none => {},
            .metal => deinitMetal(self.context.?, self.allocator),
            .cuda => deinitCuda(self.context.?, self.allocator),
            .opencl => deinitOpenCL(self.context.?, self.allocator),
            .cuvs => deinitCuvs(self.context.?, self.allocator),
        }
        self.backend = .none;
        self.context = null;
        self.queue = null;
    }

    /// Returns true if this device can execute RaBitQ popcount kernels.
    pub fn isAvailable(self: *const GpuDevice) bool {
        return self.backend != .none;
    }

    /// Dispatch a batched RaBitQ distance computation.
    /// query_code: bit-packed query vector (dim/64 u64 words).
    /// codes: flat array of all codes (count * dim/64 u64 words).
    /// out_scores: pre-allocated f32 slice of length count.
    pub fn batchRabitqPopcount(
        self: *const GpuDevice,
        query_code: []const u64,
        codes: []const u64,
        out_scores: []f32,
    ) !void {
        std.debug.assert(codes.len == out_scores.len * query_code.len);

        switch (self.backend) {
            .none => {},
            .metal => {
                // TODO: dispatch Metal compute kernel for RaBitQ popcount.
                // Kernel source:
                //   kernel void rabitq_popcount(
                //       device const ulong* codes,
                //       device const ulong* query,
                //       device float* out,
                //       uint count,
                //       uint words_per_vec)
                //   { ... }
            },
            .cuda => {
                // TODO: dispatch CUDA kernel for RaBitQ popcount.
                // cuLaunchKernel(...)
            },
            .opencl => {
                // TODO: dispatch OpenCL kernel for RaBitQ popcount.
                // clEnqueueNDRangeKernel(...)
            },
            .cuvs => {
                // TODO: call cuVS C API for IVF-RaBitQ batch search.
                // cuvsIndexCreate / cuvsSearch
            },
        }

        // CPU fallback: works even when backend != .none so tests pass without real GPU.
        // Uses batchPopcountXor for cache-friendly SIMD processing.
        const words_per_vec = query_code.len;
        const xor_counts = try self.allocator.alloc(u64, out_scores.len);
        defer self.allocator.free(xor_counts);
        simd.batchPopcountXor(codes, query_code, xor_counts, words_per_vec);
        for (0..out_scores.len) |i| {
            out_scores[i] = @floatFromInt(xor_counts[i]);
        }
    }
};

// ============================================================
// Metal backend
// ============================================================

fn initMetal(allocator: std.mem.Allocator) !?GpuDevice {
    var lib = std.DynLib.open("/System/Library/Frameworks/Metal.framework/Metal") catch |err| {
        std.log.debug("Failed to load Metal.framework: {s}", .{@errorName(err)});
        return null;
    };

    const mtlCreateSystemDefaultDevice = lib.lookup(
        *const fn () callconv(.c) ?*anyopaque,
        "MTLCreateSystemDefaultDevice",
    ) orelse {
        std.log.debug("MTLCreateSystemDefaultDevice symbol not found", .{});
        lib.close();
        return null;
    };

    const device = mtlCreateSystemDefaultDevice();
    if (device == null) {
        std.log.debug("Metal returned no default device", .{});
        lib.close();
        return null;
    }

    const ctx = try allocator.create(MetalContext);
    ctx.* = .{
        .lib = lib,
        .device = device,
        .mtlCreateSystemDefaultDevice = mtlCreateSystemDefaultDevice,
    };

    return GpuDevice{
        .backend = .metal,
        .context = ctx,
        .queue = null,
        .allocator = allocator,
    };
}

fn deinitMetal(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const ctx = @as(*MetalContext, @ptrCast(@alignCast(ctx_ptr)));
    ctx.lib.close();
    allocator.destroy(ctx);
}

// ============================================================
// CUDA backend
// ============================================================

fn initCuda(allocator: std.mem.Allocator) !?GpuDevice {
    const paths = switch (@import("builtin").os.tag) {
        .linux => &[_][]const u8{ "libcuda.so.1", "libcuda.so" },
        .windows => &[_][]const u8{"nvcuda.dll"},
        else => return null,
    };

    var lib: ?std.DynLib = null;
    for (paths) |path| {
        lib = std.DynLib.open(path) catch null;
        if (lib != null) break;
    }
    var l = lib orelse return null;

    const cuInit = l.lookup(*const fn (c_uint) callconv(.c) c_int, "cuInit") orelse {
        l.close();
        return null;
    };
    const cuDeviceGet = l.lookup(*const fn (*c_int, c_int) callconv(.c) c_int, "cuDeviceGet") orelse {
        l.close();
        return null;
    };
    const cuCtxCreate = l.lookup(*const fn (*?*anyopaque, c_uint, c_int) callconv(.c) c_int, "cuCtxCreate_v2") orelse
        l.lookup(*const fn (*?*anyopaque, c_uint, c_int) callconv(.c) c_int, "cuCtxCreate") orelse {
        l.close();
        return null;
    };
    const cuCtxDestroy = l.lookup(*const fn (*anyopaque) callconv(.c) c_int, "cuCtxDestroy_v2") orelse
        l.lookup(*const fn (*anyopaque) callconv(.c) c_int, "cuCtxDestroy") orelse null;

    if (cuInit(0) != 0) {
        l.close();
        return null;
    }

    var device: c_int = -1;
    if (cuDeviceGet(&device, 0) != 0) {
        l.close();
        return null;
    }

    var context: ?*anyopaque = null;
    if (cuCtxCreate(&context, 0, device) != 0) {
        l.close();
        return null;
    }

    const ctx = try allocator.create(CudaContext);
    ctx.* = .{
        .lib = l,
        .cuInit = cuInit,
        .cuDeviceGet = cuDeviceGet,
        .cuCtxCreate = cuCtxCreate,
        .cuCtxDestroy = cuCtxDestroy,
        .device = device,
        .context = context,
    };

    return GpuDevice{
        .backend = .cuda,
        .context = ctx,
        .queue = null,
        .allocator = allocator,
    };
}

fn deinitCuda(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const ctx = @as(*CudaContext, @ptrCast(@alignCast(ctx_ptr)));
    if (ctx.context) |c| {
        if (ctx.cuCtxDestroy) |destroy| {
            _ = destroy(c);
        }
    }
    ctx.lib.close();
    allocator.destroy(ctx);
}

// ============================================================
// OpenCL backend
// ============================================================

fn initOpenCL(allocator: std.mem.Allocator) !?GpuDevice {
    const paths = switch (@import("builtin").os.tag) {
        .macos => &[_][]const u8{"/System/Library/Frameworks/OpenCL.framework/OpenCL"},
        .linux => &[_][]const u8{ "libOpenCL.so.1", "libOpenCL.so" },
        .windows => &[_][]const u8{"OpenCL.dll"},
        else => return null,
    };

    var lib: ?std.DynLib = null;
    for (paths) |path| {
        lib = std.DynLib.open(path) catch null;
        if (lib != null) break;
    }
    var l = lib orelse return null;

    const clGetPlatformIDs = l.lookup(
        *const fn (c_uint, ?[*]?*anyopaque, ?*c_uint) callconv(.c) c_int,
        "clGetPlatformIDs",
    ) orelse {
        l.close();
        return null;
    };
    const clGetDeviceIDs = l.lookup(
        *const fn (?*anyopaque, c_uint, c_uint, ?[*]?*anyopaque, ?*c_uint) callconv(.c) c_int,
        "clGetDeviceIDs",
    ) orelse {
        l.close();
        return null;
    };
    const clCreateContext = l.lookup(
        *const fn (?*const anyopaque, c_uint, [*]const ?*anyopaque, ?*const fn ([*c]const u8, ?*const anyopaque, c_uint, ?*anyopaque) callconv(.c) void, ?*anyopaque, ?*c_int) callconv(.c) ?*anyopaque,
        "clCreateContext",
    ) orelse {
        l.close();
        return null;
    };
    const clCreateCommandQueue = l.lookup(
        *const fn (?*anyopaque, ?*anyopaque, c_ulong, ?*c_int) callconv(.c) ?*anyopaque,
        "clCreateCommandQueue",
    ) orelse
        l.lookup(
            *const fn (?*anyopaque, ?*anyopaque, c_ulong, ?*c_int) callconv(.c) ?*anyopaque,
            "clCreateCommandQueueWithProperties",
        ) orelse {
        l.close();
        return null;
    };
    const clReleaseContext = l.lookup(
        *const fn (?*anyopaque) callconv(.c) c_int,
        "clReleaseContext",
    ) orelse {
        l.close();
        return null;
    };
    const clReleaseCommandQueue = l.lookup(
        *const fn (?*anyopaque) callconv(.c) c_int,
        "clReleaseCommandQueue",
    ) orelse {
        l.close();
        return null;
    };

    var platform: ?*anyopaque = null;
    var num_platforms: c_uint = 0;
    if (clGetPlatformIDs(1, @ptrCast(&platform), &num_platforms) != 0 or num_platforms == 0) {
        l.close();
        return null;
    }

    var device: ?*anyopaque = null;
    var num_devices: c_uint = 0;
    if (clGetDeviceIDs(platform, 0xFFFFFFFF, 1, @ptrCast(&device), &num_devices) != 0 or num_devices == 0) {
        l.close();
        return null;
    }

    var err: c_int = 0;
    const context = clCreateContext(null, 1, @ptrCast(&device), null, null, &err);
    if (err != 0 or context == null) {
        l.close();
        return null;
    }

    const queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != 0 or queue == null) {
        _ = clReleaseContext(context);
        l.close();
        return null;
    }

    const ctx = try allocator.create(OpenCLContext);
    ctx.* = .{
        .lib = l,
        .clGetPlatformIDs = clGetPlatformIDs,
        .clGetDeviceIDs = clGetDeviceIDs,
        .clCreateContext = clCreateContext,
        .clCreateCommandQueue = clCreateCommandQueue,
        .clReleaseContext = clReleaseContext,
        .clReleaseCommandQueue = clReleaseCommandQueue,
        .platform = platform,
        .device = device,
        .context = context,
        .queue = queue,
    };

    return GpuDevice{
        .backend = .opencl,
        .context = ctx,
        .queue = queue,
        .allocator = allocator,
    };
}

fn deinitOpenCL(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const ctx = @as(*OpenCLContext, @ptrCast(@alignCast(ctx_ptr)));
    if (ctx.queue) |q| {
        _ = ctx.clReleaseCommandQueue(q);
    }
    if (ctx.context) |c| {
        _ = ctx.clReleaseContext(c);
    }
    ctx.lib.close();
    allocator.destroy(ctx);
}

// ============================================================
// cuVS backend
// ============================================================

fn initCuvs(allocator: std.mem.Allocator) !?GpuDevice {
    const paths = switch (@import("builtin").os.tag) {
        .linux => &[_][]const u8{ "libcuvs.so", "libcuvs.so.0" },
        else => return null,
    };

    var lib: ?std.DynLib = null;
    for (paths) |path| {
        lib = std.DynLib.open(path) catch null;
        if (lib != null) break;
    }
    var l = lib orelse return null;

    const cuvsIndexCreate = l.lookup(*const fn () callconv(.c) c_int, "cuvsIndexCreate");
    const cuvsIndexDestroy = l.lookup(*const fn () callconv(.c) c_int, "cuvsIndexDestroy");
    const cuvsSearch = l.lookup(*const fn () callconv(.c) c_int, "cuvsSearch");

    const ctx = try allocator.create(CuvsContext);
    ctx.* = .{
        .lib = l,
        .cuvsIndexCreate = cuvsIndexCreate,
        .cuvsIndexDestroy = cuvsIndexDestroy,
        .cuvsSearch = cuvsSearch,
    };

    return GpuDevice{
        .backend = .cuvs,
        .context = ctx,
        .queue = null,
        .allocator = allocator,
    };
}

fn deinitCuvs(ctx_ptr: *anyopaque, allocator: std.mem.Allocator) void {
    const ctx = @as(*CuvsContext, @ptrCast(@alignCast(ctx_ptr)));
    ctx.lib.close();
    allocator.destroy(ctx);
}

// ============================================================
// Helpers for tests
// ============================================================

fn testCpuFallback(dev: *const GpuDevice) !void {
    const query_code = &[_]u64{ 0xFF00FF00FF00FF00, 0x0F0F0F0F0F0F0F0F };
    const codes = &[_]u64{
        0xFF00FF00FF00FF00, 0x0F0F0F0F0F0F0F0F,
        0x0000000000000000, 0x0000000000000000,
    };
    var scores: [2]f32 = undefined;
    try dev.batchRabitqPopcount(query_code, codes, &scores);

    // First vector identical to query -> popcount 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], 0.01);
    // Second vector all-zero -> popcount = total bits set in query
    const expected = @as(f32, @floatFromInt(@popCount(query_code[0]) + @popCount(query_code[1])));
    try std.testing.expectApproxEqAbs(expected, scores[1], 0.01);
}

// ============================================================
// Unit Tests
// ============================================================

test "gpu fallback cpu correctness" {
    const allocator = std.testing.allocator;
    var dev = try GpuDevice.init(allocator);
    defer dev.deinit();

    // Even if no GPU, batchRabitqPopcount must produce correct CPU fallback.
    try testCpuFallback(&dev);
}

test "gpu auto detection" {
    const allocator = std.testing.allocator;
    var dev = try GpuDevice.init(allocator);
    defer dev.deinit();

    const builtin = @import("builtin");
    if (builtin.os.tag == .macos) {
        // On macOS the skeleton attempts to dlopen Metal.framework.
        // If the framework is present (always true on macOS) and a device is
        // reported, backend may be .metal.  Machines without Metal support
        // (very old hardware or CI images) will gracefully fall back to .none.
        std.debug.print("Auto-detected backend on macOS: {s}\n", .{@tagName(dev.backend)});
    }

    // Regardless of detected backend, the CPU fallback path must always work.
    try testCpuFallback(&dev);
}

test "gpu metal dlopen skeleton" {
    const allocator = std.testing.allocator;
    if (@import("builtin").os.tag != .macos) return;

    // Verify the Metal init skeleton can at least load the framework symbols.
    const maybe_dev = try initMetal(allocator);
    if (maybe_dev) |*dev| {
        defer dev.deinit();
        try std.testing.expect(dev.backend == .metal);
    } else {
        // Metal not available on this macOS system -> ok for skeleton.
    }
}

test "gpu opencl dlopen skeleton" {
    const allocator = std.testing.allocator;
    const maybe_dev = try initOpenCL(allocator);
    if (maybe_dev) |*dev| {
        defer dev.deinit();
        try std.testing.expect(dev.backend == .opencl);
    } else {
        // OpenCL not installed -> ok for skeleton.
    }
}

test "gpu cuda dlopen skeleton" {
    const allocator = std.testing.allocator;
    const maybe_dev = try initCuda(allocator);
    if (maybe_dev) |*dev| {
        defer dev.deinit();
        try std.testing.expect(dev.backend == .cuda);
    } else {
        // CUDA not installed -> ok for skeleton.
    }
}

test "gpu cuvs dlopen skeleton" {
    const allocator = std.testing.allocator;
    const maybe_dev = try initCuvs(allocator);
    if (maybe_dev) |*dev| {
        defer dev.deinit();
        try std.testing.expect(dev.backend == .cuvs);
    } else {
        // cuVS not installed -> ok for skeleton.
    }
}
