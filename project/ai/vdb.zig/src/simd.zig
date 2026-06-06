const std = @import("std");

/// Platform SIMD abstraction for RaBitQ popcount and rotation matrix ops.
/// Keeps CPU backend pure Zig with @Vector intrinsics.
pub const SimdBackend = enum {
    scalar,
    x86_avx512,
    x86_avx2,
    aarch64_neon,
};

pub inline fn detectBackend() SimdBackend {
    const cpu = @import("builtin").cpu;
    if (cpu.arch.isX86()) {
        if (std.Target.x86.featureSetHas(cpu.features, .avx512f)) {
            return .x86_avx512;
        }
        if (std.Target.x86.featureSetHas(cpu.features, .avx2)) {
            return .x86_avx2;
        }
    }
    if (cpu.arch == .aarch64) {
        if (std.Target.aarch64.featureSetHas(cpu.features, .neon)) {
            return .aarch64_neon;
        }
    }
    return .scalar;
}

/// Popcount over a slice of u64 words.
pub fn popcountWords(words: []const u64) u64 {
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => popcountVec(words, 8),
        .x86_avx2 => popcountVec(words, 4),
        .aarch64_neon => popcountVec(words, 2),
        .scalar => popcountScalar(words),
    };
}

/// Popcount of XOR between two equal-length u64 slices.
/// RaBitQ distance = popcount(xor(a, b)).
pub fn popcountXorWords(a: []const u64, b: []const u64) u64 {
    std.debug.assert(a.len == b.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => popcountXorVec(a, b, 8),
        .x86_avx2 => popcountXorVec(a, b, 4),
        .aarch64_neon => popcountXorVec(a, b, 2),
        .scalar => popcountXorScalar(a, b),
    };
}

fn popcountScalar(words: []const u64) u64 {
    var sum: u64 = 0;
    for (words) |w| {
        sum += @popCount(w);
    }
    return sum;
}

fn popcountXorScalar(a: []const u64, b: []const u64) u64 {
    var sum: u64 = 0;
    for (a, b) |ai, bi| {
        sum += @popCount(ai ^ bi);
    }
    return sum;
}

/// Generic vectorized popcount using @Vector and @reduce.
/// LANES must be a power of two (2, 4, 8, 16).
fn popcountVec(words: []const u64, comptime lanes: usize) u64 {
    const Vec = @Vector(lanes, u64);
    var sum: u64 = 0;
    var i: usize = 0;
    const simd_end = words.len - (words.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const v: Vec = words[i..][0..lanes].*;
        const counts: Vec = @popCount(v);
        sum += @reduce(.Add, counts);
    }
    for (i..words.len) |j| {
        sum += @popCount(words[j]);
    }
    return sum;
}

fn popcountXorVec(a: []const u64, b: []const u64, comptime lanes: usize) u64 {
    const Vec = @Vector(lanes, u64);
    var sum: u64 = 0;
    var i: usize = 0;
    const simd_end = a.len - (a.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const va: Vec = a[i..][0..lanes].*;
        const vb: Vec = b[i..][0..lanes].*;
        const xored = va ^ vb;
        const counts: Vec = @popCount(xored);
        sum += @reduce(.Add, counts);
    }
    for (i..a.len) |j| {
        sum += @popCount(a[j] ^ b[j]);
    }
    return sum;
}

/// Vector dot product used in rotation matrix application and refine layer.
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => dotVec(a, b, 16), // 512-bit = 16 x f32
        .x86_avx2 => dotVec(a, b, 8), // 256-bit = 8 x f32
        .aarch64_neon => dotVec(a, b, 4), // 128-bit = 4 x f32
        .scalar => dotScalar(a, b),
    };
}

fn dotScalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |ai, bi| {
        sum += ai * bi;
    }
    return sum;
}

fn dotVec(a: []const f32, b: []const f32, comptime lanes: usize) f32 {
    const Vec = @Vector(lanes, f32);
    var sum: Vec = @splat(0.0);
    var i: usize = 0;
    const simd_end = a.len - (a.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const va: Vec = a[i..][0..lanes].*;
        const vb: Vec = b[i..][0..lanes].*;
        sum += va * vb;
    }
    var total: f32 = @reduce(.Add, sum);
    for (i..a.len) |j| {
        total += a[j] * b[j];
    }
    return total;
}

/// L2 distance squared: sum((a[i] - b[i])^2).
/// Used for fast nearest-centroid assignment in IVF index.
pub fn l2DistanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => l2Vec(a, b, 16),
        .x86_avx2 => l2Vec(a, b, 8),
        .aarch64_neon => l2Vec(a, b, 4),
        .scalar => l2Scalar(a, b),
    };
}

/// Batch XOR-popcount: compute popcount(xor(codes[i], query)) for multiple code vectors.
/// This is the core inner loop of RaBitQ search; batch processing improves cache locality.
/// codes: flat array of N code vectors, each words_per_vec u64 words long.
/// query: single query code vector, words_per_vec u64 words long.
/// out: pre-allocated u64 slice of length n.
pub fn batchPopcountXor(codes: []const u64, query: []const u64, out: []u64, words_per_vec: usize) void {
    std.debug.assert(codes.len == out.len * words_per_vec);
    std.debug.assert(query.len == words_per_vec);
    const backend = detectBackend();
    switch (backend) {
        .x86_avx512 => {
            for (0..out.len) |i| {
                out[i] = popcountXorVec(codes[i * words_per_vec .. (i + 1) * words_per_vec], query, 8);
            }
        },
        .x86_avx2 => {
            for (0..out.len) |i| {
                out[i] = popcountXorVec(codes[i * words_per_vec .. (i + 1) * words_per_vec], query, 4);
            }
        },
        .aarch64_neon => {
            for (0..out.len) |i| {
                out[i] = popcountXorVec(codes[i * words_per_vec .. (i + 1) * words_per_vec], query, 2);
            }
        },
        .scalar => {
            for (0..out.len) |i| {
                out[i] = popcountXorScalar(codes[i * words_per_vec .. (i + 1) * words_per_vec], query);
            }
        },
    }
}

/// Batch dot product: compute dot(row[i], vec) for multiple rows.
/// Used for batch rotation matrix application.
/// rows: flat array of N rows, each row_len f32 elements long.
/// vec: single vector, row_len f32 elements long.
/// out: pre-allocated f32 slice of length n.
pub fn batchDotProduct(rows: []const f32, vec: []const f32, out: []f32, row_len: usize) void {
    std.debug.assert(rows.len == out.len * row_len);
    std.debug.assert(vec.len == row_len);
    for (0..out.len) |i| {
        out[i] = dotProduct(rows[i * row_len ..][0..row_len], vec);
    }
}

/// SQ8 L2 distance: dequantize SQ8 codes and compute L2 distance to query.
/// Dequantization: deq[d] = centroid[d] + (code[d] / 255.0) * 2.0 - 1.0
/// Then: L2^2 = sum((query[d] - deq[d])^2)
/// Uses SIMD vectorized processing for 2-4x speedup over scalar loop.
pub fn sq8L2Distance(query: []const f32, centroid: []const f32, sq8_codes: []const u8) f32 {
    std.debug.assert(query.len == centroid.len);
    std.debug.assert(sq8_codes.len == centroid.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => sq8L2Vec(query, centroid, sq8_codes, 16),
        .x86_avx2 => sq8L2Vec(query, centroid, sq8_codes, 8),
        .aarch64_neon => sq8L2Vec(query, centroid, sq8_codes, 4),
        .scalar => sq8L2Scalar(query, centroid, sq8_codes),
    };
}

fn sq8L2Scalar(query: []const f32, centroid: []const f32, sq8_codes: []const u8) f32 {
    var sum: f32 = 0.0;
    for (query, centroid, sq8_codes) |q, c, code| {
        const deq = c + (@as(f32, @floatFromInt(code)) / 255.0) * 2.0 - 1.0;
        const diff = q - deq;
        sum += diff * diff;
    }
    return sum;
}

fn sq8L2Vec(query: []const f32, centroid: []const f32, sq8_codes: []const u8, comptime lanes: usize) f32 {
    const VecF = @Vector(lanes, f32);
    const VecU = @Vector(lanes, u8);
    var sum: VecF = @splat(0.0);
    var i: usize = 0;
    const simd_end = query.len - (query.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const q: VecF = query[i..][0..lanes].*;
        const c: VecF = centroid[i..][0..lanes].*;
        const codes: VecU = sq8_codes[i..][0..lanes].*;
        // Dequantize: centroid + (code / 255.0) * 2.0 - 1.0
        const codes_f: VecF = @floatFromInt(codes);
        const deq = c + (codes_f / @as(VecF, @splat(255.0))) * @as(VecF, @splat(2.0)) - @as(VecF, @splat(1.0));
        const diff = q - deq;
        sum += diff * diff;
    }
    var total: f32 = @reduce(.Add, sum);
    // Handle remainder
    for (i..query.len) |j| {
        const deq = centroid[j] + (@as(f32, @floatFromInt(sq8_codes[j])) / 255.0) * 2.0 - 1.0;
        const diff = query[j] - deq;
        total += diff * diff;
    }
    return total;
}

/// Dynamic-range SQ8 L2 distance.
/// Dequantization: deq[d] = centroid[d] + sq8_min[d] + (code[d] / sq8_scale[d])
/// scale[d] is always > 0 (initialized to 1.0 as dummy when range == 0).
/// Uses SIMD vectorized processing for 2-4x speedup over scalar loop.
pub fn sq8L2DistanceDynamic(query: []const f32, centroid: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_scale: []const f32) f32 {
    std.debug.assert(query.len == centroid.len);
    std.debug.assert(sq8_codes.len == centroid.len);
    std.debug.assert(sq8_min.len == centroid.len);
    std.debug.assert(sq8_scale.len == centroid.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => sq8L2DynamicVec(query, centroid, sq8_codes, sq8_min, sq8_scale, 16),
        .x86_avx2 => sq8L2DynamicVec(query, centroid, sq8_codes, sq8_min, sq8_scale, 8),
        .aarch64_neon => sq8L2DynamicVec(query, centroid, sq8_codes, sq8_min, sq8_scale, 4),
        .scalar => sq8L2DynamicScalar(query, centroid, sq8_codes, sq8_min, sq8_scale),
    };
}

fn sq8L2DynamicScalar(query: []const f32, centroid: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_scale: []const f32) f32 {
    var sum: f32 = 0.0;
    for (query, centroid, sq8_codes, sq8_min, sq8_scale) |q, c, code, min, scale| {
        const deq = c + min + @as(f32, @floatFromInt(code)) / scale;
        const diff = q - deq;
        sum += diff * diff;
    }
    return sum;
}

fn sq8L2DynamicVec(query: []const f32, centroid: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_scale: []const f32, comptime lanes: usize) f32 {
    const VecF = @Vector(lanes, f32);
    const VecU = @Vector(lanes, u8);
    var sum: VecF = @splat(0.0);
    var i: usize = 0;
    const simd_end = query.len - (query.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const q: VecF = query[i..][0..lanes].*;
        const c: VecF = centroid[i..][0..lanes].*;
        const codes: VecU = sq8_codes[i..][0..lanes].*;
        const min: VecF = sq8_min[i..][0..lanes].*;
        const scale: VecF = sq8_scale[i..][0..lanes].*;

        const codes_f: VecF = @floatFromInt(codes);
        const deq = c + min + codes_f / scale;
        const diff = q - deq;
        sum += diff * diff;
    }
    var total: f32 = @reduce(.Add, sum);
    // Handle remainder
    for (i..query.len) |j| {
        const deq = centroid[j] + sq8_min[j] + @as(f32, @floatFromInt(sq8_codes[j])) / sq8_scale[j];
        const diff = query[j] - deq;
        total += diff * diff;
    }
    return total;
}

fn l2Scalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |ai, bi| {
        const diff = ai - bi;
        sum += diff * diff;
    }
    return sum;
}

fn l2Vec(a: []const f32, b: []const f32, comptime lanes: usize) f32 {
    const Vec = @Vector(lanes, f32);
    var sum: Vec = @splat(0.0);
    var i: usize = 0;
    const simd_end = a.len - (a.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const va: Vec = a[i..][0..lanes].*;
        const vb: Vec = b[i..][0..lanes].*;
        const diff = va - vb;
        sum += diff * diff;
    }
    var total: f32 = @reduce(.Add, sum);
    for (i..a.len) |j| {
        const diff = a[j] - b[j];
        total += diff * diff;
    }
    return total;
}

// ============================================
// Unit Tests
// ============================================

test "popcount correctness" {
    const words = &[_]u64{ 0, 0xFF, 0xFFFF_FFFF_FFFF_FFFF };
    try std.testing.expectEqual(@as(u64, 0), popcountWords(words[0..1]));
    try std.testing.expectEqual(@as(u64, 8), popcountWords(words[1..2]));
    try std.testing.expectEqual(@as(u64, 64), popcountWords(words[2..3]));
}

test "popcount xor correctness" {
    const a = &[_]u64{ 0x0F0F, 0xFFFF };
    const b = &[_]u64{ 0x00FF, 0x0F0F };
    // 0x0F0F ^ 0x00FF = 0x0FF0 => popcount 8
    // 0xFFFF ^ 0x0F0F = 0xF0F0 => popcount 8
    try std.testing.expectEqual(@as(u64, 16), popcountXorWords(a, b));
}

test "dot product correctness" {
    const a = &[_]f32{ 1.0, 2.0, 3.0 };
    const b = &[_]f32{ 4.0, 5.0, 6.0 };
    const result = dotProduct(a, b);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 0.001);
}
