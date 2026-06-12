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

/// SQ8 L2 distance from precomputed query-centroid residual.
/// More efficient than sq8L2DistanceDynamic for batch processing:
/// avoids redundant centroid subtraction per candidate.
/// dist = ||q_residual - (min + code * inv_scale)||^2
pub fn sq8L2DistanceFromResidual(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32) f32 {
    std.debug.assert(q_residual.len == sq8_codes.len);
    std.debug.assert(sq8_min.len == sq8_codes.len);
    std.debug.assert(sq8_inv_scale.len == sq8_codes.len);
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx512 => sq8FromResidualVec(q_residual, sq8_codes, sq8_min, sq8_inv_scale, 16),
        .x86_avx2 => sq8FromResidualVec(q_residual, sq8_codes, sq8_min, sq8_inv_scale, 8),
        .aarch64_neon => sq8FromResidualVec(q_residual, sq8_codes, sq8_min, sq8_inv_scale, 4),
        .scalar => sq8FromResidualScalar(q_residual, sq8_codes, sq8_min, sq8_inv_scale),
    };
}

fn sq8FromResidualScalar(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32) f32 {
    var sum: f32 = 0.0;
    for (q_residual, sq8_codes, sq8_min, sq8_inv_scale) |qr, code, min, inv_scale| {
        const deq_residual = min + @as(f32, @floatFromInt(code)) * inv_scale;
        const diff = qr - deq_residual;
        sum += diff * diff;
    }
    return sum;
}

fn sq8FromResidualVec(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32, comptime lanes: usize) f32 {
    const VecF = @Vector(lanes, f32);
    const VecU = @Vector(lanes, u8);
    var sum: VecF = @splat(0.0);
    var i: usize = 0;
    const simd_end = q_residual.len - (q_residual.len % lanes);
    while (i < simd_end) : (i += lanes) {
        const qr: VecF = q_residual[i..][0..lanes].*;
        const codes: VecU = sq8_codes[i..][0..lanes].*;
        const min: VecF = sq8_min[i..][0..lanes].*;
        const inv_scale: VecF = sq8_inv_scale[i..][0..lanes].*;

        const codes_f: VecF = @floatFromInt(codes);
        const deq_residual = min + codes_f * inv_scale;
        const diff = qr - deq_residual;
        sum += diff * diff;
    }
    var total: f32 = @reduce(.Add, sum);
    // Handle remainder
    for (i..q_residual.len) |j| {
        const deq_residual = sq8_min[j] + @as(f32, @floatFromInt(sq8_codes[j])) * sq8_inv_scale[j];
        const diff = q_residual[j] - deq_residual;
        total += diff * diff;
    }
    return total;
}

/// Batch SQ8 L2 distance from precomputed query-centroid residual.
/// Processes multiple vectors from the same partition in one call.
/// sq8_codes_all is the full SQ8 code array for the partition (count * dim elements).
/// vector_indices specifies which vectors to compute distances for.
/// Results are written to out_distances (must have len >= vector_indices.len).
/// This reduces function call overhead and improves cache locality for sq8_min/inv_scale.
pub fn sq8BatchL2DistanceFromResidual(
    q_residual: []const f32,
    sq8_codes_all: []const u8,
    sq8_min: []const f32,
    sq8_inv_scale: []const f32,
    dim: usize,
    vector_indices: []const u32,
    out_distances: []f32,
) void {
    std.debug.assert(q_residual.len == dim);
    std.debug.assert(sq8_min.len == dim);
    std.debug.assert(sq8_inv_scale.len == dim);
    std.debug.assert(out_distances.len >= vector_indices.len);

    const backend = detectBackend();
    switch (backend) {
        .x86_avx2 => {
            const lanes = 8;
            const VecF = @Vector(lanes, f32);
            const VecU = @Vector(lanes, u8);
            const simd_end = dim - (dim % lanes);

            // Preload sq8_min and inv_scale into cache (they're shared across all vectors)
            var min_vec_arr: [768]VecF = undefined;
            var inv_scale_vec_arr: [768]VecF = undefined;
            var qr_vec_arr: [768]VecF = undefined;
            @memset(min_vec_arr[0 .. dim / lanes], undefined);
            @memset(inv_scale_vec_arr[0 .. dim / lanes], undefined);
            @memset(qr_vec_arr[0 .. dim / lanes], undefined);

            // Pre-vectorize shared data
            var vi: usize = 0;
            while (vi < simd_end) : (vi += lanes) {
                const idx = vi / lanes;
                qr_vec_arr[idx] = q_residual[vi..][0..lanes].*;
                min_vec_arr[idx] = sq8_min[vi..][0..lanes].*;
                inv_scale_vec_arr[idx] = sq8_inv_scale[vi..][0..lanes].*;
            }

            for (vector_indices, 0..) |vec_idx, out_idx| {
                const code_offset = @as(usize, vec_idx) * dim;
                const codes = sq8_codes_all[code_offset..][0..dim];

                var sum: VecF = @splat(0.0);
                var j: usize = 0;
                while (j < simd_end) : (j += lanes) {
                    const idx = j / lanes;
                    const codes_vec: VecU = codes[j..][0..lanes].*;
                    const codes_f: VecF = @floatFromInt(codes_vec);
                    const deq_residual = min_vec_arr[idx] + codes_f * inv_scale_vec_arr[idx];
                    const diff = qr_vec_arr[idx] - deq_residual;
                    sum += diff * diff;
                }
                var total: f32 = @reduce(.Add, sum);
                // Handle remainder
                for (simd_end..dim) |d| {
                    const deq_residual = sq8_min[d] + @as(f32, @floatFromInt(codes[d])) * sq8_inv_scale[d];
                    const diff = q_residual[d] - deq_residual;
                    total += diff * diff;
                }
                out_distances[out_idx] = total;
            }
        },
        else => {
            // Scalar / NEON fallback: call per-vector function
            for (vector_indices, 0..) |vec_idx, out_idx| {
                const code_offset = @as(usize, vec_idx) * dim;
                out_distances[out_idx] = sq8L2DistanceFromResidual(
                    q_residual,
                    sq8_codes_all[code_offset..][0..dim],
                    sq8_min,
                    sq8_inv_scale,
                );
            }
        },
    }
}

/// SQ8 L2 distance from precomputed query-centroid residual with early termination.
/// Computes distance in blocks and returns early if partial sum exceeds threshold.
/// Returns null if early terminated (distance > threshold).
pub fn sq8L2DistanceFromResidualEarlyTerm(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32, threshold: f32) ?f32 {
    const backend = detectBackend();
    return switch (backend) {
        .x86_avx2 => sq8FromResidualEarlyTermVec(q_residual, sq8_codes, sq8_min, sq8_inv_scale, threshold, 8),
        .aarch64_neon => sq8FromResidualEarlyTermVec(q_residual, sq8_codes, sq8_min, sq8_inv_scale, threshold, 4),
        .scalar => sq8FromResidualEarlyTermScalar(q_residual, sq8_codes, sq8_min, sq8_inv_scale, threshold),
        else => sq8FromResidualEarlyTermScalar(q_residual, sq8_codes, sq8_min, sq8_inv_scale, threshold),
    };
}

fn sq8FromResidualEarlyTermScalar(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32, threshold: f32) ?f32 {
    var sum: f32 = 0.0;
    const block_size = 64;
    var i: usize = 0;
    while (i < q_residual.len) : (i += block_size) {
        const end = @min(i + block_size, q_residual.len);
        for (i..end) |j| {
            const deq_residual = sq8_min[j] + @as(f32, @floatFromInt(sq8_codes[j])) * sq8_inv_scale[j];
            const diff = q_residual[j] - deq_residual;
            sum += diff * diff;
        }
        if (sum > threshold) return null;
    }
    return sum;
}

fn sq8FromResidualEarlyTermVec(q_residual: []const f32, sq8_codes: []const u8, sq8_min: []const f32, sq8_inv_scale: []const f32, threshold: f32, comptime lanes: usize) ?f32 {
    const VecF = @Vector(lanes, f32);
    const VecU = @Vector(lanes, u8);
    var sum: f32 = 0.0;
    const block_size = 64; // check threshold every 64 dimensions
    var i: usize = 0;
    const simd_end = q_residual.len - (q_residual.len % lanes);
    var block_count: usize = 0;
    while (i < simd_end) : (i += lanes) {
        const qr: VecF = q_residual[i..][0..lanes].*;
        const codes: VecU = sq8_codes[i..][0..lanes].*;
        const min: VecF = sq8_min[i..][0..lanes].*;
        const inv_scale: VecF = sq8_inv_scale[i..][0..lanes].*;

        const codes_f: VecF = @floatFromInt(codes);
        const deq_residual = min + codes_f * inv_scale;
        const diff = qr - deq_residual;
        sum += @reduce(.Add, diff * diff);

        block_count += lanes;
        if (block_count >= block_size) {
            if (sum > threshold) return null;
            block_count = 0;
        }
    }
    // Handle remainder
    for (i..q_residual.len) |j| {
        const deq_residual = sq8_min[j] + @as(f32, @floatFromInt(sq8_codes[j])) * sq8_inv_scale[j];
        const diff = q_residual[j] - deq_residual;
        sum += diff * diff;
    }
    if (sum > threshold) return null;
    return sum;
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
