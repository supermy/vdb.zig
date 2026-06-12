const std = @import("std");
const posix = std.posix;
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");

const DIM: u32 = 128;

fn writeStdout(msg: []const u8) void {
    const file = std.Io.File.stdout();
    const io = std.Io.Threaded.global_single_threaded.io();
    file.writeStreamingAll(io, msg) catch |err| std.log.err("stdout write failed: {}", .{err});
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var buf: [2048]u8 = undefined;

    writeStdout("=== Step-by-step RaBitQ Distance Diagnosis ===\n\n");

    var rng = std.Random.DefaultPrng.init(42);

    // Generate centroid, one data vector, one query
    var centroid: [128]f32 = undefined;
    var data_vec: [128]f32 = undefined;
    var query_vec: [128]f32 = undefined;
    for (&centroid, &data_vec, &query_vec) |*c, *d, *q| {
        c.* = rng.random().floatNorm(f32) * 2.0;
        d.* = rng.random().floatNorm(f32) * 3.0;
        q.* = rng.random().floatNorm(f32) * 3.0;
    }

    // Build minimal index with 1 partition
    var idx = try index_mod.Index.init(allocator, DIM, .{
        .num_partitions = 1,
        .refine_sq8 = false,
        .fastscan = false,
        .rotation_seed = 12345,
    });
    defer idx.deinit();

    // Train with some vectors to set centroid
    const train_data = try allocator.alloc(f32, 100 * DIM);
    defer allocator.free(train_data);
    for (train_data) |*v| v.* = rng.random().floatNorm(f32) * 2.0;
    const train_slices = try allocator.alloc([]const f32, 100);
    defer allocator.free(train_slices);
    for (0..100) |i| train_slices[i] = train_data[i * DIM ..][0..DIM];
    try idx.train(train_slices);

    // Override centroid with our specific centroid for controlled test
    @memcpy(idx.partitions[0].centroid, &centroid);

    // Insert data vector
    try idx.batchInsert(&.{&data_vec});

    // ====== Manual step-by-step computation ======
    writeStdout("--- Step-by-step RaBitQ distance estimation ---\n\n");

    // Step 1: Compute residual and its norm
    var residual: [128]f32 = undefined;
    var residual_norm: f32 = 0.0;
    for (0..DIM) |i| {
        residual[i] = data_vec[i] - centroid[i];
        residual_norm += residual[i] * residual[i];
    }
    residual_norm = @sqrt(residual_norm);
    const line1 = try std.fmt.bufPrint(&buf, "1. residual_norm = ||x-c|| = {d:.6}\n", .{residual_norm});
    writeStdout(line1);

    // Step 2: Query residual
    var q_residual: [128]f32 = undefined;
    var q_residual_norm_sq: f32 = 0.0;
    for (0..DIM) |i| {
        q_residual[i] = query_vec[i] - centroid[i];
        q_residual_norm_sq += q_residual[i] * q_residual[i];
    }
    const q_residual_norm = @sqrt(q_residual_norm_sq);
    const line2 = try std.fmt.bufPrint(&buf, "2. ||q-c|| = {d:.6}, ||q-c||² = {d:.6}\n", .{ q_residual_norm, q_residual_norm_sq });
    writeStdout(line2);

    // Step 3: Exact L2 distance
    var exact_dist: f32 = 0.0;
    for (0..DIM) |i| {
        const diff = data_vec[i] - query_vec[i];
        exact_dist += diff * diff;
    }
    const line3 = try std.fmt.bufPrint(&buf, "3. Exact ||x-q||² = {d:.6}\n", .{exact_dist});
    writeStdout(line3);

    // Verify: ||x-c||² + ||q-c||² - 2*<x-c, q-c>
    var exact_cross: f32 = 0.0;
    for (0..DIM) |i| {
        exact_cross += residual[i] * q_residual[i];
    }
    const verify = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * exact_cross;
    const line4 = try std.fmt.bufPrint(&buf, "4. Verify: ||x-c||²+||q-c||²-2*<x-c,q-c> = {d:.6} (should match #3)\n", .{verify});
    writeStdout(line4);
    const line5 = try std.fmt.bufPrint(&buf, "   <x-c, q-c> = {d:.6}\n", .{exact_cross});
    writeStdout(line5);

    // Step 4: Apply rotation to normalized residual
    const rotation = idx.rotation;
    var x_rot: [128]f32 = undefined;
    const inv_residual_norm = if (residual_norm > 1e-8) 1.0 / residual_norm else 0.0;
    for (0..DIM) |i| {
        const row = rotation[i * DIM ..][0..DIM];
        var sum: f32 = 0.0;
        for (0..DIM) |j| {
            sum += row[j] * residual[j] * inv_residual_norm;
        }
        x_rot[i] = sum;
    }

    // Verify x_rot is unit length
    var x_rot_norm: f32 = 0.0;
    for (&x_rot) |v| x_rot_norm += v * v;
    x_rot_norm = @sqrt(x_rot_norm);
    const line6 = try std.fmt.bufPrint(&buf, "5. ||x_rot|| = {d:.6} (should be 1.0)\n", .{x_rot_norm});
    writeStdout(line6);

    // Step 5: Compute binary code and correction
    var dot_o_bar_o: f32 = 0.0;
    var code: [128]bool = undefined;
    for (0..DIM) |i| {
        const is_positive = x_rot[i] >= 0.0;
        code[i] = is_positive;
        if (is_positive) {
            dot_o_bar_o += x_rot[i];
        } else {
            dot_o_bar_o -= x_rot[i];
        }
    }
    const line7 = try std.fmt.bufPrint(&buf, "6. <ō,o> = sum(|x_rot_i|) = {d:.6} (expected ≈ 0.798*sqrt({d}) = {d:.6})\n", .{ dot_o_bar_o, DIM, 0.798 * @sqrt(@as(f64, @floatFromInt(DIM))) });
    writeStdout(line7);

    // Step 6: Apply rotation to query residual
    var q_r_rot: [128]f32 = undefined;
    for (0..DIM) |i| {
        const row = rotation[i * DIM ..][0..DIM];
        var sum: f32 = 0.0;
        for (0..DIM) |j| {
            sum += row[j] * q_residual[j];
        }
        q_r_rot[i] = sum;
    }

    var q_r_rot_norm: f32 = 0.0;
    for (&q_r_rot) |v| q_r_rot_norm += v * v;
    q_r_rot_norm = @sqrt(q_r_rot_norm);
    const line8 = try std.fmt.bufPrint(&buf, "7. ||R*(q-c)|| = {d:.6} (should equal ||q-c|| = {d:.6})\n", .{ q_r_rot_norm, q_residual_norm });
    writeStdout(line8);

    // Step 7: Compute <sign(code), R*(q-c)>
    var ip_sign_qr_rot: f32 = 0.0;
    for (0..DIM) |i| {
        const sign: f32 = if (code[i]) 1.0 else -1.0;
        ip_sign_qr_rot += sign * q_r_rot[i];
    }
    const line9 = try std.fmt.bufPrint(&buf, "8. <ō, R*(q-c)> = {d:.6}\n", .{ip_sign_qr_rot});
    writeStdout(line9);

    // Step 8: Compute <o, R*(q-c)> exactly
    var ip_exact: f32 = 0.0;
    for (0..DIM) |i| {
        ip_exact += x_rot[i] * q_r_rot[i];
    }
    const line10 = try std.fmt.bufPrint(&buf, "9. <o, R*(q-c)> = {d:.6} (exact)\n", .{ip_exact});
    writeStdout(line10);

    // Step 9: Compare estimators
    const dim_f: f32 = @floatFromInt(DIM);
    const correction = dot_o_bar_o;

    // Multiplication: <o, y> ≈ <ō, y> * <ō, o> / D
    const ip_mult = ip_sign_qr_rot * correction / dim_f;
    const score_mult = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_mult;

    // Ratio: <o, y> ≈ <ō, y> / <ō, o>
    const ip_ratio = ip_sign_qr_rot / correction;
    const score_ratio = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_ratio;

    // Exact cross-term
    const exact_ip_term = residual_norm * ip_exact;
    const score_exact = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * exact_ip_term;

    writeStdout("\n--- Estimator Comparison ---\n");
    const line11 = try std.fmt.bufPrint(&buf,
        "  Exact <o, R*(q-c)>     = {d:.6}\n" ++
        "  Mult  <ō,y>*<ō,o>/D   = {d:.6}  (error: {d:.6})\n" ++
        "  Ratio <ō,y>/<ō,o>     = {d:.6}  (error: {d:.6})\n\n",
        .{
            ip_exact,
            ip_mult, ip_mult - ip_exact,
            ip_ratio, ip_ratio - ip_exact,
        },
    );
    writeStdout(line11);

    const line12 = try std.fmt.bufPrint(&buf,
        "  Exact dist    = {d:.6}\n" ++
        "  Mult  dist    = {d:.6}  (error: {d:.6})\n" ++
        "  Ratio dist    = {d:.6}  (error: {d:.6})\n",
        .{
            score_exact,
            score_mult, score_mult - score_exact,
            score_ratio, score_ratio - score_exact,
        },
    );
    writeStdout(line12);

    // Step 10: Self-query test
    writeStdout("\n--- Self-query test (q = x) ---\n");

    // Rotate data residual for self-query
    var q_r_rot_self: [128]f32 = undefined;
    for (0..DIM) |i| {
        const row = rotation[i * DIM ..][0..DIM];
        var sum: f32 = 0.0;
        for (0..DIM) |j| {
            sum += row[j] * residual[j];
        }
        q_r_rot_self[i] = sum;
    }

    var ip_sign_self: f32 = 0.0;
    var ip_exact_self: f32 = 0.0;
    for (0..DIM) |i| {
        const sign: f32 = if (code[i]) 1.0 else -1.0;
        ip_sign_self += sign * q_r_rot_self[i];
        ip_exact_self += x_rot[i] * q_r_rot_self[i];
    }

    const ip_mult_self = ip_sign_self * correction / dim_f;
    const score_mult_self = residual_norm * residual_norm + residual_norm * residual_norm - 2.0 * residual_norm * ip_mult_self;

    const ip_ratio_self = ip_sign_self / correction;
    const score_ratio_self = residual_norm * residual_norm + residual_norm * residual_norm - 2.0 * residual_norm * ip_ratio_self;

    const line13 = try std.fmt.bufPrint(&buf,
        "  <ō, R*(x-c)> = {d:.6}\n" ++
        "  <o, R*(x-c)> = {d:.6} (should = ||x-c|| = {d:.6})\n" ++
        "  Mult  dist = {d:.6} (should = 0)\n" ++
        "  Ratio dist = {d:.6} (should = 0)\n",
        .{
            ip_sign_self,
            ip_exact_self, residual_norm,
            score_mult_self,
            score_ratio_self,
        },
    );
    writeStdout(line13);

    // Now search using the index and compare
    writeStdout("\n--- Index search result ---\n");
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&query_vec, 10, 1, &results);
    if (found > 0) {
        const line14 = try std.fmt.bufPrint(&buf,
            "  Index search score = {d:.6} (should match one of the estimators above)\n",
            .{results[0].score},
        );
        writeStdout(line14);
    }

    writeStdout("\nDiagnosis complete.\n");
}
