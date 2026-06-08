const std = @import("std");
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");

const DIM: u32 = 768;
const N: usize = 1000;

fn writeStdout(msg: []const u8) void {
    const file = std.Io.File.stdout();
    const io = std.Io.Threaded.global_single_threaded.io();
    file.writeStreamingAll(io, msg) catch |err| std.log.err("stdout write failed: {}", .{err});
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var buf: [1024]u8 = undefined;

    writeStdout("=== SQ8 Refinement Verification ===\n\n");

    var rng = std.Random.DefaultPrng.init(42);

    // Generate data
    const data = try allocator.alloc(f32, N * DIM);
    defer allocator.free(data);
    for (data) |*v| v.* = rng.random().floatNorm(f32);

    const learn = try allocator.alloc(f32, N * DIM);
    defer allocator.free(learn);
    for (learn) |*v| v.* = rng.random().floatNorm(f32);

    // Generate query
    var query: [768]f32 = undefined;
    for (&query) |*v| v.* = rng.random().floatNorm(f32);

    // Build index WITH SQ8
    var idx = try index_mod.Index.init(allocator, DIM, .{
        .num_partitions = 4,
        .refine_sq8 = true,
        .refine_k = 100,
        .fastscan = false,
        .rotation_seed = 12345,
    });
    defer idx.deinit();

    const learn_slices = try allocator.alloc([]const f32, N);
    defer allocator.free(learn_slices);
    for (0..N) |i| learn_slices[i] = learn[i * DIM ..][0..DIM];
    try idx.train(learn_slices);

    const data_slices = try allocator.alloc([]const f32, N);
    defer allocator.free(data_slices);
    for (0..N) |i| data_slices[i] = data[i * DIM ..][0..DIM];
    try idx.batchInsert(data_slices);

    // Check SQ8 codes exist
    var total_sq8: u32 = 0;
    for (idx.partitions) |p| {
        if (p.sq8_codes.len > 0) total_sq8 += 1;
    }
    const sq8_info = try std.fmt.bufPrint(&buf, "Partitions with SQ8 codes: {d}/{d}\n", .{ total_sq8, idx.partitions.len });
    writeStdout(sq8_info);

    // Compute exact top-10
    var exact_top: [10]struct { id: u32, dist: f32 } = undefined;
    var exact_dists = try allocator.alloc(f32, N);
    defer allocator.free(exact_dists);
    for (0..N) |i| {
        exact_dists[i] = simd.l2DistanceSquared(&query, data_slices[i]);
    }
    var used = try allocator.alloc(bool, N);
    defer allocator.free(used);
    @memset(used, false);
    for (0..10) |ti| {
        var best_id: u32 = 0;
        var best_dist: f32 = std.math.floatMax(f32);
        for (0..N) |i| {
            if (!used[i] and exact_dists[i] < best_dist) {
                best_dist = exact_dists[i];
                best_id = @intCast(i);
            }
        }
        used[best_id] = true;
        exact_top[ti] = .{ .id = best_id, .dist = best_dist };
    }

    // Search with SQ8 refinement
    var results_sq8: [10]index_mod.SearchResult = undefined;
    const found_sq8 = try idx.search(&query, 10, 4, &results_sq8);

    // Search WITHOUT SQ8 for comparison
    var idx_no_sq8 = try index_mod.Index.init(allocator, DIM, .{
        .num_partitions = 4,
        .refine_sq8 = false,
        .refine_k = 10,
        .fastscan = false,
        .rotation_seed = 12345,
    });
    defer idx_no_sq8.deinit();
    try idx_no_sq8.train(learn_slices);
    try idx_no_sq8.batchInsert(data_slices);

    var results_coarse: [10]index_mod.SearchResult = undefined;
    const found_coarse = try idx_no_sq8.search(&query, 10, 4, &results_coarse);

    // Print comparison
    writeStdout("Rank | Exact ID | Exact Dist | SQ8 ID | SQ8 Score | Coarse ID | Coarse Score\n");
    for (0..10) |i| {
        const sq8_id = if (i < found_sq8) results_sq8[i].id else 0;
        const sq8_score = if (i < found_sq8) results_sq8[i].score else 0.0;
        const coarse_id = if (i < found_coarse) results_coarse[i].id else 0;
        const coarse_score = if (i < found_coarse) results_coarse[i].score else 0.0;
        const line = try std.fmt.bufPrint(&buf,
            "  {d: >2} | {d: >7} | {d: >10.2} | {d: >5} | {d: >9.2} | {d: >8} | {d: >12.2}\n",
            .{ i, exact_top[i].id, exact_top[i].dist, sq8_id, sq8_score, coarse_id, coarse_score },
        );
        writeStdout(line);
    }

    // Compute SQ8 recall vs coarse recall
    const base_offset = idx.next_id.load(.monotonic) - @as(u32, @intCast(N));
    var sq8_recall: f64 = 0.0;
    var coarse_recall: f64 = 0.0;
    for (0..10) |gi| {
        const gt_id = exact_top[gi].id;
        for (0..found_sq8) |ri| {
            if (results_sq8[ri].id == gt_id + base_offset) {
                sq8_recall += 1.0;
                break;
            }
        }
        for (0..found_coarse) |ri| {
            if (results_coarse[ri].id == gt_id + base_offset) {
                coarse_recall += 1.0;
                break;
            }
        }
    }
    sq8_recall /= 10.0;
    coarse_recall /= 10.0;

    const recall_line = try std.fmt.bufPrint(&buf, "\nSQ8 recall@10 = {d:.4}, Coarse recall@10 = {d:.4}\n", .{ sq8_recall, coarse_recall });
    writeStdout(recall_line);

    writeStdout("\nVerification complete.\n");
}
