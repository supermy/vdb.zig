const std = @import("std");
const posix = std.posix;
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");

const DIM: u32 = 128; // Use 128D first for faster diagnosis
const BASE_N: usize = 1_000;
const LEARN_N: usize = 1_000;

fn monoNs() u64 {
    var ts: posix.timespec = undefined;
    const rc = posix.system.clock_gettime(posix.CLOCK.MONOTONIC, &ts);
    std.debug.assert(rc == 0);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn writeStdout(msg: []const u8) void {
    const file = std.Io.File.stdout();
    const io = std.Io.Threaded.global_single_threaded.io();
    file.writeStreamingAll(io, msg) catch |err| std.log.err("stdout write failed: {}", .{err});
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var buf: [1024]u8 = undefined;

    writeStdout("=== RaBitQ Distance Estimation Diagnostic ===\n\n");

    var rng = std.Random.DefaultPrng.init(42);

    // Generate base vectors
    const base_data = try allocator.alloc(f32, BASE_N * DIM);
    defer allocator.free(base_data);
    for (base_data) |*v| v.* = rng.random().floatNorm(f32);

    const learn_data = try allocator.alloc(f32, LEARN_N * DIM);
    defer allocator.free(learn_data);
    for (learn_data) |*v| v.* = rng.random().floatNorm(f32);

    // Generate one query
    var query: [128]f32 = undefined;
    for (&query) |*v| v.* = rng.random().floatNorm(f32);

    // Build index WITHOUT SQ8 refinement to test coarse ranking only
    var idx = try index_mod.Index.init(allocator, DIM, .{
        .num_partitions = 16,
        .refine_sq8 = false,
        .fastscan = false,
        .rotation_seed = 12345,
    });
    defer idx.deinit();

    const learn_slices = try allocator.alloc([]const f32, LEARN_N);
    defer allocator.free(learn_slices);
    for (0..LEARN_N) |i| learn_slices[i] = learn_data[i * DIM ..][0..DIM];

    // Train with learn data, but DON'T add learn vectors to index
    try idx.train(learn_slices);

    const base_id_offset = idx.next_id.load(.monotonic);

    const base_slices = try allocator.alloc([]const f32, BASE_N);
    defer allocator.free(base_slices);
    for (0..BASE_N) |i| base_slices[i] = base_data[i * DIM ..][0..DIM];
    try idx.batchInsert(base_slices);

    // Compute exact L2 distances from query to all base vectors
    const exact_dists = try allocator.alloc(f32, BASE_N);
    defer allocator.free(exact_dists);

    for (0..BASE_N) |i| {
        exact_dists[i] = simd.l2DistanceSquared(&query, base_slices[i]);
    }

    // Find exact top-20
    const top_k: u32 = 20;
    var exact_top: [20]struct { id: u32, dist: f32 } = undefined;
    // Simple selection sort for top-k
    var used = try allocator.alloc(bool, BASE_N);
    defer allocator.free(used);
    @memset(used, false);
    for (0..top_k) |ti| {
        var best_id: u32 = 0;
        var best_dist: f32 = std.math.floatMax(f32);
        for (0..BASE_N) |i| {
            if (!used[i] and exact_dists[i] < best_dist) {
                best_dist = exact_dists[i];
                best_id = @intCast(i);
            }
        }
        used[best_id] = true;
        exact_top[ti] = .{ .id = best_id, .dist = best_dist };
    }

    // Search using RaBitQ (all partitions)
    var rabitq_results: [20]index_mod.SearchResult = undefined;
    const found = try idx.search(&query, top_k, 16, &rabitq_results);

    // Print comparison
    writeStdout("Exact Top-20 vs RaBitQ Top-20 (no SQ8 refinement):\n");
    writeStdout("Rank | Exact ID | Exact Dist | RaBitQ ID | RaBitQ Score | Match?\n");
    writeStdout("-----|----------|------------|-----------|--------------|-------\n");

    var matched: u32 = 0;
    for (0..@min(top_k, found)) |i| {
        const adjusted_rabitq_id: i32 = @as(i32, @intCast(rabitq_results[i].id)) - @as(i32, @intCast(base_id_offset));
        const is_match = adjusted_rabitq_id == @as(i32, @intCast(exact_top[i].id));
        if (is_match) matched += 1;

        // Also find the RaBitQ rank of each exact top vector
        const line = try std.fmt.bufPrint(&buf,
            "  {d: >2} | {d: >7} | {d: >10.2} | {d: >9} | {d: >12.2} | {s}\n",
            .{
                i,
                exact_top[i].id,
                exact_top[i].dist,
                adjusted_rabitq_id,
                rabitq_results[i].score,
                if (is_match) "YES" else "no",
            },
        );
        writeStdout(line);
    }

    const recall = @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(top_k));
    const recall_line = try std.fmt.bufPrint(&buf, "\nRecall@{d} = {d:.4}\n", .{ top_k, recall });
    writeStdout(recall_line);

    // Also check: where does each exact top-20 vector rank in RaBitQ results?
    writeStdout("\nExact top-20 vectors' RaBitQ ranks:\n");
    // Get all RaBitQ scores
    var all_rabitq = try allocator.alloc(struct { id: u32, score: f32 }, BASE_N);
    defer allocator.free(all_rabitq);
    for (0..BASE_N) |i| {
        all_rabitq[i] = .{ .id = @intCast(i), .score = exact_dists[i] }; // placeholder
    }

    // Search with large k to get more candidates
    var big_results: [1000]index_mod.SearchResult = undefined;
    const big_found = try idx.search(&query, 1000, 16, &big_results);

    for (0..@min(top_k, 20)) |ti| {
        const target_id = @as(i32, @intCast(exact_top[ti].id)) + @as(i32, @intCast(base_id_offset));
        var rank: i32 = -1;
        for (0..big_found) |ri| {
            if (big_results[ri].id == target_id) {
                rank = @intCast(ri);
                break;
            }
        }
        const rank_line = try std.fmt.bufPrint(&buf,
            "  Exact rank {d: >2} (id={d}, dist={d:.2}) -> RaBitQ rank: {s}\n",
            .{
                ti,
                exact_top[ti].id,
                exact_top[ti].dist,
                if (rank >= 0) try std.fmt.bufPrint(&buf, "{d}", .{rank}) else "NOT FOUND",
            },
        );
        writeStdout(rank_line);
    }

    writeStdout("\nDiagnostic complete.\n");
}
