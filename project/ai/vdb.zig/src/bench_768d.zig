const std = @import("std");
const posix = std.posix;
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");

const DIM: u32 = 768;
const BASE_N: usize = 10_000;
const LEARN_N: usize = 10_000;
const QUERY_N: usize = 100;
const GT_K: u32 = 10;

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

/// Generate Gaussian random vectors.
fn generateVectors(allocator: std.mem.Allocator, rng: *std.Random.DefaultPrng, n: usize, dim: u32) ![]f32 {
    const data = try allocator.alloc(f32, n * dim);
    for (data) |*v| {
        v.* = rng.random().floatNorm(f32);
    }
    return data;
}

/// Compute exact top-k groundtruth using brute-force L2.
fn computeGroundtruth(allocator: std.mem.Allocator, queries: []const f32, base: []const f32, n_query: usize, n_base: usize, dim: u32, k: u32) ![]i32 {
    const gt = try allocator.alloc(i32, n_query * k);
    errdefer allocator.free(gt);

    for (0..n_query) |qi| {
        const q = queries[qi * dim ..][0..dim];

        // Max-heap of size k for exact top-k (root = largest of the k smallest distances)
        var heap: [10]struct { id: i32, dist: f32 } = undefined;
        var heap_len: u32 = 0;

        for (0..n_base) |bi| {
            const b = base[bi * dim ..][0..dim];
            const d = simd.l2DistanceSquared(q, b);

            if (heap_len < k) {
                heap[heap_len] = .{ .id = @intCast(bi), .dist = d };
                heap_len += 1;
                // sift up (max-heap: largest at root)
                var i = heap_len - 1;
                while (i > 0) {
                    const parent = (i - 1) / 2;
                    if (heap[i].dist > heap[parent].dist) {
                        const tmp = heap[i];
                        heap[i] = heap[parent];
                        heap[parent] = tmp;
                        i = parent;
                    } else break;
                }
            } else if (d < heap[0].dist) {
                heap[0] = .{ .id = @intCast(bi), .dist = d };
                // sift down (max-heap)
                var i: u32 = 0;
                while (true) {
                    const left = 2 * i + 1;
                    const right = 2 * i + 2;
                    var largest = i;
                    if (left < heap_len and heap[left].dist > heap[largest].dist) largest = left;
                    if (right < heap_len and heap[right].dist > heap[largest].dist) largest = right;
                    if (largest == i) break;
                    const tmp = heap[i];
                    heap[i] = heap[largest];
                    heap[largest] = tmp;
                    i = largest;
                }
            }
        }

        // Sort heap to get ordered results (ascending by distance)
        var remaining = heap_len;
        while (remaining > 1) {
            remaining -= 1;
            const tmp = heap[0];
            heap[0] = heap[remaining];
            heap[remaining] = tmp;
            var i: u32 = 0;
            while (true) {
                const left = 2 * i + 1;
                const right = 2 * i + 2;
                var largest = i;
                if (left < remaining and heap[left].dist > heap[largest].dist) largest = left;
                if (right < remaining and heap[right].dist > heap[largest].dist) largest = right;
                if (largest == i) break;
                const t = heap[i];
                heap[i] = heap[largest];
                heap[largest] = t;
                i = largest;
            }
        }

        for (0..heap_len) |i| {
            gt[qi * k + i] = heap[i].id;
        }
    }

    return gt;
}

fn computeRecall(approx: []const index_mod.SearchResult, gt: []const i32, base_id_offset: u32, k: u32) f64 {
    if (approx.len == 0 or gt.len == 0) return 0.0;
    var matched: u32 = 0;
    const check_len = @min(approx.len, k);
    for (approx[0..check_len]) |a| {
        for (gt[0..k]) |g| {
            const adjusted_id: i32 = @as(i32, @intCast(a.id)) - @as(i32, @intCast(base_id_offset));
            if (adjusted_id == g) {
                matched += 1;
                break;
            }
        }
    }
    return @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(k));
}

fn computeMemoryPerVector(idx: *const index_mod.Index) f64 {
    const n = idx.next_id.load(.monotonic);
    if (n == 0) return 0.0;
    var total_bytes: u64 = 0;
    total_bytes += idx.rotation.len * @sizeOf(f32);
    for (idx.partitions) |p| {
        total_bytes += p.codes.len * @sizeOf(u64);
        total_bytes += p.scalars.len * @sizeOf(f32);
        total_bytes += p.ids.len * @sizeOf(u32);
        total_bytes += p.centroid.len * @sizeOf(f32);
        total_bytes += p.sq8_codes.len * @sizeOf(u8);
        total_bytes += p.sq8_min.len * @sizeOf(f32);
        total_bytes += p.sq8_scale.len * @sizeOf(f32);
        total_bytes += p.sq8_max.len * @sizeOf(f32);
    }
    for (idx.super_partitions) |sp| {
        total_bytes += sp.centroid.len * @sizeOf(f32);
        total_bytes += sp.sub_ids.len * @sizeOf(u32);
    }
    return @as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(n));
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var buf: [512]u8 = undefined;

    writeStdout("========================================\n");
    writeStdout("  768D Synthetic Benchmark (vdb.zig)\n");
    writeStdout("  10k vectors, 100 queries, Gaussian\n");
    writeStdout("========================================\n\n");

    var rng = std.Random.DefaultPrng.init(42);

    // Generate datasets
    writeStdout("Generating synthetic vectors (Gaussian)...\n");
    const base_data = try generateVectors(allocator, &rng, BASE_N, DIM);
    defer allocator.free(base_data);
    const learn_data = try generateVectors(allocator, &rng, LEARN_N, DIM);
    defer allocator.free(learn_data);
    const query_data = try generateVectors(allocator, &rng, QUERY_N, DIM);
    defer allocator.free(query_data);

    writeStdout("Computing exact groundtruth (brute-force top-10)...\n");
    const gt_start = monoNs();
    const gt = try computeGroundtruth(allocator, query_data, base_data, QUERY_N, BASE_N, DIM, GT_K);
    defer allocator.free(gt);
    const gt_ns = monoNs() - gt_start;
    const gt_ms = @divTrunc(@as(i64, @intCast(gt_ns)), 1_000_000);
    const gt_info = try std.fmt.bufPrint(&buf, "  Groundtruth computed in {d}ms\n\n", .{gt_ms});
    writeStdout(gt_info);

    // Build slices
    const base_slices = try allocator.alloc([]const f32, BASE_N);
    defer allocator.free(base_slices);
    for (0..BASE_N) |i| base_slices[i] = base_data[i * DIM ..][0..DIM];

    const learn_slices = try allocator.alloc([]const f32, LEARN_N);
    defer allocator.free(learn_slices);
    for (0..LEARN_N) |i| learn_slices[i] = learn_data[i * DIM ..][0..DIM];

    const query_slices = try allocator.alloc([]const f32, QUERY_N);
    defer allocator.free(query_slices);
    for (0..QUERY_N) |i| query_slices[i] = query_data[i * DIM ..][0..DIM];

    // Benchmark configurations: explore partition count and refine_k tradeoffs
    const configs = &[_]struct {
        num_partitions: u32,
        refine_sq8: bool,
        refine_k: u32,
        fastscan: bool,
        label: []const u8,
    }{
        .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 100, .fastscan = true, .label = "8p, FS, rk=100" },
        .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 200, .fastscan = true, .label = "8p, FS, rk=200" },
        .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 500, .fastscan = true, .label = "8p, FS, rk=500" },
        .{ .num_partitions = 16, .refine_sq8 = true, .refine_k = 100, .fastscan = true, .label = "16p, FS, rk=100" },
        .{ .num_partitions = 16, .refine_sq8 = true, .refine_k = 200, .fastscan = true, .label = "16p, FS, rk=200" },
        .{ .num_partitions = 16, .refine_sq8 = true, .refine_k = 500, .fastscan = true, .label = "16p, FS, rk=500" },
    };

    const k: u32 = 10;

    for (configs) |cfg| {
        const header = try std.fmt.bufPrint(&buf, "--- {s} ---\n", .{cfg.label});
        writeStdout(header);

        const build_start = monoNs();
        var idx = try index_mod.Index.init(allocator, DIM, .{
            .num_partitions = cfg.num_partitions,
            .refine_sq8 = cfg.refine_sq8,
            .refine_k = cfg.refine_k,
            .fastscan = cfg.fastscan,
            .rotation_seed = 12345,
        });
        defer idx.deinit();

        if (LEARN_N >= cfg.num_partitions) {
            try idx.train(learn_slices);
        }
        const base_id_offset = idx.next_id.load(.monotonic);
        try idx.batchInsert(base_slices);
        const build_ns = monoNs() - build_start;
        const build_ms = @divTrunc(@as(i64, @intCast(build_ns)), 1_000_000);

        const mem_per_vec = computeMemoryPerVector(&idx);
        const original_bytes = @as(f64, @floatFromInt(DIM)) * @sizeOf(f32);
        const compression = if (mem_per_vec > 0) original_bytes / mem_per_vec else 0.0;

        const build_info = try std.fmt.bufPrint(&buf,
            "  Build: {d}ms | Mem/vec: {d:.1}B ({d:.2}x)\n",
            .{ build_ms, mem_per_vec, compression },
        );
        writeStdout(build_info);

        // Test with different nprobe values
        const nprobe_values = &[_]u32{ 1, 2, 4, 8, 16, 32 };

        for (nprobe_values) |nprobe| {
            if (nprobe > idx.partitions.len) continue;

            var latencies = try allocator.alloc(u64, QUERY_N);
            defer allocator.free(latencies);

            var approx_results = try allocator.alloc(index_mod.SearchResult, k);
            defer allocator.free(approx_results);

            var total_recall: f64 = 0.0;
            for (0..QUERY_N) |qi| {
                const t0 = monoNs();
                const found = idx.search(query_slices[qi], k, nprobe, approx_results) catch 0;
                const t1 = monoNs();
                latencies[qi] = t1 - t0;
                if (found > 0) {
                    const gt_row = gt[qi * k ..][0..k];
                    const recall = computeRecall(approx_results[0..found], gt_row, base_id_offset, k);
                    total_recall += recall;
                }
            }

            std.mem.sortUnstable(u64, latencies, {}, comptime std.sort.asc(u64));
            const p50_ns = latencies[QUERY_N / 2];
            const p99_ns = latencies[@min(QUERY_N - 1, QUERY_N * 99 / 100)];
            var total_ns: u64 = 0;
            for (latencies) |l| total_ns += l;
            const qps = if (total_ns > 0)
                @as(f64, @floatFromInt(QUERY_N)) / (@as(f64, @floatFromInt(total_ns)) / 1e9)
            else
                0.0;
            const avg_recall = total_recall / @as(f64, @floatFromInt(QUERY_N));

            const result = try std.fmt.bufPrint(&buf,
                "  nprobe={d: >2} | QPS={d: >7.1} | p50={d: >7.0}us | p99={d: >7.0}us | R@{d}={d:.4}\n",
                .{ nprobe, qps, @as(f64, @floatFromInt(p50_ns)) / 1000.0, @as(f64, @floatFromInt(p99_ns)) / 1000.0, k, avg_recall },
            );
            writeStdout(result);
        }
        writeStdout("\n");
    }

    writeStdout("Benchmark complete.\n");
}
