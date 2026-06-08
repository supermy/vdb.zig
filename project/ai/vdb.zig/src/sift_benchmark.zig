const std = @import("std");
const posix = std.posix;
const index_mod = @import("index_ivf_rq");

/// Simple monotonic clock using POSIX clock_gettime.
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

/// Read entire file into memory using Zig 0.16.0 I/O API.
fn readFileBytes(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFile(std.Io.Dir.cwd(), io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const data = try allocator.alloc(u8, @intCast(stat.size));
    errdefer allocator.free(data);
    _ = try file.readPositionalAll(io, data, 0);
    return data;
}

/// Read .fvecs file: each vector prefixed with 4-byte dimension.
fn readFvecs(allocator: std.mem.Allocator, path: []const u8) !struct { data: []f32, dim: u32, n: usize } {
    const bytes = try readFileBytes(allocator, path);
    defer allocator.free(bytes);
    if (bytes.len < 4) return error.InvalidFormat;

    const dim = std.mem.readInt(i32, bytes[0..4], .little);
    if (dim <= 0 or @rem(dim, 64) != 0) return error.InvalidDimension;
    const dim_u: u32 = @intCast(dim);
    const vec_size = 4 + dim_u * 4;
    const n = @divExact(bytes.len, vec_size);

    const data = try allocator.alloc(f32, n * dim_u);
    errdefer allocator.free(data);

    for (0..n) |i| {
        const offset = i * vec_size;
        const read_dim = std.mem.readInt(i32, bytes[offset..][0..4], .little);
        if (read_dim != dim) return error.DimensionMismatch;
        const floats = std.mem.bytesAsSlice(f32, bytes[offset + 4 ..][0 .. dim_u * 4]);
        @memcpy(data[i * dim_u ..][0..dim_u], floats);
    }

    return .{ .data = data, .dim = dim_u, .n = n };
}

/// Read .ivecs file: each row prefixed with 4-byte count.
fn readIvecs(allocator: std.mem.Allocator, path: []const u8) !struct { data: []i32, count: u32, n: usize } {
    const bytes = try readFileBytes(allocator, path);
    defer allocator.free(bytes);
    if (bytes.len < 4) return error.InvalidFormat;

    const count = std.mem.readInt(i32, bytes[0..4], .little);
    if (count <= 0) return error.InvalidFormat;
    const count_u: u32 = @intCast(count);
    const row_size = 4 + count_u * 4;
    const n = @divExact(bytes.len, row_size);

    const data = try allocator.alloc(i32, n * count_u);
    errdefer allocator.free(data);

    for (0..n) |i| {
        const offset = i * row_size;
        const read_count = std.mem.readInt(i32, bytes[offset..][0..4], .little);
        if (read_count != count) return error.CountMismatch;
        const ints = std.mem.bytesAsSlice(i32, bytes[offset + 4 ..][0 .. count_u * 4]);
        @memcpy(data[i * count_u ..][0..count_u], ints);
    }

    return .{ .data = data, .count = count_u, .n = n };
}

fn computeRecall(approx: []const index_mod.SearchResult, gt: []const i32, base_id_offset: u32) f64 {
    if (approx.len == 0 or gt.len == 0) return 0.0;
    var matched: u32 = 0;
    for (approx) |a| {
        for (gt) |g| {
            // Groundtruth IDs are base file indices (0..n_base-1).
            // Index global IDs include learn vectors first, so subtract offset.
            const adjusted_id: i32 = @as(i32, @intCast(a.id)) - @as(i32, @intCast(base_id_offset));
            if (adjusted_id == g) {
                matched += 1;
                break;
            }
        }
    }
    return @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(@min(approx.len, gt.len)));
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
    const data_dir = "/Users/moyong/project/ai/models/data/siftsmall";

    writeStdout("========================================\n");
    writeStdout("  SIFT Small Benchmark (vdb.zig)\n");
    writeStdout("========================================\n\n");

    // Load datasets
    const base_path = try std.fs.path.join(allocator, &.{ data_dir, "siftsmall_base.fvecs" });
    defer allocator.free(base_path);
    const learn_path = try std.fs.path.join(allocator, &.{ data_dir, "siftsmall_learn.fvecs" });
    defer allocator.free(learn_path);
    const query_path = try std.fs.path.join(allocator, &.{ data_dir, "siftsmall_query.fvecs" });
    defer allocator.free(query_path);
    const gt_path = try std.fs.path.join(allocator, &.{ data_dir, "siftsmall_groundtruth.ivecs" });
    defer allocator.free(gt_path);

    writeStdout("Loading datasets...\n");

    const base = try readFvecs(allocator, base_path);
    defer allocator.free(base.data);
    const learn = try readFvecs(allocator, learn_path);
    defer allocator.free(learn.data);
    const query = try readFvecs(allocator, query_path);
    defer allocator.free(query.data);
    const gt = try readIvecs(allocator, gt_path);
    defer allocator.free(gt.data);

    var buf: [512]u8 = undefined;
    const info = try std.fmt.bufPrint(&buf,
        "  Base:     {d} vectors x {d}D\n" ++
        "  Learn:    {d} vectors x {d}D\n" ++
        "  Query:    {d} vectors x {d}D\n" ++
        "  Groundtruth: {d} queries x top-{d}\n\n",
        .{ base.n, base.dim, learn.n, learn.dim, query.n, query.dim, gt.n, gt.count },
    );
    writeStdout(info);

    // Build slices for batch insert
    const base_slices = try allocator.alloc([]const f32, base.n);
    defer allocator.free(base_slices);
    for (0..base.n) |i| {
        base_slices[i] = base.data[i * base.dim ..][0..base.dim];
    }

    const learn_slices = try allocator.alloc([]const f32, learn.n);
    defer allocator.free(learn_slices);
    for (0..learn.n) |i| {
        learn_slices[i] = learn.data[i * learn.dim ..][0..learn.dim];
    }

    const query_slices = try allocator.alloc([]const f32, query.n);
    defer allocator.free(query_slices);
    for (0..query.n) |i| {
        query_slices[i] = query.data[i * query.dim ..][0..query.dim];
    }

    // Benchmark configurations
    const configs = &[_]struct {
        num_partitions: u32,
        refine_sq8: bool,
        refine_k: u32,
        fastscan: bool,
        label: []const u8,
    }{
        .{ .num_partitions = 32, .refine_sq8 = true, .refine_k = 10, .fastscan = false, .label = "Refined (32p, SQ8 rk=10, no fastscan)" },
    };

    const nprobe_values = &[_]u32{ 4, 8, 16, 32, 64 };
    const k: u32 = 10;

    for (configs) |cfg| {
        const header = try std.fmt.bufPrint(&buf, "--- {s} ---\n", .{cfg.label});
        writeStdout(header);

        // Build index: train on learn, insert base
        const build_start = monoNs();
        var idx = try index_mod.Index.init(allocator, base.dim, .{
            .num_partitions = cfg.num_partitions,
            .refine_sq8 = cfg.refine_sq8,
            .refine_k = cfg.refine_k,
            .fastscan = cfg.fastscan,
        });
        defer idx.deinit();

        // Train centroids using learn set (if enough vectors)
        if (learn.n >= cfg.num_partitions) {
            try idx.train(learn_slices);
        }

        // Record base vector starting global ID before inserting
        const base_id_offset = idx.next_id.load(.monotonic);

        // Insert base vectors
        try idx.batchInsert(base_slices);
        const build_ns = monoNs() - build_start;
        const build_ms = @divTrunc(@as(i64, @intCast(build_ns)), 1_000_000);

        const mem_per_vec = computeMemoryPerVector(&idx);
        const original_bytes = @as(f64, @floatFromInt(base.dim)) * @sizeOf(f32);
        const compression = if (mem_per_vec > 0) original_bytes / mem_per_vec else 0.0;

        const build_info = try std.fmt.bufPrint(&buf,
            "  Build time: {d}ms | Memory/vec: {d:.2}B | Compression: {d:.2}x\n",
            .{ build_ms, mem_per_vec, compression },
        );
        writeStdout(build_info);

        for (nprobe_values) |nprobe| {
            if (nprobe > idx.partitions.len) continue;

            var latencies = try allocator.alloc(u64, query.n);
            defer allocator.free(latencies);

            var approx_results = try allocator.alloc(index_mod.SearchResult, k);
            defer allocator.free(approx_results);

            var total_recall: f64 = 0.0;
            for (0..query.n) |qi| {
                const t0 = monoNs();
                const found = idx.search(query_slices[qi], k, nprobe, approx_results) catch 0;
                const t1 = monoNs();
                latencies[qi] = t1 - t0;
                if (found > 0) {
                    const gt_row = gt.data[qi * gt.count ..][0..gt.count];
                    total_recall += computeRecall(approx_results[0..found], gt_row, base_id_offset);
                }
            }

            std.mem.sortUnstable(u64, latencies, {}, comptime std.sort.asc(u64));
            const p50_ns = latencies[query.n / 2];
            const p99_ns = latencies[@min(query.n - 1, query.n * 99 / 100)];
            var total_ns: u64 = 0;
            for (latencies) |l| total_ns += l;
            const qps = if (total_ns > 0)
                @as(f64, @floatFromInt(query.n)) / (@as(f64, @floatFromInt(total_ns)) / 1e9)
            else
                0.0;
            const avg_recall = total_recall / @as(f64, @floatFromInt(query.n));

            const result = try std.fmt.bufPrint(&buf,
                "  nprobe={d: >3} | QPS={d: >7.1} | p50={d: >4.0}us | p99={d: >4.0}us | Recall@{d}={d:.4}\n",
                .{ nprobe, qps, @as(f64, @floatFromInt(p50_ns)) / 1000.0, @as(f64, @floatFromInt(p99_ns)) / 1000.0, k, avg_recall },
            );
            writeStdout(result);
        }
        writeStdout("\n");
    }

    writeStdout("Benchmark complete.\n");
}
