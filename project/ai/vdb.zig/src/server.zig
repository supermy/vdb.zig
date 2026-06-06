const std = @import("std");
const vdb = @import("vdb");
const index_mod = @import("index_ivf_rq");
const search_mod = @import("search");
const gpu = @import("gpu");
const posix = std.posix;

/// OpenAI/Anthropic-compatible HTTP API server for vdb.zig.
/// Serves the default web test page at GET / and API endpoints at POST /v1/search, etc.
const PORT = 8080;

/// Spin-lock helper for std.atomic.Mutex (same pattern as thread_pool.zig).
fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

const Response = struct {
    status: []const u8,
    data: ?std.json.Value,
    error_message: ?[]const u8,
};

/// Simple monotonic clock using POSIX clock_gettime.
fn monoNs() u64 {
    var ts: posix.timespec = undefined;
    const rc = posix.system.clock_gettime(posix.CLOCK.MONOTONIC, &ts);
    std.debug.assert(rc == 0);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Brute-force exact top-k search for recall measurement.
fn bruteForceTopK(
    allocator: std.mem.Allocator,
    dataset: []const f32,
    dim: u32,
    query: []const f32,
    k: u32,
) ![]index_mod.SearchResult {
    const n = dataset.len / dim;
    var results = try allocator.alloc(index_mod.SearchResult, n);
    defer allocator.free(results);
    for (0..n) |i| {
        const vec = dataset[i * dim ..][0..dim];
        var dist: f32 = 0.0;
        for (vec, query) |v, q| {
            const diff = v - q;
            dist += diff * diff;
        }
        results[i] = .{ .id = @intCast(i), .partition_id = 0, .score = dist };
    }
    std.mem.sortUnstable(index_mod.SearchResult, results, {}, struct {
        fn lessThan(_: void, a: index_mod.SearchResult, b: index_mod.SearchResult) bool {
            return a.score < b.score;
        }
    }.lessThan);
    const out = try allocator.alloc(index_mod.SearchResult, @min(k, n));
    @memcpy(out, results[0..out.len]);
    return out;
}

/// Compute recall@k: fraction of approximate top-k IDs that appear in exact top-k.
fn computeRecall(approx: []index_mod.SearchResult, exact: []index_mod.SearchResult) f64 {
    if (approx.len == 0 or exact.len == 0) return 0.0;
    var matched: u32 = 0;
    for (approx) |a| {
        for (exact) |e| {
            if (a.id == e.id) {
                matched += 1;
                break;
            }
        }
    }
    return @as(f64, @floatFromInt(matched)) / @as(f64, @floatFromInt(@min(approx.len, exact.len)));
}

/// Compute memory per vector in bytes.
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

/// Thread-safe storage for raw vectors, used for recall testing.
/// Protected by the same rwlock as the index.
const RawVectorStore = struct {
    data: std.ArrayList(f32),
    dim: u32,

    fn init(dim: u32) RawVectorStore {
        return .{
            .data = .empty,
            .dim = dim,
        };
    }

    fn addVector(self: *RawVectorStore, allocator: std.mem.Allocator, vec: []const f32) !void {
        try self.data.appendSlice(allocator, vec);
    }

    fn getVector(self: *const RawVectorStore, i: usize) []const f32 {
        const start = i * self.dim;
        return self.data.items[start..][0..self.dim];
    }

    fn len(self: *const RawVectorStore) usize {
        if (self.dim == 0) return 0;
        return self.data.items.len / self.dim;
    }

    fn deinit(self: *RawVectorStore, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
    }
};

pub fn main() !void {
    // Production server uses page_allocator; use DebugAllocator only in debug builds.
    const allocator = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    // Load or create a demo index
    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 16 });
    defer idx.deinit();

    var raw_store = RawVectorStore.init(128);
    defer raw_store.deinit(allocator);

    // Pre-populate with random demo vectors
    var rng = std.Random.DefaultPrng.init(99);
    var vec: [128]f32 = undefined;
    for (0..1000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
        try raw_store.addVector(allocator, &vec);
    }

    const address = try std.Io.net.IpAddress.parseIp4("0.0.0.0", PORT);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.log.info("vdb-server listening on http://0.0.0.0:{d}", .{PORT});

    var idx_mutex = std.atomic.Mutex.unlocked;

    while (true) {
        const stream = try server.accept(io);
        const t = std.Thread.spawn(.{}, connectionHandler, .{ allocator, stream, &idx, &idx_mutex, &raw_store }) catch {
            stream.close(io);
            continue;
        };
        t.detach();
    }
}

fn connectionHandler(
    allocator: std.mem.Allocator,
    stream: std.Io.net.Stream,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) void {
    // Each thread must use its own Io instance; global_single_threaded is not safe to share across threads.
    const io = std.Io.Threaded.global_single_threaded.io();
    handleConnection(allocator, io, stream, idx, rwlock, raw_store) catch |err| {
        std.log.err("Connection error: {}", .{err});
    };
}

fn parseContentLength(headers: []const u8) ?usize {
    var it = std.mem.splitSequence(u8, headers, "\r\n");
    while (it.next()) |line| {
        if (line.len > "content-length:".len) {
            const prefix = line[0.."content-length:".len];
            if (std.ascii.eqlIgnoreCase(prefix, "content-length:")) {
                const val = std.mem.trim(u8, line["content-length:".len..], " ");
                return std.fmt.parseInt(usize, val, 10) catch null;
            }
        }
    }
    return null;
}

fn readHttpRequest(
    allocator: std.mem.Allocator,
    reader: anytype,
    max_size: usize,
) !?[]const u8 {
    var buf = std.array_list.Managed(u8).init(allocator);
    errdefer buf.deinit();

    var temp: [4096]u8 = undefined;
    while (buf.items.len < max_size) {
        const n = reader.readSliceShort(&temp) catch break;
        if (n == 0) break;
        try buf.appendSlice(temp[0..n]);

        if (std.mem.indexOf(u8, buf.items, "\r\n\r\n")) |header_end| {
            const body_start = header_end + 4;
            const content_length = parseContentLength(buf.items[0..body_start]) orelse 0;
            const total_needed = body_start + content_length;

            if (total_needed > max_size) return error.PayloadTooLarge;
            if (buf.items.len >= total_needed) {
                return try buf.toOwnedSlice();
            }
        }
    }

    if (buf.items.len > max_size) return error.PayloadTooLarge;
    if (buf.items.len == 0) {
        buf.deinit();
        return null;
    }
    return try buf.toOwnedSlice();
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    stream: std.Io.net.Stream,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) !void {
    defer stream.close(io);

    var reader_buf: [4096]u8 = undefined;
    var reader = stream.reader(io, &reader_buf);

    const MAX_REQUEST_SIZE = 65536;
    const request_opt = readHttpRequest(allocator, &reader.interface, MAX_REQUEST_SIZE) catch |err| {
        var writer_buf: [256]u8 = undefined;
        var writer = stream.writer(io, &writer_buf);
        if (err == error.PayloadTooLarge) {
            try writer.interface.writeAll("HTTP/1.1 413 Payload Too Large\r\nConnection: close\r\n\r\n");
            return;
        }
        return err;
    };
    defer if (request_opt) |r| allocator.free(r);
    if (request_opt == null or request_opt.?.len < 4) return;
    const request = request_opt.?;

    var writer_buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &writer_buf);

    // Very naive HTTP parsing for skeleton
    if (std.mem.startsWith(u8, request, "GET / ")) {
        try serveStaticFile(&writer, allocator, io, "web/index.html", "text/html");
    } else if (std.mem.startsWith(u8, request, "GET /app.js ")) {
        try serveStaticFile(&writer, allocator, io, "web/app.js", "application/javascript");
    } else if (std.mem.startsWith(u8, request, "GET /style.css ")) {
        try serveStaticFile(&writer, allocator, io, "web/style.css", "text/css");
    } else if (std.mem.startsWith(u8, request, "GET /readme ")) {
        try serveStaticFile(&writer, allocator, io, "README.md", "text/markdown; charset=UTF-8");
    } else if (std.mem.startsWith(u8, request, "POST /v1/search ")) {
        try handleSearch(&writer, allocator, request, idx, rwlock);
    } else if (std.mem.startsWith(u8, request, "POST /v1/batch_search ")) {
        try handleBatchSearch(&writer, allocator, request, idx, rwlock);
    } else if (std.mem.startsWith(u8, request, "POST /v1/import ")) {
        try handleImport(&writer, allocator, request, idx, rwlock, raw_store);
    } else if (std.mem.startsWith(u8, request, "POST /v1/export ")) {
        try handleExport(&writer, allocator, request, idx);
    } else if (std.mem.startsWith(u8, request, "POST /v1/benchmark ")) {
        try handleBenchmark(&writer, allocator, request);
    } else if (std.mem.startsWith(u8, request, "GET /v1/stats ")) {
        try handleStats(&writer, allocator, idx);
    } else if (std.mem.startsWith(u8, request, "POST /v1/recall_test ")) {
        try handleRecallTest(&writer, allocator, request, idx, rwlock, raw_store);
    } else if (std.mem.startsWith(u8, request, "GET /health ")) {
        try sendJson(&writer, allocator, .{ .status = "ok", .version = "0.1.0" });
    } else {
        try sendJsonError(&writer, allocator, .{ .status = "error", .message = "Not Found" }, "404 Not Found");
    }
}

fn serveStaticFile(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    content_type: []const u8,
) !void {
    const MAX_STATIC_FILE_SIZE = 1024 * 1024; // 1MB limit
    const file = std.Io.Dir.openFile(std.Io.Dir.cwd(), io, path, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "File not found" });
        return;
    };
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size > MAX_STATIC_FILE_SIZE) {
        try sendJson(writer, allocator, .{ .status = "error", .message = "File too large" });
        return;
    }
    const body = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(body);
    _ = try file.readPositionalAll(io, body, 0);

    const header = try std.fmt.allocPrint(allocator, "HTTP/1.1 200 OK\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ content_type, body.len });
    defer allocator.free(header);
    try writer.interface.writeAll(header);
    try writer.interface.writeAll(body);
}

fn handleSearch(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
) !void {
    // Extract JSON body after \r\n\r\n
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        vector: []const f32,
        k: ?u32,
        nprobe: ?u32,
    }, allocator, body, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Invalid JSON" });
        return;
    };
    defer parsed.deinit();

    const k = @min(parsed.value.k orelse 10, 256);
    const nprobe = parsed.value.nprobe orelse 8;

    var results: [256]index_mod.SearchResult = undefined;
    spinLock(rwlock);
    defer rwlock.unlock();
    const found = idx.search(parsed.value.vector, k, nprobe, &results) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Search failed" });
        return;
    };

    var arr = std.json.Array.init(allocator);
    defer arr.deinit();
    for (0..found) |i| {
        var obj: std.json.ObjectMap = .{};
        try obj.put(allocator, "id", .{ .integer = results[i].id });
        try obj.put(allocator, "partition_id", .{ .integer = results[i].partition_id });
        try obj.put(allocator, "score", .{ .float = results[i].score });
        try arr.append(.{ .object = obj });
    }

    const response = .{
        .status = "ok",
        .count = found,
        .results = arr.items,
    };
    try sendJson(writer, allocator, response);
}

fn handleBatchSearch(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        queries: []const struct { vector: []const f32, k: ?u32 },
        nprobe: ?u32,
    }, allocator, body, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Invalid JSON" });
        return;
    };
    defer parsed.deinit();

    const nprobe = parsed.value.nprobe orelse 8;
    var batch_results = std.json.Array.init(allocator);
    defer batch_results.deinit();

    for (parsed.value.queries) |q| {
        const k = @min(q.k orelse 10, 256);
        var results: [256]index_mod.SearchResult = undefined;
        spinLock(rwlock);
        defer rwlock.unlock();
        const found = idx.search(q.vector, k, nprobe, &results) catch {
            continue;
        };

        var arr = std.json.Array.init(allocator);
        for (0..found) |i| {
            var obj: std.json.ObjectMap = .{};
            try obj.put(allocator, "id", .{ .integer = results[i].id });
            try obj.put(allocator, "score", .{ .float = results[i].score });
            try arr.append(.{ .object = obj });
        }

        var item: std.json.ObjectMap = .{};
        try item.put(allocator, "results", .{ .array = arr });
        try batch_results.append(.{ .object = item });
    }

    try sendJson(writer, allocator, .{ .status = "ok", .results = batch_results.items });
}

fn handleImport(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        vectors: []const []const f32,
    }, allocator, body, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Invalid JSON" });
        return;
    };
    defer parsed.deinit();

    spinLock(rwlock);
    defer rwlock.unlock();
    // Use batchInsert for better cache locality and pre-assigned partition mapping.
    try idx.batchInsert(parsed.value.vectors);
    const imported = parsed.value.vectors.len;

    // Store raw vectors for recall testing
    for (parsed.value.vectors) |vec| {
        raw_store.addVector(allocator, vec) catch break;
    }

    try sendJson(writer, allocator, .{ .status = "ok", .imported = imported });
}

fn handleExport(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
    idx: *index_mod.Index,
) !void {
    _ = request;
    // Placeholder: export metadata about current index state
    try sendJson(writer, allocator, .{
        .status = "ok",
        .partitions = idx.partitions.len,
        .dimension = idx.dim,
        .note = "Full vector export not yet implemented in skeleton",
    });
}

fn handleBenchmark(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        dim: u32,
        n: u32,
        query_count: u32,
        k: u32,
        nprobe_values: []const u32,
        fastscan: bool,
        query_bits: u32,
    }, allocator, body, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Invalid JSON" });
        return;
    };
    defer parsed.deinit();

    const dim = parsed.value.dim;
    const n = parsed.value.n;
    const query_count = parsed.value.query_count;
    const k = parsed.value.k;
    const fastscan = parsed.value.fastscan;
    const query_bits = parsed.value.query_bits;

    if (dim % 64 != 0) {
        try sendJson(writer, allocator, .{ .status = "error", .message = "dim must be a multiple of 64" });
        return;
    }
    if (n == 0) {
        try sendJson(writer, allocator, .{ .status = "error", .message = "n must be > 0" });
        return;
    }

    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    // Generate dataset
    const dataset = bench_allocator.alloc(f32, n * dim) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate dataset" });
        return;
    };
    defer bench_allocator.free(dataset);
    for (dataset) |*v| {
        v.* = rng.random().float(f32);
    }

    // Build index — use sqrt(n) partitions capped at 128
    const num_partitions = @max(4, @min(@as(u32, @intFromFloat(@sqrt(@as(f64, @floatFromInt(n))))), 128));
    const build_start = monoNs();

    var idx = index_mod.Index.init(bench_allocator, dim, .{
        .num_partitions = num_partitions,
        .refine_sq8 = true,
        .refine_k = 20,
        .fastscan = fastscan,
        .query_bits = query_bits,
    }) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to build index" });
        return;
    };
    defer idx.deinit();

    const vec_slices = bench_allocator.alloc([]const f32, n) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate vec_slices" });
        return;
    };
    defer bench_allocator.free(vec_slices);
    for (0..n) |i| {
        vec_slices[i] = dataset[i * dim ..][0..dim];
    }
    idx.batchInsert(vec_slices) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to insert vectors" });
        return;
    };
    const build_ns = monoNs() - build_start;
    const build_ms = @divTrunc(@as(i64, @intCast(build_ns)), 1_000_000);

    // Memory and compression metrics
    const mem_per_vec = computeMemoryPerVector(&idx);
    const original_bytes_per_vec: f64 = @as(f64, @floatFromInt(dim)) * @sizeOf(f32);
    const compression_ratio = if (mem_per_vec > 0) original_bytes_per_vec / mem_per_vec else 0.0;

    // Generate queries
    const actual_query_count = @min(query_count, n);
    const query_indices = bench_allocator.alloc(u32, actual_query_count) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate query indices" });
        return;
    };
    defer bench_allocator.free(query_indices);
    for (0..actual_query_count) |i| {
        query_indices[i] = rng.random().intRangeLessThan(u32, 0, @intCast(n));
    }

    // Pre-compute exact top-k for recall measurement
    var exact_results_list = std.ArrayList([]index_mod.SearchResult).empty;
    defer {
        for (exact_results_list.items) |r| bench_allocator.free(r);
        exact_results_list.deinit(bench_allocator);
    }
    for (0..actual_query_count) |qi| {
        const q = dataset[query_indices[qi] * dim ..][0..dim];
        const exact = bruteForceTopK(bench_allocator, dataset, dim, q, k) catch {
            try sendJson(writer, allocator, .{ .status = "error", .message = "Brute-force search failed" });
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to store exact results" });
            return;
        };
    }

    // Benchmark across nprobe values
    var results_arr = std.json.Array.init(allocator);
    defer results_arr.deinit();

    for (parsed.value.nprobe_values) |nprobe_val| {
        if (nprobe_val > idx.partitions.len) continue;

        var latencies = bench_allocator.alloc(u64, actual_query_count) catch continue;
        defer bench_allocator.free(latencies);
        var approx_results = bench_allocator.alloc(index_mod.SearchResult, k) catch continue;
        defer bench_allocator.free(approx_results);

        var total_recall: f64 = 0.0;
        for (0..actual_query_count) |qi| {
            const q = dataset[query_indices[qi] * dim ..][0..dim];
            const t0 = monoNs();
            const found = idx.search(q, k, nprobe_val, approx_results) catch 0;
            const t1 = monoNs();
            latencies[qi] = t1 - t0;
            if (found > 0) {
                total_recall += computeRecall(approx_results[0..found], exact_results_list.items[qi]);
            }
        }

        // Sort latencies for percentile calculation
        std.mem.sortUnstable(u64, latencies, {}, comptime std.sort.asc(u64));
        const p50_ns = latencies[actual_query_count / 2];
        const p99_ns = latencies[@min(actual_query_count - 1, actual_query_count * 99 / 100)];
        var total_ns: u64 = 0;
        for (latencies) |l| total_ns += l;
        const qps = if (total_ns > 0)
            @as(f64, @floatFromInt(actual_query_count)) / (@as(f64, @floatFromInt(total_ns)) / 1e9)
        else
            0.0;
        const avg_recall = total_recall / @as(f64, @floatFromInt(actual_query_count));

        var obj: std.json.ObjectMap = .{};
        try obj.put(allocator, "nprobe", .{ .integer = nprobe_val });
        try obj.put(allocator, "qps", .{ .float = qps });
        try obj.put(allocator, "p50_us", .{ .float = @as(f64, @floatFromInt(p50_ns)) / 1000.0 });
        try obj.put(allocator, "p99_us", .{ .float = @as(f64, @floatFromInt(p99_ns)) / 1000.0 });
        try obj.put(allocator, "recall_at_k", .{ .float = avg_recall });
        try results_arr.append(.{ .object = obj });
    }

    try sendJson(writer, allocator, .{
        .status = "ok",
        .build_time_ms = build_ms,
        .memory_per_vector = mem_per_vec,
        .compression_ratio = compression_ratio,
        .results = results_arr.items,
    });
}

fn handleStats(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    idx: *index_mod.Index,
) !void {
    try sendJson(writer, allocator, .{
        .status = "ok",
        .dimension = idx.dim,
        .num_partitions = idx.partitions.len,
        .total_vectors = idx.next_id.load(.monotonic),
        .config = .{
            .num_bits = idx.config.num_bits,
            .epsilon_0 = idx.config.epsilon_0,
            .refine_sq8 = idx.config.refine_sq8,
            .refine_k = idx.config.refine_k,
            .fastscan = idx.config.fastscan,
            .query_bits = idx.config.query_bits,
        },
    });
}

fn handleRecallTest(
    writer: *std.Io.net.Stream.Writer,
    allocator: std.mem.Allocator,
    request: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        query_count: u32,
        k: u32,
        nprobe: u32,
    }, allocator, body, .{}) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Invalid JSON" });
        return;
    };
    defer parsed.deinit();

    const query_count = parsed.value.query_count;
    const k = parsed.value.k;
    const nprobe = parsed.value.nprobe;

    spinLock(rwlock);
    defer rwlock.unlock();

    const total_vectors = raw_store.len();
    if (total_vectors == 0) {
        try sendJson(writer, allocator, .{ .status = "error", .message = "No vectors in index" });
        return;
    }

    const actual_nprobe = @min(nprobe, @as(u32, @intCast(idx.partitions.len)));
    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    // Build flat dataset from raw store for brute-force
    const dataset = bench_allocator.alloc(f32, total_vectors * idx.dim) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate dataset" });
        return;
    };
    defer bench_allocator.free(dataset);
    for (0..total_vectors) |i| {
        const vec = raw_store.getVector(i);
        @memcpy(dataset[i * idx.dim ..][0..idx.dim], vec);
    }

    // Generate random query indices
    const actual_query_count = @min(query_count, @as(u32, @intCast(total_vectors)));
    const query_indices = bench_allocator.alloc(u32, actual_query_count) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate query indices" });
        return;
    };
    defer bench_allocator.free(query_indices);
    for (0..actual_query_count) |i| {
        query_indices[i] = rng.random().intRangeLessThan(u32, 0, @intCast(total_vectors));
    }

    // Pre-compute exact top-k
    var exact_results_list = std.ArrayList([]index_mod.SearchResult).empty;
    defer {
        for (exact_results_list.items) |r| bench_allocator.free(r);
        exact_results_list.deinit(bench_allocator);
    }
    for (0..actual_query_count) |qi| {
        const q = raw_store.getVector(query_indices[qi]);
        const exact = bruteForceTopK(bench_allocator, dataset, idx.dim, q, k) catch {
            try sendJson(writer, allocator, .{ .status = "error", .message = "Brute-force search failed" });
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to store exact results" });
            return;
        };
    }

    // Run search and measure
    var latencies = bench_allocator.alloc(u64, actual_query_count) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate latencies" });
        return;
    };
    defer bench_allocator.free(latencies);
    var approx_results = bench_allocator.alloc(index_mod.SearchResult, k) catch {
        try sendJson(writer, allocator, .{ .status = "error", .message = "Failed to allocate results" });
        return;
    };
    defer bench_allocator.free(approx_results);

    var total_recall: f64 = 0.0;
    for (0..actual_query_count) |qi| {
        const q = raw_store.getVector(query_indices[qi]);
        const t0 = monoNs();
        const found = idx.search(q, k, actual_nprobe, approx_results) catch 0;
        const t1 = monoNs();
        latencies[qi] = t1 - t0;
        if (found > 0) {
            total_recall += computeRecall(approx_results[0..found], exact_results_list.items[qi]);
        }
    }

    var total_ns: u64 = 0;
    for (latencies) |l| total_ns += l;
    const qps = if (total_ns > 0)
        @as(f64, @floatFromInt(actual_query_count)) / (@as(f64, @floatFromInt(total_ns)) / 1e9)
    else
        0.0;
    const avg_recall = total_recall / @as(f64, @floatFromInt(actual_query_count));

    try sendJson(writer, allocator, .{
        .status = "ok",
        .recall_at_k = avg_recall,
        .qps = qps,
    });
}

fn sendJson(writer: *std.Io.net.Stream.Writer, allocator: std.mem.Allocator, value: anytype) !void {
    return sendJsonWithStatus(writer, allocator, value, "200 OK");
}

fn sendJsonError(writer: *std.Io.net.Stream.Writer, allocator: std.mem.Allocator, value: anytype, status: []const u8) !void {
    return sendJsonWithStatus(writer, allocator, value, status);
}

fn sendJsonWithStatus(writer: *std.Io.net.Stream.Writer, allocator: std.mem.Allocator, value: anytype, status: []const u8) !void {
    var json_out: std.Io.Writer.Allocating = .init(allocator);
    defer json_out.deinit();
    try json_out.writer.print("{f}", .{std.json.fmt(value, .{})});
    const json_str = json_out.written();

    const header = try std.fmt.allocPrint(allocator, "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ status, json_str.len });
    defer allocator.free(header);
    try writer.interface.writeAll(header);
    try writer.interface.writeAll(json_str);
}

const MockReader = struct {
    data: []const u8,
    pos: usize = 0,
    chunk_size: usize = 1,

    pub fn readSliceShort(self: *@This(), buf: []u8) !usize {
        if (self.pos >= self.data.len) return 0;
        const remaining = self.data.len - self.pos;
        const to_read = @min(buf.len, @min(self.chunk_size, remaining));
        @memcpy(buf[0..to_read], self.data[self.pos .. self.pos + to_read]);
        self.pos += to_read;
        return to_read;
    }
};

test "readHttpRequest assembles fragmented GET request" {
    const allocator = std.testing.allocator;
    const request_str = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";

    var reader = MockReader{
        .data = request_str,
        .chunk_size = 1,
    };

    const result = try readHttpRequest(allocator, &reader, 65536);
    defer if (result) |r| allocator.free(r);

    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings(request_str, result.?);
}

test "readHttpRequest assembles fragmented POST with body" {
    const allocator = std.testing.allocator;
    const body = "{\"vector\":[1.0,2.0]}";
    const request_str = "POST /v1/search HTTP/1.1\r\nContent-Length: 20\r\n\r\n" ++ body;

    var reader = MockReader{
        .data = request_str,
        .chunk_size = 3,
    };

    const result = try readHttpRequest(allocator, &reader, 65536);
    defer if (result) |r| allocator.free(r);

    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings(request_str, result.?);
}

test "readHttpRequest returns 413 for oversized Content-Length" {
    const allocator = std.testing.allocator;
    const request_str = "POST /v1/search HTTP/1.1\r\nContent-Length: 100000\r\n\r\n";

    var reader = MockReader{
        .data = request_str,
        .chunk_size = 4096,
    };

    const result = readHttpRequest(allocator, &reader, 65536);
    try std.testing.expectError(error.PayloadTooLarge, result);
}

test "readHttpRequest handles request without Content-Length" {
    const allocator = std.testing.allocator;
    const request_str = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";

    var reader = MockReader{
        .data = request_str,
        .chunk_size = 5,
    };

    const result = try readHttpRequest(allocator, &reader, 65536);
    defer if (result) |r| allocator.free(r);

    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings(request_str, result.?);
}

test "readHttpRequest returns 413 when total request exceeds max_size" {
    const allocator = std.testing.allocator;
    const body = "xxxxxxxxxxxxxxxx";
    const request_str = "POST /v1/search HTTP/1.1\r\nContent-Length: 256\r\n\r\n" ++ body;

    var reader = MockReader{
        .data = request_str,
        .chunk_size = 64,
    };

    const result = readHttpRequest(allocator, &reader, 128);
    try std.testing.expectError(error.PayloadTooLarge, result);
}
