const std = @import("std");
const vdb = @import("vdb");
const index_mod = @import("index_ivf_rq");
const search_mod = @import("search");
const gpu = @import("gpu");
const posix = std.posix;

/// OpenAI/Anthropic-compatible HTTP API server for vdb.zig.
/// Serves the default web test page at GET / and API endpoints at POST /v1/search, etc.
/// Uses POSIX socket directly (borrowed from tsdb.zig pattern).
const PORT = 8080;

// POSIX socket wrapper (macOS / Linux compatible)
const c = struct {
    pub extern "c" fn socket(domain: c_int, sock_type: c_int, protocol: c_int) c_int;
    pub extern "c" fn bind(sockfd: c_int, addr: *const anyopaque, addrlen: c_uint) c_int;
    pub extern "c" fn listen(sockfd: c_int, backlog: c_int) c_int;
    pub extern "c" fn accept(sockfd: c_int, addr: ?*anyopaque, addrlen: ?*c_uint) c_int;
    pub extern "c" fn recv(sockfd: c_int, buf: *anyopaque, len: usize, flags: c_int) isize;
    pub extern "c" fn send(sockfd: c_int, buf: *const anyopaque, len: usize, flags: c_int) isize;
    pub extern "c" fn close(fd: c_int) c_int;
    pub extern "c" fn setsockopt(sockfd: c_int, level: c_int, optname: c_int, optval: *const anyopaque, optlen: c_uint) c_int;

    const AF_INET: c_int = 2;
    const SOCK_STREAM: c_int = 1;
    const SOL_SOCKET: c_int = 1;
    const SO_REUSEADDR: c_int = 2;
    const INADDR_ANY: u32 = 0;

    const sockaddr_in = extern struct {
        sin_len: u8 = @sizeOf(sockaddr_in),
        sin_family: u8 = 2,
        sin_port: u16,
        sin_addr: u32,
        sin_zero: [8]u8 = .{0} ** 8,
    };
};

/// Static web assets embedded at compile time (no runtime filesystem dependency).
const INDEX_HTML = @embedFile("web/index.html");
const APP_JS = @embedFile("web/app.js");
const STYLE_CSS = @embedFile("web/style.css");

/// Debug logging toggle. Set to true to enable verbose request/response logging.
var debug_log: bool = false;

fn logDebug(comptime fmt: []const u8, args: anytype) void {
    if (debug_log) {
        std.log.info("[DEBUG] " ++ fmt, args);
    }
}

/// Spin-lock helper for std.atomic.Mutex (same pattern as thread_pool.zig).
fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

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
    const allocator = std.heap.page_allocator;

    // Enable debug logging via VDB_DEBUG environment variable
    const vdb_debug = std.c.getenv("VDB_DEBUG");
    if (vdb_debug != null) {
        const val = std.mem.sliceTo(vdb_debug.?, 0);
        if (val.len > 0 and val[0] != '0') {
            debug_log = true;
            std.log.info("Debug logging enabled (VDB_DEBUG={s})", .{val});
        }
    }

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

    // POSIX socket setup (same pattern as tsdb.zig)
    const fd = c.socket(c.AF_INET, c.SOCK_STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    defer _ = c.close(fd);

    var reuse: c_int = 1;
    _ = c.setsockopt(fd, c.SOL_SOCKET, c.SO_REUSEADDR, &reuse, @sizeOf(c_int));

    var addr: c.sockaddr_in = .{
        .sin_len = @sizeOf(c.sockaddr_in),
        .sin_family = c.AF_INET,
        .sin_port = std.mem.nativeToBig(u16, PORT),
        .sin_addr = c.INADDR_ANY,
        .sin_zero = .{0} ** 8,
    };

    if (c.bind(fd, @ptrCast(&addr), @sizeOf(c.sockaddr_in)) < 0) {
        std.log.err("Bind failed on port {d}", .{PORT});
        return error.BindFailed;
    }
    if (c.listen(fd, 10) < 0) return error.ListenFailed;

    std.log.info("vdb-server listening on http://0.0.0.0:{d}", .{PORT});

    var idx_mutex = std.atomic.Mutex.unlocked;

    while (true) {
        var client_addr: c.sockaddr_in = undefined;
        var addr_len: c_uint = @sizeOf(c.sockaddr_in);
        const client_fd = c.accept(fd, @ptrCast(&client_addr), &addr_len);
        if (client_fd < 0) continue;

        // Spawn a thread per connection so slow requests (benchmark) don't block others
        const t = std.Thread.spawn(.{}, handleConnection, .{ allocator, client_fd, &idx, &idx_mutex, &raw_store }) catch {
            // Fallback: handle synchronously if thread spawn fails
            handleConnection(allocator, client_fd, &idx, &idx_mutex, &raw_store) catch |err| {
                std.log.err("Connection error: {}", .{err});
            };
            _ = c.close(client_fd);
            continue;
        };
        t.detach();
    }
}

// ── HTTP helpers (same pattern as tsdb.zig) ──────────────────────────

/// Loop-send until all bytes are written or an error occurs.
fn sendAll(client_fd: c_int, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = c.send(client_fd, data.ptr + sent, data.len - sent, 0);
        if (n < 0) return error.SendFailed;
        if (n == 0) return error.ConnectionClosed;
        sent += @as(usize, @intCast(n));
    }
}

fn sendResponse(client_fd: c_int, status: []const u8, content_type: []const u8, body: []const u8) !void {
    var header_buf: [512]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{ status, content_type, body.len });
    try sendAll(client_fd, header);
    try sendAll(client_fd, body);
}

fn sendJson(client_fd: c_int, body: []const u8) !void {
    try sendResponse(client_fd, "200 OK", "application/json", body);
}

fn sendJsonError(client_fd: c_int, status: []const u8, body: []const u8) !void {
    try sendResponse(client_fd, status, "application/json", body);
}

fn sendCorsResponse(client_fd: c_int) !void {
    const headers = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n";
    try sendAll(client_fd, headers);
}

// ── Connection handler (same pattern as tsdb.zig) ────────────────────

fn handleConnection(
    allocator: std.mem.Allocator,
    client_fd: c_int,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) !void {
    defer _ = c.close(client_fd);

    // Allocate 1MB buffer for request (same as tsdb.zig)
    const alloc_buf = try allocator.alloc(u8, 1048576);
    defer allocator.free(alloc_buf);

    // Loop recv until complete HTTP request received
    var total: usize = 0;
    var request_complete = false;
    while (total < alloc_buf.len) {
        const n = c.recv(client_fd, alloc_buf.ptr + total, alloc_buf.len - total, 0);
        if (n <= 0) break;
        total += @as(usize, @intCast(n));

        // Check if we have the complete request
        if (total >= 4) {
            const req_so_far = alloc_buf[0..total];
            if (std.mem.indexOf(u8, req_so_far, "\r\n\r\n")) |body_start| {
                const header = req_so_far[0..body_start];
                const cl_str = "Content-Length: ";
                if (std.mem.indexOf(u8, header, cl_str)) |cl_pos| {
                    const cl_val_start = cl_pos + cl_str.len;
                    const cl_val_end = std.mem.indexOfAnyPos(u8, header, cl_val_start, "\r\n") orelse header.len;
                    const content_length = std.fmt.parseInt(usize, header[cl_val_start..cl_val_end], 10) catch 0;
                    const body_received = total - (body_start + 4);
                    if (body_received >= content_length) {
                        request_complete = true;
                        break;
                    }
                } else {
                    // No Content-Length, header end means request complete
                    request_complete = true;
                    break;
                }
            }
        }
    }
    if (total == 0) return;
    if (!request_complete) {
        try sendJsonError(client_fd, "413 Payload Too Large", "{\"status\":\"error\",\"message\":\"Request too large\"}");
        return;
    }
    const request = alloc_buf[0..total];

    // Extract method and path for logging
    const method_end = std.mem.indexOfScalar(u8, request, ' ') orelse 0;
    const method = request[0..method_end];
    var path: []const u8 = "";
    if (method_end > 0 and method_end + 1 < request.len) {
        const after_method = request[method_end + 1 ..];
        const path_end = std.mem.indexOfScalar(u8, after_method, ' ') orelse after_method.len;
        path = after_method[0..path_end];
    }

    const t0 = monoNs();
    var status_code: []const u8 = "200";

    // Route matching — use extracted path for precise matching
    if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html")) {
        try sendResponse(client_fd, "200 OK", "text/html; charset=utf-8", INDEX_HTML);
    } else if (std.mem.eql(u8, path, "/app.js")) {
        try sendResponse(client_fd, "200 OK", "application/javascript", APP_JS);
    } else if (std.mem.eql(u8, path, "/style.css")) {
        try sendResponse(client_fd, "200 OK", "text/css", STYLE_CSS);
    } else if (std.mem.eql(u8, method, "OPTIONS")) {
        try sendCorsResponse(client_fd);
        status_code = "204";
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/search")) {
        try handleSearch(allocator, client_fd, request, idx, rwlock);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/batch_search")) {
        try handleBatchSearch(allocator, client_fd, request, idx, rwlock);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/import")) {
        try handleImport(allocator, client_fd, request, idx, rwlock, raw_store);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/export")) {
        try handleExport(allocator, client_fd, request, idx);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/benchmark")) {
        try handleBenchmark(allocator, client_fd, request);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/stats")) {
        try handleStats(allocator, client_fd, idx);
    } else if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/recall_test")) {
        try handleRecallTest(allocator, client_fd, request, idx, rwlock, raw_store);
    } else if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        try sendJson(client_fd, "{\"status\":\"ok\",\"version\":\"0.3.0\"}");
    } else {
        try sendJsonError(client_fd, "404 Not Found", "{\"status\":\"error\",\"message\":\"Not Found\"}");
        status_code = "404";
    }

    const elapsed_us = (monoNs() - t0) / 1000;
    const srv_log = std.log.scoped(.http);
    srv_log.info("{s} {s} {s} {d}us", .{ method, path, status_code, elapsed_us });
    logDebug("Full request: {s}", .{request[0..@min(request.len, 200)]});
}

// ── API handlers ──────────────────────────────────────────────────────

fn handleSearch(
    allocator: std.mem.Allocator,
    client_fd: c_int,
    request: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
) !void {
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse request.len;
    const body = request[body_start + 4 ..];

    const parsed = std.json.parseFromSlice(struct {
        vector: []const f32,
        k: ?u32,
        nprobe: ?u32,
    }, allocator, body, .{}) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    const k = @min(parsed.value.k orelse 10, 256);
    const nprobe = parsed.value.nprobe orelse 8;

    if (parsed.value.vector.len != idx.dim) {
        var err_buf: [256]u8 = undefined;
        const err_msg = try std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Vector dimension mismatch: expected {d}, got {d}\"}}", .{ idx.dim, parsed.value.vector.len });
        try sendJson(client_fd, err_msg);
        return;
    }

    var results: [256]index_mod.SearchResult = undefined;
    spinLock(rwlock);
    defer rwlock.unlock();
    const found = idx.search(parsed.value.vector, k, nprobe, &results) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Search failed\"}");
        return;
    };

    // Build JSON response manually (same pattern as tsdb.zig)
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);
    try json.appendSlice(allocator, "{\"status\":\"ok\",\"count\":");
    var count_buf: [16]u8 = undefined;
    const count_str = try std.fmt.bufPrint(&count_buf, "{d}", .{found});
    try json.appendSlice(allocator, count_str);
    try json.appendSlice(allocator, ",\"results\":[");
    for (0..found) |i| {
        if (i > 0) try json.appendSlice(allocator, ",");
        var item_buf: [128]u8 = undefined;
        const item = try std.fmt.bufPrint(&item_buf, "{{\"id\":{d},\"partition_id\":{d},\"score\":{d:.6}}}", .{ results[i].id, results[i].partition_id, results[i].score });
        try json.appendSlice(allocator, item);
    }
    try json.appendSlice(allocator, "]}");
    try sendJson(client_fd, json.items);
}

fn handleBatchSearch(
    allocator: std.mem.Allocator,
    client_fd: c_int,
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    const nprobe = parsed.value.nprobe orelse 8;
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);
    try json.appendSlice(allocator, "{\"status\":\"ok\",\"results\":[");

    for (parsed.value.queries, 0..) |q, qi| {
        if (qi > 0) try json.appendSlice(allocator, ",");
        if (q.vector.len != idx.dim) {
            var err_buf: [256]u8 = undefined;
            const err_item = try std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Query {d} dimension mismatch: expected {d}, got {d}\"}}", .{ qi, idx.dim, q.vector.len });
            try json.appendSlice(allocator, err_item);
            continue;
        }
        const k = @min(q.k orelse 10, 256);
        var results: [256]index_mod.SearchResult = undefined;

        spinLock(rwlock);
        const found = idx.search(q.vector, k, nprobe, &results) catch {
            rwlock.unlock();
            continue;
        };
        rwlock.unlock();

        try json.appendSlice(allocator, "[");
        for (0..found) |i| {
            if (i > 0) try json.appendSlice(allocator, ",");
            var item_buf: [128]u8 = undefined;
            const item = try std.fmt.bufPrint(&item_buf, "{{\"id\":{d},\"score\":{d:.6}}}", .{ results[i].id, results[i].score });
            try json.appendSlice(allocator, item);
        }
        try json.appendSlice(allocator, "]");
    }
    try json.appendSlice(allocator, "]}");
    try sendJson(client_fd, json.items);
}

fn handleImport(
    allocator: std.mem.Allocator,
    client_fd: c_int,
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    for (parsed.value.vectors, 0..) |vec, i| {
        if (vec.len != idx.dim) {
            var err_buf: [256]u8 = undefined;
            const err_msg = try std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Vector {d} dimension mismatch: expected {d}, got {d}\"}}", .{ i, idx.dim, vec.len });
            try sendJson(client_fd, err_msg);
            return;
        }
    }

    spinLock(rwlock);
    defer rwlock.unlock();
    try idx.batchInsert(parsed.value.vectors);
    const imported = parsed.value.vectors.len;

    var raw_added: u32 = 0;
    for (parsed.value.vectors) |vec| {
        raw_store.addVector(allocator, vec) catch break;
        raw_added += 1;
    }

    if (raw_added < imported) {
        var resp_buf: [256]u8 = undefined;
        const resp = try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"warning\",\"imported\":{d},\"raw_stored\":{d},\"message\":\"Some vectors failed to store in raw_store\"}}", .{ imported, raw_added });
        try sendJson(client_fd, resp);
    } else {
        var resp_buf: [128]u8 = undefined;
        const resp = try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"imported\":{d}}}", .{imported});
        try sendJson(client_fd, resp);
    }
}

fn handleExport(
    allocator: std.mem.Allocator,
    client_fd: c_int,
    request: []const u8,
    idx: *index_mod.Index,
) !void {
    _ = request;
    _ = allocator;
    var resp_buf: [256]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"partitions\":{d},\"dimension\":{d},\"note\":\"Full vector export not yet implemented\"}}", .{ idx.partitions.len, idx.dim });
    try sendJson(client_fd, resp);
}

fn handleBenchmark(
    allocator: std.mem.Allocator,
    client_fd: c_int,
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"dim must be a multiple of 64\"}");
        return;
    }
    if (n == 0) {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"n must be > 0\"}");
        return;
    }

    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    // Generate dataset
    const dataset_size = @as(usize, n) * @as(usize, dim);
    const dataset = bench_allocator.alloc(f32, dataset_size) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate dataset\"}");
        return;
    };
    defer bench_allocator.free(dataset);
    for (dataset) |*v| {
        v.* = rng.random().float(f32);
    }

    // Build index
    const num_partitions = @max(4, @min(@as(u32, @intFromFloat(@sqrt(@as(f64, @floatFromInt(n))))), 128));
    const build_start = monoNs();

    var idx = index_mod.Index.init(bench_allocator, dim, .{
        .num_partitions = num_partitions,
        .refine_sq8 = true,
        .refine_k = 20,
        .fastscan = fastscan,
        .query_bits = query_bits,
    }) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to build index\"}");
        return;
    };
    defer idx.deinit();

    const vec_slices = bench_allocator.alloc([]const f32, n) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate vec_slices\"}");
        return;
    };
    defer bench_allocator.free(vec_slices);
    for (0..n) |i| {
        vec_slices[i] = dataset[i * dim ..][0..dim];
    }
    idx.batchInsert(vec_slices) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to insert vectors\"}");
        return;
    };
    const build_ns = monoNs() - build_start;
    const build_ms = @divTrunc(@as(i64, @intCast(build_ns)), 1_000_000);

    const mem_per_vec = computeMemoryPerVector(&idx);
    const original_bytes_per_vec: f64 = @as(f64, @floatFromInt(dim)) * @sizeOf(f32);
    const compression_ratio = if (mem_per_vec > 0) original_bytes_per_vec / mem_per_vec else 0.0;

    // Generate queries
    const actual_query_count = @min(query_count, n);
    const query_indices = bench_allocator.alloc(u32, actual_query_count) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate query indices\"}");
        return;
    };
    defer bench_allocator.free(query_indices);
    for (0..actual_query_count) |i| {
        query_indices[i] = rng.random().intRangeLessThan(u32, 0, @intCast(n));
    }

    // Pre-compute exact top-k for recall
    var exact_results_list = std.ArrayList([]index_mod.SearchResult).empty;
    defer {
        for (exact_results_list.items) |r| bench_allocator.free(r);
        exact_results_list.deinit(bench_allocator);
    }
    for (0..actual_query_count) |qi| {
        const q = dataset[query_indices[qi] * dim ..][0..dim];
        const exact = bruteForceTopK(bench_allocator, dataset, dim, q, k) catch {
            try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Brute-force search failed\"}");
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to store exact results\"}");
            return;
        };
    }

    // Benchmark across nprobe values
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    var build_buf: [64]u8 = undefined;
    const build_str = try std.fmt.bufPrint(&build_buf, "{d}", .{build_ms});
    var mem_buf: [64]u8 = undefined;
    const mem_str = try std.fmt.bufPrint(&mem_buf, "{d:.2}", .{mem_per_vec});
    var comp_buf: [64]u8 = undefined;
    const comp_str = try std.fmt.bufPrint(&comp_buf, "{d:.2}", .{compression_ratio});

    try json.appendSlice(allocator, "{\"status\":\"ok\",\"build_time_ms\":");
    try json.appendSlice(allocator, build_str);
    try json.appendSlice(allocator, ",\"memory_per_vector\":");
    try json.appendSlice(allocator, mem_str);
    try json.appendSlice(allocator, ",\"compression_ratio\":");
    try json.appendSlice(allocator, comp_str);
    try json.appendSlice(allocator, ",\"results\":[");

    for (parsed.value.nprobe_values, 0..) |nprobe_val, ni| {
        if (nprobe_val > idx.partitions.len) continue;
        if (ni > 0) try json.appendSlice(allocator, ",");

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

        var item_buf: [256]u8 = undefined;
        const item = try std.fmt.bufPrint(&item_buf, "{{\"nprobe\":{d},\"qps\":{d:.1},\"p50_us\":{d:.1},\"p99_us\":{d:.1},\"recall_at_k\":{d:.4}}}", .{ nprobe_val, qps, @as(f64, @floatFromInt(p50_ns)) / 1000.0, @as(f64, @floatFromInt(p99_ns)) / 1000.0, avg_recall });
        try json.appendSlice(allocator, item);
    }

    try json.appendSlice(allocator, "]}");
    try sendJson(client_fd, json.items);
}

fn handleStats(
    allocator: std.mem.Allocator,
    client_fd: c_int,
    idx: *index_mod.Index,
) !void {
    _ = allocator;
    var resp_buf: [512]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"dimension\":{d},\"num_partitions\":{d},\"total_vectors\":{d},\"config\":{{\"num_bits\":{d},\"epsilon_0\":{d:.6},\"refine_sq8\":{s},\"refine_k\":{d},\"fastscan\":{s},\"query_bits\":{d}}}}}", .{
        idx.dim,
        idx.partitions.len,
        idx.next_id.load(.monotonic),
        idx.config.num_bits,
        idx.config.epsilon_0,
        if (idx.config.refine_sq8) "true" else "false",
        idx.config.refine_k,
        if (idx.config.fastscan) "true" else "false",
        idx.config.query_bits,
    });
    try sendJson(client_fd, resp);
}

fn handleRecallTest(
    allocator: std.mem.Allocator,
    client_fd: c_int,
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
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
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"No vectors in index\"}");
        return;
    }

    const actual_nprobe = @min(nprobe, @as(u32, @intCast(idx.partitions.len)));
    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    const dataset = bench_allocator.alloc(f32, total_vectors * idx.dim) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate dataset\"}");
        return;
    };
    defer bench_allocator.free(dataset);
    for (0..total_vectors) |i| {
        const vec = raw_store.getVector(i);
        @memcpy(dataset[i * idx.dim ..][0..idx.dim], vec);
    }

    const actual_query_count = @min(query_count, @as(u32, @intCast(total_vectors)));
    const query_indices = bench_allocator.alloc(u32, actual_query_count) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate query indices\"}");
        return;
    };
    defer bench_allocator.free(query_indices);
    for (0..actual_query_count) |i| {
        query_indices[i] = rng.random().intRangeLessThan(u32, 0, @intCast(total_vectors));
    }

    var exact_results_list = std.ArrayList([]index_mod.SearchResult).empty;
    defer {
        for (exact_results_list.items) |r| bench_allocator.free(r);
        exact_results_list.deinit(bench_allocator);
    }
    for (0..actual_query_count) |qi| {
        const q = raw_store.getVector(query_indices[qi]);
        const exact = bruteForceTopK(bench_allocator, dataset, idx.dim, q, k) catch {
            try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Brute-force search failed\"}");
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to store exact results\"}");
            return;
        };
    }

    var latencies = bench_allocator.alloc(u64, actual_query_count) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate latencies\"}");
        return;
    };
    defer bench_allocator.free(latencies);
    var approx_results = bench_allocator.alloc(index_mod.SearchResult, k) catch {
        try sendJson(client_fd, "{\"status\":\"error\",\"message\":\"Failed to allocate results\"}");
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

    var resp_buf: [128]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"recall_at_k\":{d:.4},\"qps\":{d:.1}}}", .{ avg_recall, qps });
    try sendJson(client_fd, resp);
}
