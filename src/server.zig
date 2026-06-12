const std = @import("std");
const vdb = @import("vdb");
const index_mod = @import("index_ivf_rq");
const search_mod = @import("search");
const gpu = @import("gpu");
const posix = std.posix;

/// OpenAI/Anthropic-compatible HTTP API server for vdb.zig.
/// Uses libevent evhttp for event-driven HTTP serving (replaces POSIX socket + per-thread model).
const PORT = 8080;

// ============================================================
// libevent C bindings
// ============================================================
const event_base = opaque {};
const evhttp = opaque {};
const evhttp_request = opaque {};
const evbuffer = opaque {};
const evkeyvalq = opaque {};

const le = struct {
    pub extern "c" fn event_base_new() ?*event_base;
    pub extern "c" fn event_base_dispatch(base: *event_base) c_int;
    pub extern "c" fn event_base_free(base: *event_base) void;

    pub extern "c" fn evhttp_new(base: *event_base) ?*evhttp;
    pub extern "c" fn evhttp_free(http: *evhttp) void;
    pub extern "c" fn evhttp_bind_socket(http: *evhttp, address: [*c]const u8, port: u16) c_int;
    pub extern "c" fn evhttp_set_gencb(http: *evhttp, cb: ?*const fn (*evhttp_request, ?*anyopaque) callconv(.c) void, arg: ?*anyopaque) void;

    pub extern "c" fn evhttp_request_get_uri(req: *evhttp_request) [*c]const u8;
    pub extern "c" fn evhttp_request_get_command(req: *evhttp_request) c_int;
    pub extern "c" fn evhttp_request_get_input_buffer(req: *evhttp_request) ?*evbuffer;
    pub extern "c" fn evhttp_request_get_output_buffer(req: *evhttp_request) ?*evbuffer;
    pub extern "c" fn evhttp_request_get_output_headers(req: *evhttp_request) ?*evkeyvalq;

    pub extern "c" fn evhttp_send_reply(req: *evhttp_request, code: c_int, reason: [*c]const u8, databuf: ?*evbuffer) void;

    pub extern "c" fn evbuffer_new() ?*evbuffer;
    pub extern "c" fn evbuffer_free(buf: *evbuffer) void;
    pub extern "c" fn evbuffer_add(buf: *evbuffer, data: *const anyopaque, len: usize) c_int;
    pub extern "c" fn evbuffer_pullup(buf: *evbuffer, len: isize) [*c]u8;
    pub extern "c" fn evbuffer_get_length(buf: *evbuffer) usize;

    pub extern "c" fn evhttp_add_header(headers: *evkeyvalq, key: [*c]const u8, value: [*c]const u8) c_int;

    pub const EVHTTP_REQ_GET: c_int = 1;
    pub const EVHTTP_REQ_POST: c_int = 2;
    pub const EVHTTP_REQ_OPTIONS: c_int = 7;
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

/// Server context shared across all requests.
const ServerContext = struct {
    allocator: std.mem.Allocator,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
};

/// Send an HTTP reply via libevent evhttp.
fn evSendReply(req: *evhttp_request, status_code: c_int, reason: [*c]const u8, content_type: []const u8, body: []const u8) void {
    const out_headers = le.evhttp_request_get_output_headers(req) orelse return;
    _ = le.evhttp_add_header(out_headers, "Content-Type", content_type.ptr);
    _ = le.evhttp_add_header(out_headers, "Access-Control-Allow-Origin", "*");

    const out_buf = le.evhttp_request_get_output_buffer(req) orelse return;
    _ = le.evbuffer_add(out_buf, body.ptr, body.len);
    le.evhttp_send_reply(req, status_code, reason, out_buf);
}

fn evSendJson(req: *evhttp_request, body: []const u8) void {
    evSendReply(req, 200, "OK", "application/json", body);
}

fn evSendJsonError(req: *evhttp_request, status_code: c_int, reason: [*c]const u8, body: []const u8) void {
    evSendReply(req, status_code, reason, "application/json", body);
}

fn evSendCors(req: *evhttp_request) void {
    const out_headers = le.evhttp_request_get_output_headers(req) orelse return;
    _ = le.evhttp_add_header(out_headers, "Access-Control-Allow-Origin", "*");
    _ = le.evhttp_add_header(out_headers, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    _ = le.evhttp_add_header(out_headers, "Access-Control-Allow-Headers", "Content-Type");
    const out_buf = le.evhttp_request_get_output_buffer(req) orelse return;
    le.evhttp_send_reply(req, 204, "No Content", out_buf);
}

/// Read request body from evhttp input buffer.
fn readRequestBody(req: *evhttp_request, allocator: std.mem.Allocator) ![]const u8 {
    const in_buf = le.evhttp_request_get_input_buffer(req) orelse return &[_]u8{};
    const len = le.evbuffer_get_length(in_buf);
    if (len == 0) return &[_]u8{};
    const data = le.evbuffer_pullup(in_buf, -1);
    const copy = try allocator.alloc(u8, len);
    @memcpy(copy, data[0..len]);
    return copy;
}

/// libevent evhttp generic callback — routes requests to handlers.
fn httpRequestCallback(req: *evhttp_request, arg: ?*anyopaque) callconv(.c) void {
    const ctx = @as(*ServerContext, @ptrCast(@alignCast(arg.?)));
    const allocator = ctx.allocator;

    const uri_c = le.evhttp_request_get_uri(req);
    const uri = std.mem.sliceTo(uri_c, 0);

    const method = le.evhttp_request_get_command(req);

    const t0 = monoNs();
    var status_code: c_int = 200;

    // Read body for POST requests
    const body = readRequestBody(req, allocator) catch &[_]u8{};
    defer if (body.len > 0) allocator.free(body);

    // Parse path from URI (strip query string)
    var path: []const u8 = uri;
    if (std.mem.indexOfScalar(u8, uri, '?')) |qpos| {
        path = uri[0..qpos];
    }

    // Routing
    if (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html")) {
        evSendReply(req, 200, "OK", "text/html; charset=utf-8", INDEX_HTML);
    } else if (std.mem.eql(u8, path, "/app.js")) {
        evSendReply(req, 200, "OK", "application/javascript", APP_JS);
    } else if (std.mem.eql(u8, path, "/style.css")) {
        evSendReply(req, 200, "OK", "text/css", STYLE_CSS);
    } else if (method == le.EVHTTP_REQ_OPTIONS) {
        evSendCors(req);
        status_code = 204;
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/search")) {
        handleSearch(allocator, req, body, ctx.idx, ctx.rwlock);
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/batch_search")) {
        handleBatchSearch(allocator, req, body, ctx.idx, ctx.rwlock);
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/import")) {
        handleImport(allocator, req, body, ctx.idx, ctx.rwlock, ctx.raw_store);
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/export")) {
        handleExport(allocator, req, ctx.idx);
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/benchmark")) {
        handleBenchmark(allocator, req, body);
    } else if (method == le.EVHTTP_REQ_GET and std.mem.eql(u8, path, "/v1/stats")) {
        handleStats(allocator, req, ctx.idx);
    } else if (method == le.EVHTTP_REQ_POST and std.mem.eql(u8, path, "/v1/recall_test")) {
        handleRecallTest(allocator, req, body, ctx.idx, ctx.rwlock, ctx.raw_store);
    } else if (method == le.EVHTTP_REQ_GET and std.mem.eql(u8, path, "/health")) {
        evSendJson(req, "{\"status\":\"ok\",\"version\":\"0.3.2\"}");
    } else {
        evSendJsonError(req, 404, "Not Found", "{\"status\":\"error\",\"message\":\"Not Found\"}");
        status_code = 404;
    }

    const elapsed_us = (monoNs() - t0) / 1000;
    const srv_log = std.log.scoped(.http);
    const method_str = switch (method) {
        le.EVHTTP_REQ_GET => "GET",
        le.EVHTTP_REQ_POST => "POST",
        le.EVHTTP_REQ_OPTIONS => "OPTIONS",
        else => "UNKNOWN",
    };
    srv_log.info("{s} {s} {d} {d}us", .{ method_str, path, status_code, elapsed_us });
    logDebug("URI: {s}", .{uri});
}

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

    var idx_mutex = std.atomic.Mutex.unlocked;

    // libevent evhttp setup
    const base = le.event_base_new() orelse return error.EventBaseFailed;
    defer le.event_base_free(base);

    const http = le.evhttp_new(base) orelse return error.EvhttpFailed;
    defer le.evhttp_free(http);

    if (le.evhttp_bind_socket(http, "0.0.0.0", PORT) != 0) {
        std.log.err("Bind failed on port {d}", .{PORT});
        return error.BindFailed;
    }

    var ctx = ServerContext{
        .allocator = allocator,
        .idx = &idx,
        .rwlock = &idx_mutex,
        .raw_store = &raw_store,
    };

    le.evhttp_set_gencb(http, httpRequestCallback, &ctx);

    std.log.info("vdb-server (libevent) listening on http://0.0.0.0:{d}", .{PORT});

    // Run event loop (blocks until interrupted)
    _ = le.event_base_dispatch(base);
}

// ── API handlers ──────────────────────────────────────────────────────
// All handlers use evSendJson / evSendJsonError to send responses.

fn handleSearch(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    body: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
) void {
    const parsed = std.json.parseFromSlice(struct {
        vector: []const f32,
        k: ?u32,
        nprobe: ?u32,
    }, allocator, body, .{}) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    const k = @min(parsed.value.k orelse 10, 256);
    const nprobe = parsed.value.nprobe orelse 8;

    if (parsed.value.vector.len != idx.dim) {
        var err_buf: [256]u8 = undefined;
        const err_msg = std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Vector dimension mismatch: expected {d}, got {d}\"}}", .{ idx.dim, parsed.value.vector.len }) catch {
            evSendJson(req, "{\"status\":\"error\",\"message\":\"Dimension mismatch\"}");
            return;
        };
        evSendJson(req, err_msg);
        return;
    }

    var results: [256]index_mod.SearchResult = undefined;
    spinLock(rwlock);
    defer rwlock.unlock();
    const found = idx.search(parsed.value.vector, k, nprobe, &results) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Search failed\"}");
        return;
    };

    // Build JSON response
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);
    json.appendSlice(allocator, "{\"status\":\"ok\",\"count\":") catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"JSON build failed\"}");
        return;
    };
    var count_buf: [16]u8 = undefined;
    const count_str = std.fmt.bufPrint(&count_buf, "{d}", .{found}) catch "0";
    json.appendSlice(allocator, count_str) catch return;
    json.appendSlice(allocator, ",\"results\":[") catch return;
    for (0..found) |i| {
        if (i > 0) json.appendSlice(allocator, ",") catch return;
        var item_buf: [128]u8 = undefined;
        const item = std.fmt.bufPrint(&item_buf, "{{\"id\":{d},\"partition_id\":{d},\"score\":{d:.6}}}", .{ results[i].id, results[i].partition_id, results[i].score }) catch continue;
        json.appendSlice(allocator, item) catch return;
    }
    json.appendSlice(allocator, "]}") catch return;
    evSendJson(req, json.items);
}

fn handleBatchSearch(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    body: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
) void {
    const parsed = std.json.parseFromSlice(struct {
        queries: []const struct { vector: []const f32, k: ?u32 },
        nprobe: ?u32,
    }, allocator, body, .{}) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    const nprobe = parsed.value.nprobe orelse 8;
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);
    json.appendSlice(allocator, "{\"status\":\"ok\",\"results\":[") catch return;

    for (parsed.value.queries, 0..) |q, qi| {
        if (qi > 0) json.appendSlice(allocator, ",") catch return;
        if (q.vector.len != idx.dim) {
            var err_buf: [256]u8 = undefined;
            const err_item = std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Query {d} dimension mismatch: expected {d}, got {d}\"}}", .{ qi, idx.dim, q.vector.len }) catch continue;
            json.appendSlice(allocator, err_item) catch return;
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

        json.appendSlice(allocator, "[") catch return;
        for (0..found) |i| {
            if (i > 0) json.appendSlice(allocator, ",") catch return;
            var item_buf: [128]u8 = undefined;
            const item = std.fmt.bufPrint(&item_buf, "{{\"id\":{d},\"score\":{d:.6}}}", .{ results[i].id, results[i].score }) catch continue;
            json.appendSlice(allocator, item) catch return;
        }
        json.appendSlice(allocator, "]") catch return;
    }
    json.appendSlice(allocator, "]}") catch return;
    evSendJson(req, json.items);
}

fn handleImport(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    body: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) void {
    const parsed = std.json.parseFromSlice(struct {
        vectors: []const []const f32,
    }, allocator, body, .{}) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
        return;
    };
    defer parsed.deinit();

    for (parsed.value.vectors, 0..) |vec, i| {
        if (vec.len != idx.dim) {
            var err_buf: [256]u8 = undefined;
            const err_msg = std.fmt.bufPrint(&err_buf, "{{\"status\":\"error\",\"message\":\"Vector {d} dimension mismatch: expected {d}, got {d}\"}}", .{ i, idx.dim, vec.len }) catch {
                evSendJson(req, "{\"status\":\"error\",\"message\":\"Dimension mismatch\"}");
                return;
            };
            evSendJson(req, err_msg);
            return;
        }
    }

    spinLock(rwlock);
    defer rwlock.unlock();
    idx.batchInsert(parsed.value.vectors) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Insert failed\"}");
        return;
    };
    const imported = parsed.value.vectors.len;

    var raw_added: u32 = 0;
    for (parsed.value.vectors) |vec| {
        raw_store.addVector(allocator, vec) catch break;
        raw_added += 1;
    }

    if (raw_added < imported) {
        var resp_buf: [256]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf, "{{\"status\":\"warning\",\"imported\":{d},\"raw_stored\":{d},\"message\":\"Some vectors failed to store in raw_store\"}}", .{ imported, raw_added }) catch {
            evSendJson(req, "{\"status\":\"warning\",\"message\":\"Partial import\"}");
            return;
        };
        evSendJson(req, resp);
    } else {
        var resp_buf: [128]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"imported\":{d}}}", .{imported}) catch {
            evSendJson(req, "{\"status\":\"ok\"}");
            return;
        };
        evSendJson(req, resp);
    }
}

fn handleExport(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    idx: *index_mod.Index,
) void {
    _ = allocator;
    var resp_buf: [256]u8 = undefined;
    const resp = std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"partitions\":{d},\"dimension\":{d},\"note\":\"Full vector export not yet implemented\"}}", .{ idx.partitions.len, idx.dim }) catch {
        evSendJson(req, "{\"status\":\"ok\",\"note\":\"Export not yet implemented\"}");
        return;
    };
    evSendJson(req, resp);
}

fn handleBenchmark(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    body: []const u8,
) void {
    const parsed = std.json.parseFromSlice(struct {
        dim: u32,
        n: u32,
        query_count: u32,
        k: u32,
        nprobe_values: []const u32,
        fastscan: bool,
        query_bits: u32,
    }, allocator, body, .{}) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
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
        evSendJson(req, "{\"status\":\"error\",\"message\":\"dim must be a multiple of 64\"}");
        return;
    }
    if (n == 0) {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"n must be > 0\"}");
        return;
    }

    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    // Generate dataset
    const dataset_size = @as(usize, n) * @as(usize, dim);
    const dataset = bench_allocator.alloc(f32, dataset_size) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate dataset\"}");
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
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to build index\"}");
        return;
    };
    defer idx.deinit();

    const vec_slices = bench_allocator.alloc([]const f32, n) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate vec_slices\"}");
        return;
    };
    defer bench_allocator.free(vec_slices);
    for (0..n) |i| {
        vec_slices[i] = dataset[i * dim ..][0..dim];
    }
    idx.batchInsert(vec_slices) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to insert vectors\"}");
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
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate query indices\"}");
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
            evSendJson(req, "{\"status\":\"error\",\"message\":\"Brute-force search failed\"}");
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to store exact results\"}");
            return;
        };
    }

    // Benchmark across nprobe values
    var json = std.ArrayList(u8).empty;
    defer json.deinit(allocator);

    var build_buf: [64]u8 = undefined;
    const build_str = std.fmt.bufPrint(&build_buf, "{d}", .{build_ms}) catch "0";
    var mem_buf: [64]u8 = undefined;
    const mem_str = std.fmt.bufPrint(&mem_buf, "{d:.2}", .{mem_per_vec}) catch "0";
    var comp_buf: [64]u8 = undefined;
    const comp_str = std.fmt.bufPrint(&comp_buf, "{d:.2}", .{compression_ratio}) catch "0";

    json.appendSlice(allocator, "{\"status\":\"ok\",\"build_time_ms\":") catch return;
    json.appendSlice(allocator, build_str) catch return;
    json.appendSlice(allocator, ",\"memory_per_vector\":") catch return;
    json.appendSlice(allocator, mem_str) catch return;
    json.appendSlice(allocator, ",\"compression_ratio\":") catch return;
    json.appendSlice(allocator, comp_str) catch return;
    json.appendSlice(allocator, ",\"results\":[") catch return;

    for (parsed.value.nprobe_values, 0..) |nprobe_val, ni| {
        if (nprobe_val > idx.partitions.len) continue;
        if (ni > 0) json.appendSlice(allocator, ",") catch return;

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
        const item = std.fmt.bufPrint(&item_buf, "{{\"nprobe\":{d},\"qps\":{d:.1},\"p50_us\":{d:.1},\"p99_us\":{d:.1},\"recall_at_k\":{d:.4}}}", .{ nprobe_val, qps, @as(f64, @floatFromInt(p50_ns)) / 1000.0, @as(f64, @floatFromInt(p99_ns)) / 1000.0, avg_recall }) catch continue;
        json.appendSlice(allocator, item) catch return;
    }

    json.appendSlice(allocator, "]}") catch return;
    evSendJson(req, json.items);
}

fn handleStats(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    idx: *index_mod.Index,
) void {
    _ = allocator;
    var resp_buf: [512]u8 = undefined;
    const resp = std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"dimension\":{d},\"num_partitions\":{d},\"total_vectors\":{d},\"config\":{{\"num_bits\":{d},\"epsilon_0\":{d:.6},\"refine_sq8\":{s},\"refine_k\":{d},\"fastscan\":{s},\"query_bits\":{d}}}}}", .{
        idx.dim,
        idx.partitions.len,
        idx.next_id.load(.monotonic),
        idx.config.num_bits,
        idx.config.epsilon_0,
        if (idx.config.refine_sq8) "true" else "false",
        idx.config.refine_k,
        if (idx.config.fastscan) "true" else "false",
        idx.config.query_bits,
    }) catch {
        evSendJson(req, "{\"status\":\"ok\",\"dimension\":{d}}");
        return;
    };
    evSendJson(req, resp);
}

fn handleRecallTest(
    allocator: std.mem.Allocator,
    req: *evhttp_request,
    body: []const u8,
    idx: *index_mod.Index,
    rwlock: *std.atomic.Mutex,
    raw_store: *RawVectorStore,
) void {
    const parsed = std.json.parseFromSlice(struct {
        query_count: u32,
        k: u32,
        nprobe: u32,
    }, allocator, body, .{}) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Invalid JSON\"}");
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
        evSendJson(req, "{\"status\":\"error\",\"message\":\"No vectors in index\"}");
        return;
    }

    const actual_nprobe = @min(nprobe, @as(u32, @intCast(idx.partitions.len)));
    const bench_allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    const dataset = bench_allocator.alloc(f32, total_vectors * idx.dim) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate dataset\"}");
        return;
    };
    defer bench_allocator.free(dataset);
    for (0..total_vectors) |i| {
        const vec = raw_store.getVector(i);
        @memcpy(dataset[i * idx.dim ..][0..idx.dim], vec);
    }

    const actual_query_count = @min(query_count, @as(u32, @intCast(total_vectors)));
    const query_indices = bench_allocator.alloc(u32, actual_query_count) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate query indices\"}");
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
            evSendJson(req, "{\"status\":\"error\",\"message\":\"Brute-force search failed\"}");
            return;
        };
        exact_results_list.append(bench_allocator, exact) catch {
            bench_allocator.free(exact);
            evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to store exact results\"}");
            return;
        };
    }

    var latencies = bench_allocator.alloc(u64, actual_query_count) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate latencies\"}");
        return;
    };
    defer bench_allocator.free(latencies);
    var approx_results = bench_allocator.alloc(index_mod.SearchResult, k) catch {
        evSendJson(req, "{\"status\":\"error\",\"message\":\"Failed to allocate results\"}");
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
    const resp = std.fmt.bufPrint(&resp_buf, "{{\"status\":\"ok\",\"recall_at_k\":{d:.4},\"qps\":{d:.1}}}", .{ avg_recall, qps }) catch {
        evSendJson(req, "{\"status\":\"ok\"}");
        return;
    };
    evSendJson(req, resp);
}
