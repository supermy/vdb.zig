const std = @import("std");
const vdb = @import("vdb");
const index_mod = @import("index_ivf_rq");
const search_mod = @import("search");
const gpu = @import("gpu");

/// NNG (Next Generation Node) high-performance interface server.
/// Uses raw TCP with a lightweight binary protocol for minimal latency.
/// Protocol: [4 bytes: message length][1 byte: command][payload]
/// Commands:
///   0x01 = PING
///   0x02 = SEARCH (payload: k:u32, nprobe:u32, dim*f32 query)
///   0x03 = BATCH_SEARCH (payload: n:u32, then n searches back-to-back)
///   0x04 = INSERT (payload: dim*f32 vector)
///   0x05 = IMPORT_JSON (payload: JSON bytes)
///   0x06 = EXPORT_JSON (payload: empty)
const PORT = 9090;

pub const ProtocolError = enum {
    InvalidCommand,
    PayloadTooLarge,
    MalformedRequest,
};

pub fn main() !void {
    // Production server uses page_allocator; use DebugAllocator only in debug builds.
    const allocator = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    // Load or create index
    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 16 });
    defer idx.deinit();

    // Pre-populate demo vectors
    var rng = std.Random.DefaultPrng.init(42);
    var vec: [128]f32 = undefined;
    for (0..1000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    const address = try std.Io.net.IpAddress.parseIp4("0.0.0.0", PORT);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    std.log.info("vdb-nng-server listening on tcp://0.0.0.0:{d}", .{PORT});

    var idx_mutex = std.atomic.Mutex.unlocked;

    while (true) {
        const stream = try server.accept(io);
        const t = std.Thread.spawn(.{}, connectionHandler, .{ allocator, stream, &idx, &idx_mutex }) catch {
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
    mutex: *std.atomic.Mutex,
) void {
    // Each thread must use its own Io instance; global_single_threaded is not safe to share across threads.
    const io = std.Io.Threaded.global_single_threaded.io();
    handleConnection(allocator, io, stream, idx, mutex) catch |err| {
        std.log.err("NNG connection error: {}", .{err});
    };
}

fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    io: std.Io,
    stream: std.Io.net.Stream,
    idx: *index_mod.Index,
    mutex: *std.atomic.Mutex,
) !void {
    defer stream.close(io);

    var reader_buf: [4096]u8 = undefined;
    var reader = stream.reader(io, &reader_buf);

    var writer_buf: [4096]u8 = undefined;
    var writer = stream.writer(io, &writer_buf);

    while (true) {
        // Read 4-byte length
        var len_buf: [4]u8 = undefined;
        const len_read = reader.interface.readSliceShort(&len_buf) catch break;
        if (len_read < 4) break;
        const msg_len = std.mem.readInt(u32, &len_buf, .little);
        if (msg_len > 16 * 1024 * 1024) return error.PayloadTooLarge;
        if (msg_len == 0) return error.MalformedRequest;

        // Read 1-byte command
        var cmd_buf: [1]u8 = undefined;
        const cmd_read = reader.interface.readSliceShort(&cmd_buf) catch break;
        if (cmd_read < 1) break;
        const cmd = cmd_buf[0];

        const payload_len = msg_len - 1;
        const payload = try allocator.alloc(u8, payload_len);
        defer allocator.free(payload);
        const payload_read = reader.interface.readSliceShort(payload) catch break;
        if (payload_read < payload_len) break;

        switch (cmd) {
            0x01 => try handlePing(&writer),
            0x02 => try handleBinarySearch(allocator, &writer, payload, idx, mutex),
            0x03 => try handleBinaryBatchSearch(allocator, &writer, payload, idx, mutex),
            0x04 => try handleBinaryInsert(&writer, payload, idx, mutex),
            0x05 => try handleBinaryImport(allocator, &writer, payload, idx, mutex),
            0x06 => try handleBinaryExport(allocator, &writer, idx),
            else => {
                try writeError(&writer, .InvalidCommand, "Unknown command");
            },
        }
    }
}

fn handlePing(writer: *std.Io.net.Stream.Writer) !void {
    // Response: [4 bytes: 1][1 byte: 0x81 (PONG)][0x00]
    var buf: [5]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 1, .little);
    buf[4] = 0x81;
    try writer.interface.writeAll(&buf);
}

fn handleBinarySearch(
    allocator: std.mem.Allocator,
    writer: *std.Io.net.Stream.Writer,
    payload: []const u8,
    idx: *index_mod.Index,
    mutex: *std.atomic.Mutex,
) !void {
    if (payload.len < 8) {
        try writeError(writer, .MalformedRequest, "Search payload too short");
        return;
    }
    const k = std.mem.readInt(u32, payload[0..4], .little);
    const nprobe = std.mem.readInt(u32, payload[4..8], .little);
    if (k > 256) {
        try writeError(writer, .PayloadTooLarge, "k exceeds maximum of 256");
        return;
    }
    const dim = idx.dim;
    const expected_vec_len = dim * @sizeOf(f32);
    if (payload.len - 8 < expected_vec_len) {
        try writeError(writer, .MalformedRequest, "Vector dimension mismatch");
        return;
    }
    const raw = payload[8..][0..expected_vec_len];
    const vec = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, raw)));
    if (vec.len != dim) {
        try writeError(writer, .MalformedRequest, "Vector dimension mismatch");
        return;
    }

    var results: [256]index_mod.SearchResult = undefined;
    spinLock(mutex);
    defer mutex.unlock();
    const found = idx.search(vec, k, nprobe, &results) catch {
        try writeError(writer, .MalformedRequest, "Search failed");
        return;
    };

    // Response: [4 bytes: 1 + found*12][1 byte: 0x82][count:u32][results...]
    const resp_body_len = 1 + 4 + found * 12;
    var resp = try allocator.alloc(u8, 4 + resp_body_len);
    defer allocator.free(resp);
    std.mem.writeInt(u32, resp[0..4], @intCast(resp_body_len), .little);
    resp[4] = 0x82;
    std.mem.writeInt(u32, resp[5..9], found, .little);
    for (0..found) |i| {
        const off = 9 + i * 12;
        std.mem.writeInt(u32, resp[off..][0..4], results[i].id, .little);
        std.mem.writeInt(u32, resp[off + 4 ..][0..4], results[i].partition_id, .little);
        std.mem.writeInt(u32, resp[off + 8 ..][0..4], @bitCast(results[i].score), .little);
    }
    try writer.interface.writeAll(resp);
}

fn handleBinaryBatchSearch(
    allocator: std.mem.Allocator,
    writer: *std.Io.net.Stream.Writer,
    payload: []const u8,
    idx: *index_mod.Index,
    mutex: *std.atomic.Mutex,
) !void {
    if (payload.len < 4) {
        try writeError(writer, .MalformedRequest, "Batch payload too short");
        return;
    }
    const n = std.mem.readInt(u32, payload[0..4], .little);
    var offset: usize = 4;

    var resp_items = std.ArrayList(u8).empty;
    defer resp_items.deinit(allocator);

    for (0..n) |_| {
        if (payload.len - offset < 8) break;
        const k = std.mem.readInt(u32, payload[offset..][0..4], .little);
        const nprobe = std.mem.readInt(u32, payload[offset + 4 ..][0..4], .little);
        if (k > 256) {
            try writeError(writer, .PayloadTooLarge, "k exceeds maximum of 256");
            return;
        }
        offset += 8;
        const expected_vec_len = idx.dim * @sizeOf(f32);
        if (payload.len - offset < expected_vec_len) break;
        const raw = payload[offset..][0..expected_vec_len];
        const vec = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, raw)));
        if (vec.len != idx.dim) {
            mutex.unlock();
            continue;
        }
        offset += expected_vec_len;

        var results: [256]index_mod.SearchResult = undefined;
        spinLock(mutex);
        const found = idx.search(vec, k, nprobe, &results) catch {
            mutex.unlock();
            continue;
        };
        mutex.unlock();

        var count_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &count_buf, found, .little);
        try resp_items.appendSlice(allocator, &count_buf);
        for (0..found) |i| {
            var id_buf: [4]u8 = undefined;
            var pid_buf: [4]u8 = undefined;
            var score_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &id_buf, results[i].id, .little);
            std.mem.writeInt(u32, &pid_buf, results[i].partition_id, .little);
            std.mem.writeInt(u32, &score_buf, @bitCast(results[i].score), .little);
            try resp_items.appendSlice(allocator, &id_buf);
            try resp_items.appendSlice(allocator, &pid_buf);
            try resp_items.appendSlice(allocator, &score_buf);
        }
    }

    const resp_body_len = 1 + resp_items.items.len;
    var resp = try allocator.alloc(u8, 4 + resp_body_len);
    defer allocator.free(resp);
    std.mem.writeInt(u32, resp[0..4], @intCast(resp_body_len), .little);
    resp[4] = 0x83;
    @memcpy(resp[5..], resp_items.items);
    try writer.interface.writeAll(resp);
}

fn handleBinaryInsert(
    writer: *std.Io.net.Stream.Writer,
    payload: []const u8,
    idx: *index_mod.Index,
    mutex: *std.atomic.Mutex,
) !void {
    const expected_vec_len = idx.dim * @sizeOf(f32);
    if (payload.len < expected_vec_len) {
        try writeError(writer, .MalformedRequest, "Insert payload too short");
        return;
    }
    const raw = payload[0..expected_vec_len];
    const vec = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, raw)));
    spinLock(mutex);
    defer mutex.unlock();
    idx.insert(vec) catch {
        try writeError(writer, .MalformedRequest, "Insert failed");
        return;
    };
    var buf: [5]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 1, .little);
    buf[4] = 0x84;
    try writer.interface.writeAll(&buf);
}

fn handleBinaryImport(
    allocator: std.mem.Allocator,
    writer: *std.Io.net.Stream.Writer,
    payload: []const u8,
    idx: *index_mod.Index,
    mutex: *std.atomic.Mutex,
) !void {
    const parsed = std.json.parseFromSlice(struct {
        vectors: []const []const f32,
    }, allocator, payload, .{}) catch {
        try writeError(writer, .MalformedRequest, "Invalid JSON");
        return;
    };
    defer parsed.deinit();

    spinLock(mutex);
    defer mutex.unlock();
    // Use batchInsert for better cache locality and pre-assigned partition mapping.
    try idx.batchInsert(parsed.value.vectors);
    const imported = parsed.value.vectors.len;

    const msg = try std.fmt.allocPrint(allocator, "{{\"imported\":{d}}}", .{imported});
    defer allocator.free(msg);
    const resp_body_len = 1 + msg.len;
    var resp = try allocator.alloc(u8, 4 + resp_body_len);
    defer allocator.free(resp);
    std.mem.writeInt(u32, resp[0..4], @intCast(resp_body_len), .little);
    resp[4] = 0x85;
    @memcpy(resp[5..], msg);
    try writer.interface.writeAll(resp);
}

fn handleBinaryExport(
    allocator: std.mem.Allocator,
    writer: *std.Io.net.Stream.Writer,
    idx: *index_mod.Index,
) !void {
    var vec_count: u32 = 0;
    for (idx.partitions) |p| {
        vec_count += p.count;
    }
    const msg = try std.fmt.allocPrint(
        allocator,
        "{{\"partitions\":{d},\"dimension\":{d},\"vectors\":{d}}}",
        .{ idx.partitions.len, idx.dim, vec_count },
    );
    defer allocator.free(msg);
    const resp_body_len = 1 + msg.len;
    var resp = try allocator.alloc(u8, 4 + resp_body_len);
    defer allocator.free(resp);
    std.mem.writeInt(u32, resp[0..4], @intCast(resp_body_len), .little);
    resp[4] = 0x86;
    @memcpy(resp[5..], msg);
    try writer.interface.writeAll(resp);
}

fn writeError(writer: *std.Io.net.Stream.Writer, code: ProtocolError, message: []const u8) !void {
    var buf: [256]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, "{{\"error\":\"{s}\",\"code\":{d}}}", .{ message, @intFromEnum(code) });
    const resp_body_len = 1 + msg.len;
    var header: [4]u8 = undefined;
    std.mem.writeInt(u32, &header, @intCast(resp_body_len), .little);
    try writer.interface.writeAll(&header);
    try writer.interface.writeByte(0xFF);
    try writer.interface.writeAll(msg);
}
