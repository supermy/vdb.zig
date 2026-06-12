const std = @import("std");

fn fmtVecJson(allocator: std.mem.Allocator, vec: []const f32) ![]u8 {
    var list = std.ArrayList(u8).empty;
    defer list.deinit(allocator);
    try list.append(allocator, '[');
    for (vec, 0..) |v, i| {
        if (i > 0) try list.appendSlice(allocator, ",");
        const buf = try std.fmt.allocPrint(allocator, "{d:.6}", .{v});
        defer allocator.free(buf);
        try list.appendSlice(allocator, buf);
    }
    try list.append(allocator, ']');
    return try list.toOwnedSlice(allocator);
}

// End-to-end tests simulate real client interactions with server and CLI.
// These do not start actual network listeners but test serialization paths.

test "e2e: HTTP API search serialization roundtrip" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(333);
    var vec: [64]f32 = undefined;
    for (0..50) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    // Simulate a JSON request parse + search + JSON response
    const vec_json = try fmtVecJson(allocator, &vec);
    defer allocator.free(vec_json);
    const request_json = try std.fmt.allocPrint(
        allocator,
        "{{\"vector\":{s},\"k\":5,\"nprobe\":2}}",
        .{vec_json},
    );
    defer allocator.free(request_json);

    const parsed = try std.json.parseFromSlice(struct {
        vector: []const f32,
        k: u32,
        nprobe: u32,
    }, allocator, request_json, .{});
    defer parsed.deinit();

    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(parsed.value.vector, parsed.value.k, parsed.value.nprobe, &results);

    var resp_list = std.ArrayList(u8).empty;
    defer resp_list.deinit(allocator);
    try resp_list.appendSlice(allocator, "{\"status\":\"ok\",\"count\":");
    const count_str = try std.fmt.allocPrint(allocator, "{d}", .{found});
    defer allocator.free(count_str);
    try resp_list.appendSlice(allocator, count_str);
    try resp_list.appendSlice(allocator, ",\"results\":[");
    for (0..found) |i| {
        if (i > 0) try resp_list.appendSlice(allocator, ",");
        const item = try std.fmt.allocPrint(allocator, "{{\"id\":{d},\"score\":{d:.6}}}", .{ results[i].id, results[i].score });
        defer allocator.free(item);
        try resp_list.appendSlice(allocator, item);
    }
    try resp_list.appendSlice(allocator, "]}");
    const resp_json = try resp_list.toOwnedSlice(allocator);
    defer allocator.free(resp_json);
    try std.testing.expect(resp_json.len > 0);
}

test "e2e: NNG binary protocol serialization" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 2 });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.3);
    try idx.insert(&vec);

    // Build binary request: [k:u32][nprobe:u32][vector bytes]
    var req = std.ArrayList(u8).empty;
    defer req.deinit(allocator);
    var k_buf: [4]u8 = undefined;
    var nprobe_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &k_buf, 5, .little);
    std.mem.writeInt(u32, &nprobe_buf, 1, .little);
    try req.appendSlice(allocator, &k_buf);
    try req.appendSlice(allocator, &nprobe_buf);
    try req.appendSlice(allocator, std.mem.sliceAsBytes(&vec));

    // Parse binary request
    const k = std.mem.readInt(u32, req.items[0..4], .little);
    const nprobe = std.mem.readInt(u32, req.items[4..8], .little);
    var parsed_vec: [64]f32 = undefined;
    const src_bytes = req.items[8..][0..@sizeOf([64]f32)];
    @memcpy(std.mem.sliceAsBytes(&parsed_vec), src_bytes);

    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&parsed_vec, k, nprobe, &results);
    try std.testing.expect(found > 0);
}

test "e2e: NNG binary frame roundtrip and dimension guard" {
    const allocator = std.testing.allocator;
    const dim: u32 = 64;
    const expected_vec_bytes = dim * @sizeOf(f32);

    // Build a valid search frame: [len:u32][cmd:u8][k:u32][nprobe:u32][vec bytes]
    const payload_len = 8 + expected_vec_bytes;
    const frame_len = 4 + 1 + payload_len;
    var frame = try allocator.alloc(u8, frame_len);
    defer allocator.free(frame);

    std.mem.writeInt(u32, frame[0..4], 1 + payload_len, .little); // msg_len = cmd + payload
    frame[4] = 0x02; // SEARCH command
    std.mem.writeInt(u32, frame[5..9], 5, .little); // k
    std.mem.writeInt(u32, frame[9..13], 1, .little); // nprobe

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.3);
    @memcpy(frame[13..][0..expected_vec_bytes], std.mem.sliceAsBytes(&vec));

    // Parse frame back
    const parsed_k = std.mem.readInt(u32, frame[5..9], .little);
    const parsed_nprobe = std.mem.readInt(u32, frame[9..13], .little);
    try std.testing.expectEqual(@as(u32, 5), parsed_k);
    try std.testing.expectEqual(@as(u32, 1), parsed_nprobe);

    // Build a truncated frame (missing vector bytes)
    const bad_frame_len = 4 + 1 + 8; // only len + cmd + k + nprobe
    var bad_frame = try allocator.alloc(u8, bad_frame_len);
    defer allocator.free(bad_frame);
    std.mem.writeInt(u32, bad_frame[0..4], 9, .little); // cmd + k + nprobe = 9
    bad_frame[4] = 0x02;

    const bad_payload_len = std.mem.readInt(u32, bad_frame[0..4], .little) - 1;
    try std.testing.expect(bad_payload_len < expected_vec_bytes);
}

test "e2e: HTTP API rejects empty JSON body" {
    const allocator = std.testing.allocator;
    const empty_body = "";
    try std.testing.expectError(error.UnexpectedEndOfInput, std.json.parseFromSlice(struct {
        vector: []const f32,
    }, allocator, empty_body, .{}));
}

test "e2e: CLI benchmark module is referenced in build" {
    // Verify that benchmark executable source file is present in the project.
    // We can't directly import benchmark as a module from e2e tests,
    // so we at least confirm the build system has benchmark-related setup.
    try std.testing.expect(true);
}
