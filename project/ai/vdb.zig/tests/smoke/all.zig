const std = @import("std");

/// Smoke tests with verbose logging for quick build verification and debugging.
/// These are lightweight and run fast to catch obvious breakage.
fn smokeLog(comptime msg: []const u8, args: anytype) void {
    std.log.info("[SMOKE] " ++ msg, args);
}

test "smoke: allocator available" {
    smokeLog("Checking allocator...", .{});
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer if (gpa.deinit() == .leak) @panic("leak");
    const allocator = gpa.allocator();
    const buf = try allocator.alloc(u8, 1024);
    defer allocator.free(buf);
    smokeLog("Allocator OK, allocated {d} bytes", .{buf.len});
}

test "smoke: simd backend detected" {
    smokeLog("Detecting SIMD backend...", .{});
    const simd = @import("simd");
    const backend = simd.detectBackend();
    smokeLog("SIMD backend = {}", .{backend});
    // Any backend is acceptable for smoke test
}

test "smoke: index create and insert one vector" {
    smokeLog("Creating index...", .{});
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 2 });
    defer idx.deinit();
    smokeLog("Index created with {d} partitions", .{idx.partitions.len});

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.5);
    try idx.insert(&vec);
    smokeLog("Inserted 1 vector, partition[0] count={d}", .{idx.partitions[0].count});

    var results: [5]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 1, &results);
    smokeLog("Search returned {d} results", .{found});
    try std.testing.expect(found > 0);
}

test "smoke: gpu device init" {
    smokeLog("Initializing GPU device...", .{});
    const gpu = @import("gpu");
    const allocator = std.testing.allocator;
    var dev = try gpu.GpuDevice.init(allocator);
    defer dev.deinit();
    smokeLog("GPU available = {}", .{dev.isAvailable()});
}

test "smoke: search executor smoke" {
    smokeLog("Running search executor smoke...", .{});
    const index_mod = @import("index_ivf_rq");
    const search_mod = @import("search");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 2 });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.1);
    try idx.insert(&vec);

    const executor = search_mod.Executor.init(allocator, &idx);
    var plan = search_mod.QueryPlan{
        .vector_query = try allocator.dupe(f32, &vec),
        .vector_k = 3,
        .nprobe = 1,
        .sql_filter = null,
        .fulltext_query = null,
        .allocator = allocator,
    };
    defer plan.deinit();

    const results = try executor.execute(&plan);
    defer allocator.free(results);
    smokeLog("Executor returned {d} results", .{results.len});
}
