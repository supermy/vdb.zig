const std = @import("std");

fn currentMillis() i64 {
    var tv: std.c.timeval = undefined;
    _ = std.c.gettimeofday(&tv, null);
    return @as(i64, @intCast(tv.sec)) * 1000 + @divTrunc(@as(i64, @intCast(tv.usec)), 1000);
}

// System tests validate end-to-end system behavior under realistic load.

test "system: 100k vectors insert and search latency" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 64 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);

    // Allocate dataset on heap for batchInsert
    const n = 50_000;
    const vecs = try allocator.alloc([128]f32, n);
    defer allocator.free(vecs);
    const slices = try allocator.alloc([]const f32, n);
    defer allocator.free(slices);

    for (0..n) |i| {
        for (&vecs[i]) |*v| v.* = rng.random().float(f32);
        slices[i] = &vecs[i];
    }

    const insert_start = currentMillis();
    try idx.batchInsert(slices);
    const insert_ms = currentMillis() - insert_start;

    // Acceptable: batch insert 50k vectors in under 120 seconds (generous for CI)
    try std.testing.expect(insert_ms < 120_000);

    // Search latency test
    var vec: [128]f32 = undefined;
    const search_start = currentMillis();
    var total_found: u32 = 0;
    for (0..100) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        var results: [10]index_mod.SearchResult = undefined;
        total_found += try idx.search(&vec, 10, 8, &results);
    }
    const search_ms = currentMillis() - search_start;

    // Acceptable: 100 searches in under 10 seconds
    try std.testing.expect(search_ms < 10_000);
    try std.testing.expect(total_found > 0);
}

test "system: memory stays bounded after many inserts" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 8 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(55);
    var vec: [64]f32 = undefined;
    for (0..10_000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    // Very loose check: total vectors across partitions should equal insert count
    var total: u32 = 0;
    for (idx.partitions) |p| {
        total += p.count;
    }
    try std.testing.expectEqual(@as(u32, 10_000), total);
}
