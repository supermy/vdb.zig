const std = @import("std");

// Acceptance tests verify that features meet user-visible requirements.

test "acceptance: can create index, insert, and search" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 4 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(2025);
    var vec: [128]f32 = undefined;
    for (0..100) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 10, 2, &results);
    try std.testing.expect(found >= 1);
}

test "acceptance: batch search returns results for all queries" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(111);
    var vec: [64]f32 = undefined;
    for (0..200) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    var batch_results: [3][5]index_mod.SearchResult = undefined;
    var batch_found: [3]u32 = undefined;
    for (0..3) |i| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        batch_found[i] = try idx.search(&vec, 5, 2, &batch_results[i]);
        try std.testing.expect(batch_found[i] > 0);
    }
}

test "acceptance: GPU fallback yields correct distances when no GPU present" {
    const gpu = @import("gpu");
    const allocator = std.testing.allocator;
    var dev = try gpu.GpuDevice.init(allocator);
    defer dev.deinit();

    const query_code = &[_]u64{0x00};
    const codes = &[_]u64{ 0x00, 0xFF };
    var scores: [2]f32 = undefined;
    try dev.batchRabitqPopcount(query_code, codes, &scores);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[0], 0.01);
    // query_code[0] = 0x00, codes[1] = 0xFF -> xor = 0xFF -> popcount = 8
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), scores[1], 0.01);
}
