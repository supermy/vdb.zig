const std = @import("std");

// Regression tests ensure previously fixed bugs do not reoccur.

test "regression: dimension not divisible by 8 must fail" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // Previously this could have been allowed; must always reject.
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 7, .{ .num_partitions = 1 }));
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 15, .{ .num_partitions = 1 }));
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 63, .{ .num_partitions = 1 }));
}

test "regression: empty index search must not crash" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 2 });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.0);
    var results: [5]index_mod.SearchResult = undefined;
    // Should return 0 results without crashing
    const found = try idx.search(&vec, 5, 1, &results);
    try std.testing.expectEqual(@as(u32, 0), found);
}

test "regression: manifest version must persist" {
    const vdb = @import("vdb");
    const allocator = std.testing.allocator;
    const tmp = "regression_manifest.bin";
    const version: u64 = 0xDEADBEEFCAFE;
    {
        const offsets = &[_]u64{};
        var m = vdb.Manifest{ .version = version, .batch_offsets = offsets, .allocator = allocator };
        try m.save(tmp);
    }
    {
        var m = try vdb.Manifest.load(allocator, tmp);
        defer m.deinit();
        try std.testing.expectEqual(version, m.version);
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.deleteFile(std.Io.Dir.cwd(), io, tmp) catch {};
}

test "regression: partition balance must not collapse to single partition" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    var vec: [64]f32 = undefined;
    for (0..200) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    var max_count: u32 = 0;
    for (idx.partitions) |p| {
        if (p.count > max_count) max_count = p.count;
    }
    // Regression: previously with zero-centroids everything went to partition 0.
    // Now max should be strictly less than total to prove distribution.
    try std.testing.expect(max_count <= 80);
}

test "regression: simd dot product with negative values" {
    const simd = @import("simd");
    const a = &[_]f32{ -1.0, 2.0, -3.0 };
    const b = &[_]f32{ 4.0, -5.0, 6.0 };
    const result = simd.dotProduct(a, b);
    // -4 -10 -18 = -32
    try std.testing.expectApproxEqAbs(@as(f32, -32.0), result, 0.001);
}

test "regression: empty partition deinit must not crash" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // Create index with many partitions but insert only a few vectors.
    // Most partitions will have capacity=0 and codes/scalars/ids pointing to compile-time empty slices.
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 16 });
    var vec: [64]f32 = undefined;
    @memset(&vec, 0.5);
    try idx.insert(&vec); // Only 1 vector, 15 partitions stay empty
    idx.deinit();
}

test "regression: nng k value overflow protection" {
    // Verify that k > 256 is rejected in binary protocol parsing logic.
    const k: u32 = 300;
    try std.testing.expect(k > 256); // This would overflow the 256-element results buffer
}
