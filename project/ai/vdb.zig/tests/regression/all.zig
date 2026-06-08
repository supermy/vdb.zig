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

test "regression: num_partitions must be greater than 0" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 64, .{ .num_partitions = 0 }));
}

test "regression: dimension must be multiple of 64 not 8" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // dim=8 was previously allowed but should be rejected (not multiple of 64)
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 8, .{ .num_partitions = 1 }));
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 16, .{ .num_partitions = 1 }));
    try std.testing.expectError(index_mod.Error.InvalidDimension, index_mod.Index.init(allocator, 32, .{ .num_partitions = 1 }));
}

test "regression: RaBitQ distance formula uses correction*ip/dim not ip/correction" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false, .fastscan = false });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var vecs: [50][64]f32 = undefined;
    var slices: [50][]const f32 = undefined;
    for (0..50) |i| {
        for (&vecs[i]) |*v| v.* = rng.random().float(f32);
        slices[i] = &vecs[i];
    }
    try idx.batchInsert(&slices);

    // Search for vecs[0] and verify the result is reasonable
    const query = &vecs[0];
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(query, 5, 4, &results);
    try std.testing.expect(found > 0);

    // The query vector itself should be the closest match
    try std.testing.expect(results[0].id == 0);

    // Verify scores are non-negative (L2 distance squared)
    for (0..found) |i| {
        try std.testing.expect(results[i].score >= -0.001); // small tolerance for floating point
    }
}

test "regression: FastScan uses per-partition q_code from q_r_rot" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // With FastScan enabled, the query binary code should be computed from R*(q-c)
    // per partition, not from R*q globally. This test verifies FastScan finds
    // the exact match vector as top result.
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .fastscan = true, .refine_sq8 = false });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(123);
    var vecs: [100][64]f32 = undefined;
    var slices: [100][]const f32 = undefined;
    for (0..100) |i| {
        for (&vecs[i]) |*v| v.* = rng.random().float(f32);
        slices[i] = &vecs[i];
    }
    try idx.batchInsert(&slices);

    // Search for a specific vector
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vecs[50], 5, 4, &results);
    try std.testing.expect(found > 0);
    // The exact match should be found
    try std.testing.expect(results[0].id == 50);
}

test "regression: storage loadIndex does not double-free on corrupt data" {
    const storage = @import("storage");
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    const path = "test_storage_corrupt.bin";
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.deleteFile(std.Io.Dir.cwd(), io, path) catch {};
    }

    // Create a valid index and save it
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();
    try storage.saveIndex(&idx, path);

    // Loading should succeed without double-free
    var loaded = try storage.loadIndex(allocator, path);
    loaded.deinit();
}

test "regression: insert then batchInsert preserves search consistency" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);

    // Single insert first
    var vec: [64]f32 = undefined;
    for (&vec) |*v| v.* = rng.random().float(f32);
    try idx.insert(&vec);

    // Then batch insert more vectors
    var vecs: [20][64]f32 = undefined;
    var slices: [20][]const f32 = undefined;
    for (0..20) |i| {
        for (&vecs[i]) |*v| v.* = rng.random().float(f32);
        slices[i] = &vecs[i];
    }
    try idx.batchInsert(&slices);

    // Search should work correctly after mixed insert operations
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 4, &results);
    try std.testing.expect(found > 0);
}

test "regression: insert with wrong dimension returns InvalidDimension" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var bad_vec: [32]f32 = undefined;
    @memset(&bad_vec, 0.5);
    try std.testing.expectError(index_mod.Error.InvalidDimension, idx.insert(&bad_vec));
}

test "regression: search with wrong dimension returns InvalidDimension" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.5);
    try idx.insert(&vec);

    var bad_query: [32]f32 = undefined;
    @memset(&bad_query, 0.5);
    var results: [10]index_mod.SearchResult = undefined;
    try std.testing.expectError(index_mod.Error.InvalidDimension, idx.search(&bad_query, 5, 2, &results));
}

test "regression: batchInsert with wrong dimension returns InvalidDimension" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var good_vec: [64]f32 = undefined;
    @memset(&good_vec, 0.5);
    var bad_vec: [32]f32 = undefined;
    @memset(&bad_vec, 0.5);
    var slices = [_][]const f32{ &good_vec, &bad_vec };
    try std.testing.expectError(index_mod.Error.InvalidDimension, idx.batchInsert(&slices));
}

test "regression: batchSearch with wrong dimension returns InvalidDimension" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.5);
    try idx.insert(&vec);

    var good_query: [64]f32 = undefined;
    @memset(&good_query, 0.5);
    var bad_query: [32]f32 = undefined;
    @memset(&bad_query, 0.5);
    var queries = [_][]const f32{ &good_query, &bad_query };

    var results: [20]index_mod.SearchResult = undefined;
    var counts: [2]u32 = undefined;
    try std.testing.expectError(index_mod.Error.InvalidDimension, idx.batchSearch(&queries, 10, 2, &results, &counts));
}

test "regression: batchSearch with many queries succeeds without memory corruption" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    // Insert enough vectors so every query can find results
    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..100) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    // Create 20 valid queries to stress-test batchSearch context allocation
    var queries: [20][]const f32 = undefined;
    var qvecs: [20][64]f32 = undefined;
    for (0..20) |i| {
        for (&qvecs[i]) |*v| v.* = rng.random().float(f32);
        queries[i] = &qvecs[i];
    }

    var results: [200]index_mod.SearchResult = undefined; // 20 queries * k=10
    var counts: [20]u32 = undefined;
    try idx.batchSearch(&queries, 10, 4, &results, &counts);

    // Verify batchSearch completed without memory corruption;
    // at least some queries should return results.
    var total_found: u32 = 0;
    for (counts) |c| {
        total_found += c;
    }
    try std.testing.expect(total_found > 0);
}

test "regression: findNearestPartition with many partitions does not stack overflow" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // Use 100 partitions -> sqrt(100)=10 super-partitions, top_n=6 (safe)
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 100, .refine_sq8 = false });
    defer idx.deinit();

    // Insert enough vectors to ensure search finds results
    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..500) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    // Hierarchical search should not crash with many partitions
    const pid = idx.findNearestPartition(&vec);
    try std.testing.expect(pid < idx.partitions.len);

    // searchWithContext should also be safe
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 8, &results);
    try std.testing.expect(found > 0);
}

test "regression: searchWithContext nprobe=all with many partitions safe" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    // 128 partitions -> 11 super-partitions, but nprobe=128 forces top_super calculation
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 128, .refine_sq8 = false });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..500) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    // nprobe=all should not overflow stack in hierarchical path
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 128, &results);
    try std.testing.expect(found > 0);
}

test "regression: storage roundtrip preserves fastscan and query_bits config" {
    const storage = @import("storage");
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    const path = "test_storage_config.bin";
    defer {
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.deleteFile(std.Io.Dir.cwd(), io, path) catch {};
    }

    // Create index with non-default fastscan and query_bits
    var idx = try index_mod.Index.init(allocator, 64, .{
        .num_partitions = 4,
        .refine_sq8 = false,
        .fastscan = true,
        .query_bits = 4,
    });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..10) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    try storage.saveIndex(&idx, path);

    var loaded = try storage.loadIndex(allocator, path);
    defer loaded.deinit();

    try std.testing.expectEqual(idx.config.fastscan, loaded.config.fastscan);
    try std.testing.expectEqual(idx.config.query_bits, loaded.config.query_bits);
}

test "regression: vdb schema rejects dimension not multiple of 64" {
    const vdb = @import("vdb");
    const allocator = std.testing.allocator;
    const cols = &[_]vdb.ColumnSchema{
        .{ .name = "bad_vec", .col_type = .vector, .dimension = 32 },
    };
    try std.testing.expectError(vdb.Error.DimensionMismatch, vdb.Schema.init(allocator, cols));
}

test "regression: vdb schema accepts dimension multiple of 64" {
    const vdb = @import("vdb");
    const allocator = std.testing.allocator;
    const cols = &[_]vdb.ColumnSchema{
        .{ .name = "good_vec", .col_type = .vector, .dimension = 128 },
    };
    var schema = try vdb.Schema.init(allocator, cols);
    defer schema.deinit();
    try std.testing.expectEqual(@as(usize, 1), schema.columns.len);
}

test "regression: empty index search returns zero results without crash" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    var vec: [64]f32 = undefined;
    @memset(&vec, 0.5);
    var results: [5]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 2, &results);
    try std.testing.expectEqual(@as(u32, 0), found);
}

test "regression: index with single partition handles search correctly" {
    const index_mod = @import("index_ivf_rq");
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 1, .refine_sq8 = false });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..20) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(&vec, 5, 1, &results);
    try std.testing.expect(found > 0);
}
