const std = @import("std");

// Integration tests verify module interactions: index + search + simd.

test "integration: end-to-end search with random data" {
    const index_mod = @import("index_ivf_rq");
    const search_mod = @import("search");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 16 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(777);
    var vec: [128]f32 = undefined;
    for (0..5000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    const executor = search_mod.Executor.init(allocator, &idx);
    var plan = search_mod.QueryPlan{
        .vector_query = try allocator.dupe(f32, &vec),
        .vector_k = 10,
        .nprobe = 4,
        .sql_filter = null,
        .fulltext_query = null,
        .allocator = allocator,
    };
    defer plan.deinit();

    const results = try executor.execute(&plan);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
}

test "integration: hybrid vector + fulltext query" {
    const index_mod = @import("index_ivf_rq");
    const search_mod = @import("search");
    const allocator = std.testing.allocator;

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 8 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(888);
    var vec: [64]f32 = undefined;
    for (0..500) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    const executor = search_mod.Executor.init(allocator, &idx);
    var plan = search_mod.QueryPlan{
        .vector_query = try allocator.dupe(f32, &vec),
        .vector_k = 5,
        .nprobe = 2,
        .sql_filter = null,
        .fulltext_query = .{ .query = "example", .top_k = 5 },
        .hybrid_mode = .rrf,
        .allocator = allocator,
    };
    defer plan.deinit();

    const results = try executor.execute(&plan);
    defer allocator.free(results);
    // Should return at least the placeholder fulltext result.
    try std.testing.expect(results.len >= 1);
}

test "integration: gpu batch popcount on large batch" {
    const gpu = @import("gpu");
    const allocator = std.testing.allocator;
    var dev = try gpu.GpuDevice.init(allocator);
    defer dev.deinit();

    const dim_words = 4; // e.g., 256-dim / 64
    const count = 1000;
    const query_code = try allocator.alloc(u64, dim_words);
    defer allocator.free(query_code);
    @memset(query_code, 0xAA);

    const codes = try allocator.alloc(u64, count * dim_words);
    defer allocator.free(codes);
    var rng = std.Random.DefaultPrng.init(999);
    for (codes) |*c| {
        c.* = rng.random().int(u64);
    }

    const scores = try allocator.alloc(f32, count);
    defer allocator.free(scores);
    try dev.batchRabitqPopcount(query_code, codes, scores);

    // All scores should be non-negative
    for (scores) |s| {
        try std.testing.expect(s >= 0.0);
    }
}
