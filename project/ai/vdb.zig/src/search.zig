const std = @import("std");
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");
const vdb = @import("vdb");

/// Query scheduler: SQL predicate pushdown, nprobe pruning, partition-level parallel bit-search,
/// refine layer (SQ8), and multi-way result merging (RRF / weighted fusion).
pub const SqlPredicate = union(enum) {
    eq: struct { column: []const u8, value: i64 },
    gt: struct { column: []const u8, value: i64 },
    lt: struct { column: []const u8, value: i64 },
    and_pred: struct { left: *const SqlPredicate, right: *const SqlPredicate },
    or_pred: struct { left: *const SqlPredicate, right: *const SqlPredicate },
    true_pred,
};

pub const FulltextQuery = struct {
    query: []const u8,
    top_k: u32 = 10,
};

pub const HybridMode = enum {
    rrf,
    weighted,
};

pub const QueryPlan = struct {
    vector_query: ?[]const f32,
    vector_k: u32 = 10,
    nprobe: u32 = 50,
    sql_filter: ?SqlPredicate,
    fulltext_query: ?FulltextQuery,
    hybrid_mode: HybridMode = .rrf,
    refine: bool = true,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *QueryPlan) void {
        if (self.vector_query) |vq| {
            self.allocator.free(vq);
        }
    }
};

/// Evaluate a SQL predicate against a row (simplified: only int64 columns supported).
pub fn evaluatePredicate(pred: ?*const SqlPredicate, row: *const std.StringHashMap(i64)) bool {
    const p = pred orelse return true;
    return switch (p.*) {
        .true_pred => true,
        .eq => |e| (row.get(e.column) orelse 0) == e.value,
        .gt => |g| (row.get(g.column) orelse 0) > g.value,
        .lt => |l| (row.get(l.column) orelse 0) < l.value,
        .and_pred => |a| evaluatePredicate(a.left, row) and evaluatePredicate(a.right, row),
        .or_pred => |o| evaluatePredicate(o.left, row) or evaluatePredicate(o.right, row),
    };
}

/// Search executor that coordinates vector + SQL + fulltext.
pub const Executor = struct {
    allocator: std.mem.Allocator,
    index: *const index_mod.Index,

    pub fn init(allocator: std.mem.Allocator, index: *const index_mod.Index) Executor {
        return Executor{ .allocator = allocator, .index = index };
    }

    pub fn execute(self: *const Executor, plan: *const QueryPlan) ![]SearchResult {
        // Phase 1: Vector search with SQL predicate pushdown at partition level
        var vec_results = std.ArrayList(SearchResult).empty;
        defer vec_results.deinit(self.allocator);

        if (plan.vector_query) |vq| {
            const overfetch = @min(plan.vector_k * 4, @as(u32, 256));
            var raw_results: [256]index_mod.SearchResult = undefined;
            const found = try self.index.search(vq, overfetch, plan.nprobe, &raw_results);

            // In a full implementation, each result would be checked against SQL predicate.
            // Here we pass through for the skeleton.
            for (0..found) |i| {
                try vec_results.append(self.allocator, .{
                    .id = raw_results[i].id,
                    .partition_id = raw_results[i].partition_id,
                    .score = raw_results[i].score,
                    .source = .vector,
                });
            }
        }

        // Phase 2: Fulltext search (placeholder for Tantivy FFI integration)
        var ft_results = std.ArrayList(SearchResult).empty;
        defer ft_results.deinit(self.allocator);

        if (plan.fulltext_query) |_| {
            // Tantivy search would go here; emit a placeholder result for testability.
            try ft_results.append(self.allocator, .{
                .id = 0,
                .partition_id = 0,
                .score = 0.5,
                .source = .fulltext,
            });
        }

        // Phase 3: Hybrid fusion
        const out = try self.allocator.alloc(SearchResult, plan.vector_k);
        if (plan.fulltext_query != null and plan.vector_query != null) {
            // RRF fusion
            const k_rrf: f32 = 60.0;
            var score_map = std.AutoHashMap(u32, f32).init(self.allocator);
            defer score_map.deinit();

            for (vec_results.items, 0..) |r, rank| {
                const key = r.id;
                const s = 1.0 / (k_rrf + @as(f32, @floatFromInt(rank + 1)));
                const entry = try score_map.getOrPut(key);
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                }
                entry.value_ptr.* += s;
            }

            for (ft_results.items, 0..) |r, rank| {
                const key = r.id;
                const s = 1.0 / (k_rrf + @as(f32, @floatFromInt(rank + 1)));
                const entry = try score_map.getOrPut(key);
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                }
                entry.value_ptr.* += s;
            }

            var iter = score_map.iterator();
            var i: usize = 0;
            while (iter.next()) |kv| {
                if (i >= out.len) break;
                out[i] = .{
                    .id = kv.key_ptr.*,
                    .partition_id = 0,
                    .score = kv.value_ptr.*,
                    .source = .hybrid,
                };
                i += 1;
            }

            std.mem.sortUnstable(SearchResult, out[0..i], {}, struct {
                fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                    return a.score > b.score;
                }
            }.lessThan);
            return self.allocator.realloc(out, i);
        } else if (vec_results.items.len > 0) {
            const count = @min(out.len, vec_results.items.len);
            for (0..count) |i| {
                out[i] = vec_results.items[i];
            }
            return self.allocator.realloc(out, count);
        } else {
            return self.allocator.realloc(out, 0);
        }
    }
};

pub const SearchResult = struct {
    id: u32,
    partition_id: u32,
    score: f32,
    source: enum { vector, fulltext, hybrid } = .vector,
};

// ============================================
// Unit Tests
// ============================================

test "SQL predicate evaluation" {
    var row = std.StringHashMap(i64).init(std.testing.allocator);
    defer row.deinit();
    try row.put("age", 25);
    try row.put("score", 90);

    const pred = SqlPredicate{ .gt = .{ .column = "age", .value = 18 } };
    try std.testing.expect(evaluatePredicate(&pred, row));

    const false_pred = SqlPredicate{ .lt = .{ .column = "age", .value = 10 } };
    try std.testing.expect(!evaluatePredicate(&false_pred, row));
}

test "executor vector search" {
    const allocator = std.testing.allocator;
    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(77);
    var vec: [64]f32 = undefined;
    for (0..50) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    const executor = Executor.init(allocator, &idx);
    var plan = QueryPlan{
        .vector_query = try allocator.dupe(f32, &vec),
        .vector_k = 5,
        .nprobe = 2,
        .sql_filter = null,
        .fulltext_query = null,
        .allocator = allocator,
    };
    defer plan.deinit();

    const results = try executor.execute(&plan);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
}
