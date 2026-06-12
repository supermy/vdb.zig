const std = @import("std");
const index_mod = @import("index_ivf_rq");
const simd = @import("simd");

const DIM: u32 = 768;
const BASE_N: usize = 100;
const LEARN_N: usize = 1000;

fn generateVectors(allocator: std.mem.Allocator, rng: *std.Random.DefaultPrng, n: usize, dim: u32) ![]f32 {
    const data = try allocator.alloc(f32, n * dim);
    for (data) |*v| {
        v.* = rng.random().floatNorm(f32);
    }
    return data;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var rng = std.Random.DefaultPrng.init(42);

    const base_data = try generateVectors(allocator, &rng, BASE_N, DIM);
    defer allocator.free(base_data);
    const learn_data = try generateVectors(allocator, &rng, LEARN_N, DIM);
    defer allocator.free(learn_data);

    const base_slices = try allocator.alloc([]const f32, BASE_N);
    defer allocator.free(base_slices);
    for (0..BASE_N) |i| base_slices[i] = base_data[i * DIM ..][0..DIM];

    const learn_slices = try allocator.alloc([]const f32, LEARN_N);
    defer allocator.free(learn_slices);
    for (0..LEARN_N) |i| learn_slices[i] = learn_data[i * DIM ..][0..DIM];

    var idx = try index_mod.Index.init(allocator, DIM, .{
        .num_partitions = 64,
        .refine_sq8 = false,
        .refine_k = 10,
        .fastscan = false,
        .rotation_seed = 12345,
    });
    defer idx.deinit();

    try idx.batchInsert(learn_slices);
    const base_id_offset = idx.next_id.load(.monotonic);
    try idx.batchInsert(base_slices);

    // Verify rotation matrix orthogonality
    const R = idx.rotation;
    std.debug.print("\n=== Rotation Matrix Check ===\n", .{});
    
    // Check a few rows
    for ([_]usize{0, 1, 2, 767}) |ri| {
        const row = R[ri * DIM ..][0..DIM];
        var norm_sq: f32 = 0.0;
        var max_abs: f32 = 0.0;
        var non_zero_count: usize = 0;
        for (row) |v| {
            norm_sq += v * v;
            if (@abs(v) > max_abs) max_abs = @abs(v);
            if (@abs(v) > 1e-6) non_zero_count += 1;
        }
        std.debug.print("row {d}: norm={d:.4} max_abs={d:.4} non_zero={d}\n", .{ ri, @sqrt(norm_sq), max_abs, non_zero_count });
    }
    
    // Check dot product between row 0 and row 1
    var dot01: f32 = 0.0;
    for (0..DIM) |i| {
        dot01 += R[0 * DIM + i] * R[1 * DIM + i];
    }
    std.debug.print("dot(row0, row1)={d:.6}\n", .{dot01});
    
    // Check row 0 with row 767
    var dot_last: f32 = 0.0;
    for (0..DIM) |i| {
        dot_last += R[0 * DIM + i] * R[767 * DIM + i];
    }
    std.debug.print("dot(row0, row767)={d:.6}\n", .{dot_last});
    
    // Check if R^T * R = I by checking a diagonal element
    var diag0: f32 = 0.0;
    for (0..DIM) |i| {
        diag0 += R[i * DIM + 0] * R[i * DIM + 0];
    }
    std.debug.print("(R^T * R)[0,0]={d:.4}\n", .{diag0});
    
    // Check off-diagonal element (0,1)
    var off01: f32 = 0.0;
    for (0..DIM) |i| {
        off01 += R[i * DIM + 0] * R[i * DIM + 1];
    }
    std.debug.print("(R^T * R)[0,1]={d:.6}\n", .{off01});

    // Find base[0] in partition
    var target_partition: u32 = 0;
    var target_vi: u32 = 0;
    var found_in_partition = false;
    for (idx.partitions, 0..) |*p, pi| {
        for (0..p.count) |vi| {
            if (p.ids[vi] == base_id_offset) {
                target_partition = @intCast(pi);
                target_vi = @intCast(vi);
                found_in_partition = true;
                break;
            }
        }
        if (found_in_partition) break;
    }
    std.debug.print("\nbase[0] in partition={d} vi={d}\n", .{ target_partition, target_vi });

    const p = &idx.partitions[target_partition];

    // Compute residual
    var residual = try allocator.alloc(f32, DIM);
    defer allocator.free(residual);
    for (0..DIM) |i| {
        residual[i] = base_slices[0][i] - p.centroid[i];
    }
    var residual_norm: f32 = 0.0;
    for (residual) |v| {
        residual_norm += v * v;
    }
    residual_norm = @sqrt(residual_norm);
    std.debug.print("residual_norm={d:.4}\n", .{residual_norm});

    // Compute x_rot = R * residual / ||residual||
    const inv = 1.0 / residual_norm;
    var x_rot = try allocator.alloc(f32, DIM);
    defer allocator.free(x_rot);
    for (0..DIM) |i| {
        const row = R[i * DIM ..][0..DIM];
        var sum: f32 = 0.0;
        for (0..DIM) |j| {
            sum += row[j] * residual[j] * inv;
        }
        x_rot[i] = sum;
    }
    
    var x_rot_norm_sq: f32 = 0.0;
    var sum_abs: f32 = 0.0;
    var max_abs: f32 = 0.0;
    var non_zero: usize = 0;
    for (x_rot) |v| {
        x_rot_norm_sq += v * v;
        sum_abs += @abs(v);
        if (@abs(v) > max_abs) max_abs = @abs(v);
        if (@abs(v) > 1e-3) non_zero += 1;
    }
    std.debug.print("||x_rot||={d:.4} sum(|x_rot|)={d:.4} max={d:.4} non_zero>{d}\n", .{ @sqrt(x_rot_norm_sq), sum_abs, max_abs, non_zero });

    // Compute histogram of x_rot values
    var hist: [10]usize = .{0} ** 10;
    for (x_rot) |v| {
        const bin = @min(9, @as(usize, @intFromFloat(@abs(v) * 10)));
        hist[bin] += 1;
    }
    std.debug.print("x_rot histogram (abs value):\n", .{});
    for (hist, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("  [{d:.1}-{d:.1}): {d}\n", .{ @as(f32, @floatFromInt(i)) / 10.0, @as(f32, @floatFromInt(i + 1)) / 10.0, count });
        }
    }

    // Also print v = residual / ||residual|| histogram
    var v_norm_sq: f32 = 0.0;
    var v_sum_abs: f32 = 0.0;
    var v_max: f32 = 0.0;
    for (residual) |v| {
        const vn = v * inv;
        v_norm_sq += vn * vn;
        v_sum_abs += @abs(vn);
        if (@abs(vn) > v_max) v_max = @abs(vn);
    }
    std.debug.print("\n||v||={d:.4} sum(|v|)={d:.4} max={d:.4}\n", .{ @sqrt(v_norm_sq), v_sum_abs, v_max });
    var v_hist: [10]usize = .{0} ** 10;
    for (residual) |v| {
        const vn = @abs(v * inv);
        const bin = @min(9, @as(usize, @intFromFloat(vn * 10)));
        v_hist[bin] += 1;
    }
    std.debug.print("v histogram (abs value):\n", .{});
    for (v_hist, 0..) |count, i| {
        if (count > 0) {
            std.debug.print("  [{d:.1}-{d:.1}): {d}\n", .{ @as(f32, @floatFromInt(i)) / 10.0, @as(f32, @floatFromInt(i + 1)) / 10.0, count });
        }
    }

    // Check dot(R[0], v)
    var dot_r0_v: f32 = 0.0;
    for (0..DIM) |j| {
        dot_r0_v += R[0 * DIM + j] * residual[j] * inv;
    }
    std.debug.print("\ndot(R[0], v)={d:.4} x_rot[0]={d:.4}\n", .{ dot_r0_v, x_rot[0] });

    // Check if R[0] equals base[0] / ||base[0]||
    var base_norm: f32 = 0.0;
    for (base_slices[0]) |v| {
        base_norm += v * v;
    }
    base_norm = @sqrt(base_norm);
    var dot_base_r0: f32 = 0.0;
    for (0..DIM) |j| {
        dot_base_r0 += base_slices[0][j] * R[0 * DIM + j] / base_norm;
    }
    std.debug.print("dot(base[0]/||base[0]||, R[0])={d:.4}\n", .{dot_base_r0});
    
    // Print first 10 elements of R[0] and v
    std.debug.print("R[0][0..10]: ", .{});
    for (0..10) |j| {
        std.debug.print("{d:.4} ", .{R[0 * DIM + j]});
    }
    std.debug.print("\nv[0..10]:    ", .{});
    for (0..10) |j| {
        std.debug.print("{d:.4} ", .{residual[j] * inv});
    }
    std.debug.print("\n", .{});

    // Self-query via search API
    var results: [10]index_mod.SearchResult = undefined;
    const found = idx.search(base_slices[0], 10, 64, &results) catch 0;
    std.debug.print("\nSelf-query found={d}\n", .{found});
    for (0..found) |i| {
        std.debug.print("  [{d}] id={d} score={d:.2}\n", .{ i, results[i].id, results[i].score });
    }
}
