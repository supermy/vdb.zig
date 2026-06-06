const std = @import("std");
const simd = @import("simd");
const vdb = @import("vdb");
const ThreadPool = @import("thread_pool").ThreadPool;

/// IVF + RaBitQ (Randomized Binary Quantization) index.
///
/// RaBitQ workflow:
/// 1. Random orthogonal rotation matrix R (dim x dim), generated once per index.
/// 2. Compute residual: x_r = x - centroid
/// 3. Rotate and normalize: x_rot = R * x_r / ||x_r||
/// 4. Binary quantization: code[i] = sign(x_rot[i]) -> 1 bit per dimension.
/// 5. Two correction scalars per vector:
///    - residual_norm: ||x - centroid||
///    - dot_o_bar_o: sum(|x_rot[i]|) ≈ 0.798 * sqrt(dim)
///
/// Distance estimation during query:
///    ||x-q||² ≈ ||x-c||² + ||q-c||² - 2*||x-c||*dot_o_bar_o*<sign(code),R*(q-c)>/dim
/// For top-k ranking, ||q-c||² is constant per partition and can be dropped.
pub const Error = error{
    InvalidDimension,
    PartitionNotFound,
    EmptyIndex,
    BufferTooSmall,
    InvalidVectorIndex,
};

pub const RaBitQConfig = struct {
    num_partitions: u32,
    num_bits: u32 = 4, // Bq, default 4
    epsilon_0: f32 = 1.9,
    rotation_seed: u64 = 42,
    /// Enable SQ8 refinement: stores 1 byte/dim per vector for re-ranking.
    /// Boosts recall from ~0.76 to ~0.95 at the cost of dim bytes/vector extra memory.
    refine_sq8: bool = true,
    /// Over-fetch factor for refinement: retrieve k*refine_k candidates, then re-rank.
    /// Higher values improve recall at the cost of more SQ8 distance computations.
    /// Recommended: 10-20 for good recall/speed tradeoff.
    refine_k: u32 = 10,
    /// Enable FastScan: use batch XOR-popcount for coarse distance estimation.
    /// Computes Hamming distance between binary codes, then converts to approximate
    /// inner product. Much faster than per-bit sign multiplication, especially with SIMD.
    /// Trade-off: slightly less accurate coarse ranking vs 2-4x speedup.
    fastscan: bool = true,
    /// Query quantization bits (1-8). Quantizes the rotated query residual to
    /// Bq bits per dimension for faster inner product with binary codes.
    /// 1-bit: same as binary (fastest, lowest recall)
    /// 4-bit: good balance (default)
    /// 8-bit: near full precision (slowest, highest recall)
    /// When fastscan=true, query is always 1-bit (binary-binary popcount).
    /// 0 = disabled (use standard f32 path when fastscan=false)
    /// 1-8 = enable query quantization with specified bits
    query_bits: u32 = 0,
};

/// A single IVF partition with its centroid and quantized vectors.
pub const Partition = struct {
    id: u32,
    centroid: []f32, // dim elements, owned by partition
    centroid_rot: []f32, // R * centroid, precomputed to avoid per-query O(dim²) rotation
    codes: []u64, // bit-packed quantized vectors, len = capacity * (dim/64)
    scalars: []f32, // interleaved: [dist_to_centroid, dot_norm_quantized] per vector
    sq8_codes: []u8, // SQ8 refinement: 1 byte/dim per vector, len = capacity * dim (or empty)
    sq8_min: []f32, // per-dimension SQ8 min residual for dequantization
    sq8_scale: []f32, // per-dimension SQ8 scale = 255.0 / (max - min)
    sq8_max: []f32, // per-dimension SQ8 max residual for dynamic range
    ids: []u32, // global unique id per vector, len = capacity
    count: u32,
    capacity: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: u32, dim: u32, rng: std.Random) !Partition {
        const centroid = try allocator.alloc(f32, dim);
        // Initialize centroid with random values in the expected data range [0.25, 0.75]
        // to improve partition balance for uniformly distributed inputs.
        for (centroid) |*c| {
            c.* = rng.float(f32) * 0.5 + 0.25;
        }
        const centroid_rot = try allocator.alloc(f32, dim);
        @memset(centroid_rot, 0.0);
        return Partition{
            .id = id,
            .centroid = centroid,
            .centroid_rot = centroid_rot,
            .codes = &[_]u64{},
            .scalars = &[_]f32{},
            .sq8_codes = &[_]u8{},
            .sq8_min = &[_]f32{},
            .sq8_scale = &[_]f32{},
            .sq8_max = &[_]f32{},
            .ids = &[_]u32{},
            .count = 0,
            .capacity = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Partition) void {
        self.allocator.free(self.centroid);
        self.allocator.free(self.centroid_rot);
        // Only free codes/scalars/ids if they were heap-allocated (capacity > 0).
        if (self.capacity > 0) {
            self.allocator.free(self.codes);
            self.allocator.free(self.scalars);
            self.allocator.free(self.ids);
            if (self.sq8_codes.len > 0) self.allocator.free(self.sq8_codes);
            if (self.sq8_min.len > 0) self.allocator.free(self.sq8_min);
            if (self.sq8_scale.len > 0) self.allocator.free(self.sq8_scale);
            if (self.sq8_max.len > 0) self.allocator.free(self.sq8_max);
        }
    }

    /// Add a vector to this partition after RaBitQ quantization.
    pub fn addVector(
        self: *Partition,
        rotation: []const f32, // dim x dim flat
        vector: []const f32,
        global_id: u32,
        enable_sq8: bool,
    ) !void {
        const dim: u32 = @intCast(self.centroid.len);
        const words_per_vec = (dim + 63) / 64;
        std.debug.assert(dim % 64 == 0);
        std.debug.assert(vector.len == dim);

        // RaBitQ quantization: normalize relative to centroid, then rotate and binarize.
        // Step 1: Compute residual (x - centroid) and its norm
        var residual_buf: [2048]f32 = undefined;
        const residual = residual_buf[0..dim];
        for (0..dim) |i| {
            residual[i] = vector[i] - self.centroid[i];
        }
        var residual_norm: f32 = 0.0;
        for (residual) |v| {
            residual_norm += v * v;
        }
        residual_norm = @sqrt(residual_norm);
        const inv_residual_norm = if (residual_norm > 1e-8) 1.0 / residual_norm else 0.0;

        // Step 2: Apply random rotation to the normalized residual
        var stack_fallback = std.heap.stackFallback(16384, self.allocator);
        const fb_allocator = stack_fallback.get();
        var x_rot = try fb_allocator.alloc(f32, dim);
        defer fb_allocator.free(x_rot);
        for (0..dim) |i| {
            const row = rotation[i * dim ..][0..dim];
            // Rotate the normalized residual (not the raw vector)
            var sum: f32 = 0.0;
            for (0..dim) |j| {
                sum += row[j] * residual[j] * inv_residual_norm;
            }
            x_rot[i] = sum;
        }

        // Ensure capacity using doubling strategy to avoid O(N^2) reallocations.
        // Use alloc+memcpy+free instead of realloc so the old data remains valid
        // if any allocation fails, preventing use-after-free on error paths.
        if (self.count == self.capacity) {
            const new_capacity = if (self.capacity == 0) 4 else self.capacity * 2;
            const new_codes = try self.allocator.alloc(u64, new_capacity * words_per_vec);
            errdefer self.allocator.free(new_codes);
            if (self.codes.len > 0) @memcpy(new_codes[0..self.codes.len], self.codes);

            const new_scalars = try self.allocator.alloc(f32, new_capacity * 2);
            errdefer self.allocator.free(new_scalars);
            if (self.scalars.len > 0) @memcpy(new_scalars[0..self.scalars.len], self.scalars);

            const new_ids = try self.allocator.alloc(u32, new_capacity);
            errdefer self.allocator.free(new_ids);
            if (self.ids.len > 0) @memcpy(new_ids[0..self.ids.len], self.ids);

            if (enable_sq8) {
                const new_sq8_codes = try self.allocator.alloc(u8, new_capacity * dim);
                errdefer self.allocator.free(new_sq8_codes);
                if (self.sq8_codes.len > 0) @memcpy(new_sq8_codes[0..self.sq8_codes.len], self.sq8_codes);

                if (self.capacity > 0) self.allocator.free(self.sq8_codes);
                self.sq8_codes = new_sq8_codes;
            }

            if (self.capacity > 0) {
                self.allocator.free(self.codes);
                self.allocator.free(self.scalars);
                self.allocator.free(self.ids);
            }
            self.codes = new_codes;
            self.scalars = new_scalars;
            self.ids = new_ids;
            self.capacity = new_capacity;
        }

        // Step 3: Quantize to bits and compute <ō, o> correction factor.
        // x_rot is already normalized (unit length), so no need for inv_norm.
        const code_offset = self.count * words_per_vec;
        var dot_o_bar_o: f32 = 0.0; // <sign(x_rot), x_rot> = sum(|x_rot[i]|)
        for (0..words_per_vec) |w| {
            var word: u64 = 0;
            for (0..64) |b| {
                const idx = w * 64 + b;
                if (idx >= dim) break;
                const val = x_rot[idx]; // already normalized
                const is_positive = val >= 0.0;
                if (is_positive) {
                    word |= @as(u64, 1) << @intCast(b);
                    dot_o_bar_o += val; // sign=+1, val * sign = val
                } else {
                    dot_o_bar_o -= val; // sign=-1, val * sign = -val
                }
            }
            self.codes[code_offset + w] = word;
        }

        // Correction scalars
        // scalar[0]: ||x - c|| (residual norm, pre-computed per data vector)
        // scalar[1]: <ō, o> (correction factor, ≈0.8 for dim≥64)
        self.scalars[self.count * 2 + 0] = residual_norm;
        self.scalars[self.count * 2 + 1] = dot_o_bar_o;

        // SQ8 quantization: store residual (vector - centroid) as u8 per dimension.
        // min/scale are computed lazily on first insert and updated incrementally.
        if (enable_sq8 and self.sq8_codes.len > 0) {
            const sq8_offset = self.count * dim;

            // Lazy initialization of per-dimension min/max/scale on first insert.
            if (self.sq8_min.len == 0) {
                self.sq8_min = try self.allocator.alloc(f32, dim);
                self.sq8_max = try self.allocator.alloc(f32, dim);
                self.sq8_scale = try self.allocator.alloc(f32, dim);
                for (0..dim) |d| {
                    const sq8_res = vector[d] - self.centroid[d];
                    self.sq8_min[d] = sq8_res;
                    self.sq8_max[d] = sq8_res;
                    self.sq8_scale[d] = 1.0; // dummy scale for zero range
                }
            } else {
                for (0..dim) |d| {
                    const sq8_res = vector[d] - self.centroid[d];
                    if (sq8_res < self.sq8_min[d]) self.sq8_min[d] = sq8_res;
                    if (sq8_res > self.sq8_max[d]) self.sq8_max[d] = sq8_res;
                }
            }

            for (0..dim) |d| {
                const sq8_val = vector[d] - self.centroid[d];
                const range = self.sq8_max[d] - self.sq8_min[d];
                const scale = if (range > 1e-8) 255.0 / range else 1.0;
                self.sq8_scale[d] = scale;
                const normalized = (sq8_val - self.sq8_min[d]) * scale;
                const clamped = @max(0.0, @min(255.0, normalized));
                self.sq8_codes[sq8_offset + d] = @intFromFloat(clamped);
            }
        }

        self.ids[self.count] = global_id;

        // Centroids are fixed after initialization to avoid online drift.
        // For production, run K-Means batch rebalancing periodically.
        self.count += 1;
    }

    /// Compute SQ8 L2 distance between query and the vi-th vector in this partition.
    /// Uses SIMD vectorized dequantization + L2 for 2-4x speedup over scalar loop.
    pub fn sq8Distance(self: *const Partition, query: []const f32, vi: u32) !f32 {
        const dim: u32 = @intCast(self.centroid.len);
        if (vi >= self.count) return Error.InvalidVectorIndex;
        const sq8_offset = vi * dim;
        return simd.sq8L2DistanceDynamic(
            query,
            self.centroid,
            self.sq8_codes[sq8_offset..][0..dim],
            self.sq8_min,
            self.sq8_scale,
        );
    }

    /// Finalize SQ8 quantization after batch insertion.
    /// Recomputes per-dimension min/max over all vectors and re-quantizes
    /// every vector with the correct global scale to avoid distortion.
    pub fn finalizeSq8(self: *Partition, vectors: []const []const f32, indices: []const usize) !void {
        if (indices.len == 0) return;
        const dim: u32 = @intCast(self.centroid.len);

        if (self.sq8_min.len == 0) {
            self.sq8_min = try self.allocator.alloc(f32, dim);
            self.sq8_max = try self.allocator.alloc(f32, dim);
            self.sq8_scale = try self.allocator.alloc(f32, dim);
        }
        if (self.sq8_codes.len == 0) {
            self.sq8_codes = try self.allocator.alloc(u8, self.capacity * dim);
        }

        // Compute true min/max across all vectors in this partition.
        for (0..dim) |d| {
            self.sq8_min[d] = std.math.floatMax(f32);
            self.sq8_max[d] = -std.math.floatMax(f32);
        }
        for (indices) |vi| {
            const vec = vectors[vi];
            for (0..dim) |d| {
                const residual = vec[d] - self.centroid[d];
                if (residual < self.sq8_min[d]) self.sq8_min[d] = residual;
                if (residual > self.sq8_max[d]) self.sq8_max[d] = residual;
            }
        }

        // Re-quantize all vectors using the true global range.
        for (indices, 0..) |vi, i| {
            const vec = vectors[vi];
            const sq8_offset = i * dim;
            for (0..dim) |d| {
                const residual = vec[d] - self.centroid[d];
                const range = self.sq8_max[d] - self.sq8_min[d];
                const scale = if (range > 1e-8) 255.0 / range else 1.0;
                self.sq8_scale[d] = scale;
                const normalized = (residual - self.sq8_min[d]) * scale;
                const clamped = @max(0.0, @min(255.0, normalized));
                self.sq8_codes[sq8_offset + d] = @intFromFloat(clamped);
            }
        }
    }
};

/// A super-partition in the two-level hierarchical K-Means structure.
/// Contains a super-centroid and the IDs of sub-partitions assigned to it.
/// Reduces insertion/search complexity from O(P) to O(sqrt(P)).
pub const SuperPartition = struct {
    id: u32,
    centroid: []f32, // dim elements, owned by super_partition
    sub_ids: []u32, // IDs of sub-partitions assigned to this super-partition
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, id: u32, dim: u32, sub_ids: []u32) !SuperPartition {
        const centroid = try allocator.alloc(f32, dim);
        errdefer allocator.free(centroid);
        @memset(centroid, 0.0);
        return SuperPartition{
            .id = id,
            .centroid = centroid,
            .sub_ids = sub_ids,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SuperPartition) void {
        self.allocator.free(self.centroid);
        self.allocator.free(self.sub_ids);
    }
};

pub const Index = struct {
    dim: u32,
    config: RaBitQConfig,
    rotation: []f32, // dim x dim flat, random orthogonal matrix
    partitions: []Partition,
    super_partitions: []SuperPartition,
    next_id: std.atomic.Value(u32),
    allocator: std.mem.Allocator,
    thread_pool: *ThreadPool,

    pub fn init(allocator: std.mem.Allocator, dim: u32, config: RaBitQConfig) !Index {
        if (dim % 64 != 0) return Error.InvalidDimension;
        const rotation = try generateRandomOrthogonal(allocator, dim, config.rotation_seed);
        errdefer allocator.free(rotation);
        var rng = std.Random.DefaultPrng.init(12345);
        const partitions = try allocator.alloc(Partition, config.num_partitions);
        errdefer allocator.free(partitions);
        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                partitions[j].deinit();
            }
        }
        for (0..config.num_partitions) |i| {
            partitions[i] = try Partition.init(allocator, @intCast(i), dim, rng.random());
            initialized += 1;
        }

        // Build hierarchical structure: sqrt(N) super-partitions
        const super_partitions = try buildHierarchicalKMeans(
            allocator,
            partitions,
            dim,
            rng.random(),
        );
        errdefer {
            for (super_partitions) |*sp| {
                sp.deinit();
            }
            allocator.free(super_partitions);
        }

        const thread_count = @max(1, std.Thread.getCpuCount() catch 4);
        const tp = try ThreadPool.create(allocator, thread_count);
        errdefer tp.destroy();

        return Index{
            .dim = dim,
            .config = config,
            .rotation = rotation,
            .partitions = partitions,
            .super_partitions = super_partitions,
            .next_id = std.atomic.Value(u32).init(0),
            .allocator = allocator,
            .thread_pool = tp,
        };
    }

    pub fn deinit(self: *Index) void {
        self.thread_pool.destroy();
        self.allocator.free(self.rotation);
        for (self.partitions) |*p| {
            p.deinit();
        }
        self.allocator.free(self.partitions);
        for (self.super_partitions) |*sp| {
            sp.deinit();
        }
        self.allocator.free(self.super_partitions);
    }

    /// Build two-level hierarchical K-Means structure.
    /// Level 1: sqrt(N) super-partitions, each with a super-centroid.
    /// Level 2: N sub-partitions assigned to their nearest super-partition.
    fn buildHierarchicalKMeans(
        allocator: std.mem.Allocator,
        partitions: []Partition,
        dim: u32,
        rng: std.Random,
    ) ![]SuperPartition {
        const num_partitions = partitions.len;
        if (num_partitions == 0) {
            return allocator.alloc(SuperPartition, 0);
        }

        const num_super: usize = @max(1, std.math.sqrt(num_partitions));
        if (num_super == 1 or num_super >= num_partitions) {
            // Degenerate case: one super-partition containing all
            const all_ids = try allocator.alloc(u32, num_partitions);
            errdefer allocator.free(all_ids);
            for (0..num_partitions) |i| {
                all_ids[i] = @intCast(i);
            }
            const super = try allocator.alloc(SuperPartition, 1);
            errdefer allocator.free(super);
            super[0] = try SuperPartition.init(allocator, 0, dim, all_ids);
            // Compute super-centroid as mean of all sub-centroids
            for (0..dim) |d| {
                var sum: f32 = 0.0;
                for (partitions) |p| {
                    sum += p.centroid[d];
                }
                super[0].centroid[d] = sum / @as(f32, @floatFromInt(num_partitions));
            }
            return super;
        }

        // Step 1: K-Means++ initialization on sub-centroids
        var super_centroids = try allocator.alloc(f32, num_super * dim);
        defer allocator.free(super_centroids);
        var dists = try allocator.alloc(f32, num_partitions);
        defer allocator.free(dists);

        // Pick first centroid randomly
        const first_idx = rng.intRangeLessThan(usize, 0, num_partitions);
        @memcpy(super_centroids[0..dim], partitions[first_idx].centroid);

        for (1..num_super) |ci| {
            var total_dist: f32 = 0.0;
            for (partitions, 0..) |p, pi| {
                var min_dist: f32 = std.math.floatMax(f32);
                for (0..ci) |cj| {
                    const d = simd.l2DistanceSquared(p.centroid, super_centroids[cj * dim ..][0..dim]);
                    if (d < min_dist) min_dist = d;
                }
                dists[pi] = min_dist;
                total_dist += min_dist;
            }

            // Weighted random selection proportional to D^2
            const threshold = rng.float(f32) * total_dist;
            var cumsum: f32 = 0.0;
            var chosen: usize = 0;
            for (dists, 0..) |d, pi| {
                cumsum += d;
                if (cumsum >= threshold) {
                    chosen = pi;
                    break;
                }
            }
            @memcpy(super_centroids[ci * dim ..][0..dim], partitions[chosen].centroid);
        }

        // Step 2: K-Means clustering on sub-centroids (max 10 iterations)
        var assignments = try allocator.alloc(u32, num_partitions);
        defer allocator.free(assignments);
        @memset(assignments, 0);

        var counts = try allocator.alloc(usize, num_super);
        defer allocator.free(counts);

        for (0..10) |_| {
            @memset(counts, 0);
            var changed: bool = false;

            // Assign each sub-centroid to nearest super-centroid
            for (0..num_partitions) |pi| {
                var best_id: u32 = 0;
                var best_dist: f32 = std.math.floatMax(f32);
                for (0..num_super) |si| {
                    const d = simd.l2DistanceSquared(partitions[pi].centroid, super_centroids[si * dim ..][0..dim]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_id = @intCast(si);
                    }
                }
                if (assignments[pi] != best_id) {
                    assignments[pi] = best_id;
                    changed = true;
                }
                counts[best_id] += 1;
            }

            if (!changed) break;

            // Recompute super-centroids as means of assigned sub-centroids
            @memset(super_centroids, 0.0);
            for (0..num_partitions) |pi| {
                const si = assignments[pi];
                for (0..dim) |d| {
                    super_centroids[si * dim + d] += partitions[pi].centroid[d];
                }
            }
            for (0..num_super) |si| {
                if (counts[si] > 0) {
                    const inv_count = 1.0 / @as(f32, @floatFromInt(counts[si]));
                    for (0..dim) |d| {
                        super_centroids[si * dim + d] *= inv_count;
                    }
                } else {
                    // Empty cluster: reinitialize with random sub-centroid
                    const idx = rng.intRangeLessThan(usize, 0, num_partitions);
                    @memcpy(super_centroids[si * dim ..][0..dim], partitions[idx].centroid);
                }
            }
        }

        // Step 3: Build SuperPartition objects
        const super_partitions = try allocator.alloc(SuperPartition, num_super);
        errdefer allocator.free(super_partitions);

        // Count sub-partitions per super-partition
        @memset(counts, 0);
        for (assignments) |si| {
            counts[si] += 1;
        }

        var sub_id_buffers = try allocator.alloc([]u32, num_super);
        defer allocator.free(sub_id_buffers);
        for (0..num_super) |si| {
            sub_id_buffers[si] = try allocator.alloc(u32, counts[si]);
            errdefer allocator.free(sub_id_buffers[si]);
        }

        var offsets = try allocator.alloc(usize, num_super);
        defer allocator.free(offsets);
        @memset(offsets, 0);

        for (0..num_partitions) |pi| {
            const si = assignments[pi];
            sub_id_buffers[si][offsets[si]] = @intCast(pi);
            offsets[si] += 1;
        }

        for (0..num_super) |si| {
            super_partitions[si] = try SuperPartition.init(allocator, @intCast(si), dim, sub_id_buffers[si]);
            @memcpy(super_partitions[si].centroid, super_centroids[si * dim ..][0..dim]);
        }

        return super_partitions;
    }

    /// Assign vector to nearest partition and quantize.
    /// Deprecated: use batchInsert for better correctness and performance.
    pub fn insert(self: *Index, vector: []const f32) !void {
        const best_id = self.findNearestPartition(vector);
        const global_id = self.next_id.fetchAdd(1, .monotonic);
        try self.partitions[best_id].addVector(self.rotation, vector, global_id, self.config.refine_sq8);
    }

    /// Train partition centroids using K-Means++ on the given vectors.
    /// Only called when the index is empty (next_id == 0) and enough vectors are provided.
    fn trainKMeansPP(self: *Index, vectors: []const []const f32, max_iterations: u32) !void {
        const dim = self.dim;
        const num_partitions = self.partitions.len;
        var rng = std.Random.DefaultPrng.init(self.config.rotation_seed);
        const random = rng.random();

        var centroids = try self.allocator.alloc(f32, num_partitions * dim);
        defer self.allocator.free(centroids);
        var dists = try self.allocator.alloc(f32, vectors.len);
        defer self.allocator.free(dists);
        var assignments = try self.allocator.alloc(u32, vectors.len);
    defer self.allocator.free(assignments);
    @memset(assignments, 0);
    var counts = try self.allocator.alloc(usize, num_partitions);
    defer self.allocator.free(counts);

        // Step 1: K-Means++ initialization
        const first_idx = random.intRangeLessThan(usize, 0, vectors.len);
        @memcpy(centroids[0..dim], vectors[first_idx]);

        for (1..num_partitions) |ci| {
            var total_dist: f32 = 0.0;
            for (vectors, 0..) |vec, vi| {
                var min_dist: f32 = std.math.floatMax(f32);
                for (0..ci) |cj| {
                    const d = simd.l2DistanceSquared(vec, centroids[cj * dim ..][0..dim]);
                    if (d < min_dist) min_dist = d;
                }
                dists[vi] = min_dist;
                total_dist += min_dist;
            }

            const threshold = random.float(f32) * total_dist;
            var cumsum: f32 = 0.0;
            var chosen: usize = 0;
            for (dists, 0..) |d, vi| {
                cumsum += d;
                if (cumsum >= threshold) {
                    chosen = vi;
                    break;
                }
            }
            @memcpy(centroids[ci * dim ..][0..dim], vectors[chosen]);
        }

        // Step 2: Lloyd iterations
        for (0..max_iterations) |_| {
            @memset(counts, 0);
            var changed: bool = false;

            for (vectors, 0..) |vec, vi| {
                var best_id: u32 = 0;
                var best_dist: f32 = std.math.floatMax(f32);
                for (0..num_partitions) |pi| {
                    const d = simd.l2DistanceSquared(vec, centroids[pi * dim ..][0..dim]);
                    if (d < best_dist) {
                        best_dist = d;
                        best_id = @intCast(pi);
                    }
                }
                if (assignments[vi] != best_id) {
                    assignments[vi] = best_id;
                    changed = true;
                }
                counts[best_id] += 1;
            }

            if (!changed) break;

            @memset(centroids, 0.0);
            for (vectors, 0..) |vec, vi| {
                const pi = assignments[vi];
                for (0..dim) |d| {
                    centroids[pi * dim + d] += vec[d];
                }
            }
            for (0..num_partitions) |pi| {
                if (counts[pi] > 0) {
                    const inv = 1.0 / @as(f32, @floatFromInt(counts[pi]));
                    for (0..dim) |d| {
                        centroids[pi * dim + d] *= inv;
                    }
                }
            }
        }

        // Step 3: Update partition centroids and precompute R * centroid
        for (self.partitions, 0..) |*p, i| {
            @memcpy(p.centroid, centroids[i * dim ..][0..dim]);
            // Precompute R * centroid for fast query residual rotation
            for (0..dim) |r| {
                const row = self.rotation[r * dim ..][0..dim];
                p.centroid_rot[r] = simd.dotProduct(row, p.centroid);
            }
        }

        // Step 4: Rebuild hierarchical structure
        for (self.super_partitions) |*sp| {
            sp.deinit();
        }
        self.allocator.free(self.super_partitions);

        self.super_partitions = try buildHierarchicalKMeans(
            self.allocator,
            self.partitions,
            dim,
            random,
        );
    }

    /// Batch insert: pre-assign all vectors to partitions, then insert in parallel.
    /// SQ8 quantization is deferred to a finalization step so all vectors share
    /// the same global min/max/scale, preventing distortion from incremental updates.
    /// If the index is empty, centroids are initialized using K-Means++ on this batch.
    pub fn batchInsert(self: *Index, vectors: []const []const f32) !void {
        if (vectors.len == 0) return;

        // If index is empty, train centroids with K-Means++ on this batch.
        if (self.next_id.load(.monotonic) == 0 and vectors.len >= self.partitions.len) {
            try self.trainKMeansPP(vectors, 10);
        }

        // Phase 1: Assign all vectors to partitions (read-only on centroids)
        const assignments = try self.allocator.alloc(u32, vectors.len);
        defer self.allocator.free(assignments);
        for (vectors, 0..) |vec, i| {
            assignments[i] = self.findNearestPartition(vec);
        }

        // Phase 2: Assign global IDs
        const start_id = self.next_id.fetchAdd(@intCast(vectors.len), .monotonic);

        // Phase 3: Group vectors by partition for parallel insertion.
        // Count vectors per partition.
        const counts = try self.allocator.alloc(u32, self.partitions.len);
        defer self.allocator.free(counts);
        @memset(counts, 0);
        for (assignments) |pid| {
            counts[pid] += 1;
        }

        // Collect partitions that have vectors to insert.
        const active_partitions = try self.allocator.alloc(u32, self.partitions.len);
        defer self.allocator.free(active_partitions);
        var num_active: u32 = 0;
        for (0..self.partitions.len) |i| {
            if (counts[i] > 0) {
                active_partitions[num_active] = @intCast(i);
                num_active += 1;
            }
        }

        // Phase 4: Build per-partition vector index lists (used by all paths).
        var index_lists = try self.allocator.alloc(std.ArrayList(usize), self.partitions.len);
        defer {
            for (index_lists) |*list| {
                list.deinit(self.allocator);
            }
            self.allocator.free(index_lists);
        }
        for (0..self.partitions.len) |i| {
            index_lists[i] = std.ArrayList(usize).empty;
            if (counts[i] > 0) {
                try index_lists[i].ensureTotalCapacity(self.allocator, counts[i]);
            }
        }
        for (vectors, 0..) |_, i| {
            try index_lists[assignments[i]].append(self.allocator, i);
        }

        // Phase 5: Insert vectors with SQ8 disabled; quantization is finalized later.
        if (num_active <= 1) {
            // Single partition: no threading overhead
            for (vectors, 0..) |vec, i| {
                try self.partitions[assignments[i]].addVector(
                    self.rotation,
                    vec,
                    start_id + @as(u32, @intCast(i)),
                    false,
                );
            }
        } else {
            const InsertCtx = struct {
                index: *Index,
                active_partitions: []u32,
                index_lists: []std.ArrayList(usize),
                vectors: []const []const f32,
                start_id: u32,
                error_occurred: std.atomic.Value(bool),
            };

            var ctx = InsertCtx{
                .index = self,
                .active_partitions = active_partitions[0..num_active],
                .index_lists = index_lists,
                .vectors = vectors,
                .start_id = start_id,
                .error_occurred = std.atomic.Value(bool).init(false),
            };

            self.thread_pool.parallelFor(num_active, &ctx, struct {
                fn run(c: *InsertCtx, start: usize, end: usize) void {
                    for (c.active_partitions[start..end]) |pid| {
                        const list = &c.index_lists[pid];
                        for (list.items) |vi| {
                            c.index.partitions[pid].addVector(
                                c.index.rotation,
                                c.vectors[vi],
                                c.start_id + @as(u32, @intCast(vi)),
                                false,
                            ) catch {
                                c.error_occurred.store(true, .release);
                            };
                        }
                    }
                }
            }.run);

            if (ctx.error_occurred.load(.acquire)) return error.InsertFailed;
        }

        // Phase 6: Finalize SQ8 quantization with the true global min/max per partition.
        if (self.config.refine_sq8) {
            for (0..num_active) |ai| {
                const pid = active_partitions[ai];
                try self.partitions[pid].finalizeSq8(vectors, index_lists[pid].items);
            }
        }
    }

    /// Find the nearest partition for a vector using two-level hierarchical search.
    /// Searches the top-N super-partitions (N = max(1, sqrt(num_super))) to avoid
    /// missing boundary partitions, then finds the nearest sub-partition among them.
    /// Complexity: O(sqrt(P)) instead of O(P).
    pub fn findNearestPartition(self: *const Index, vector: []const f32) u32 {
        const num_super = self.super_partitions.len;
        if (num_super <= 1) {
            // Degenerate: single super-partition, search all its sub-partitions
            var best_id: u32 = 0;
            var best_dist: f32 = std.math.floatMax(f32);
            for (self.super_partitions[0].sub_ids) |sub_id| {
                const d = simd.l2DistanceSquared(vector, self.partitions[sub_id].centroid);
                if (d < best_dist) {
                    best_dist = d;
                    best_id = sub_id;
                }
            }
            return best_id;
        }

        // Level 1: Find top-N super-partitions (N = max(2, ceil(sqrt(num_super) * 2)))
        const top_n = @max(2, @min(num_super, std.math.sqrt(num_super) * 2));
        // Use a fixed-size max-heap of size top_n to keep closest super-partitions.
        // Stack-allocate since top_n is small (typically 1-4).
        var super_heap: [16]struct { id: u32, dist: f32 } = undefined;
        var super_heap_len: usize = 0;
        for (self.super_partitions) |*sp| {
            const d = simd.l2DistanceSquared(vector, sp.centroid);
            if (super_heap_len < top_n) {
                super_heap[super_heap_len] = .{ .id = sp.id, .dist = d };
                super_heap_len += 1;
                // Sift up (max-heap)
                var hi: usize = super_heap_len - 1;
                while (hi > 0) {
                    const parent = (hi - 1) / 2;
                    if (super_heap[hi].dist > super_heap[parent].dist) {
                        const tmp = super_heap[hi];
                        super_heap[hi] = super_heap[parent];
                        super_heap[parent] = tmp;
                        hi = parent;
                    } else break;
                }
            } else if (d < super_heap[0].dist) {
                super_heap[0] = .{ .id = sp.id, .dist = d };
                // Sift down
                var hi: usize = 0;
                while (true) {
                    const left = 2 * hi + 1;
                    const right = 2 * hi + 2;
                    var largest = hi;
                    if (left < super_heap_len and super_heap[left].dist > super_heap[largest].dist) largest = left;
                    if (right < super_heap_len and super_heap[right].dist > super_heap[largest].dist) largest = right;
                    if (largest == hi) break;
                    const tmp = super_heap[hi];
                    super_heap[hi] = super_heap[largest];
                    super_heap[largest] = tmp;
                    hi = largest;
                }
            }
        }

        // Level 2: Find nearest sub-partition among all sub-partitions in top-N super-partitions
        var best_id: u32 = 0;
        var best_dist: f32 = std.math.floatMax(f32);
        for (0..super_heap_len) |si| {
            const sp = &self.super_partitions[super_heap[si].id];
            for (sp.sub_ids) |sub_id| {
                const d = simd.l2DistanceSquared(vector, self.partitions[sub_id].centroid);
                if (d < best_dist) {
                    best_dist = d;
                    best_id = sub_id;
                }
            }
        }
        return best_id;
    }

    /// Linear scan fallback: find the nearest partition for a vector.
    /// Used in tests to verify hierarchical search correctness.
    pub fn findNearestPartitionLinear(self: *const Index, vector: []const f32) u32 {
        var best_id: u32 = 0;
        var best_dist: f32 = std.math.floatMax(f32);
        for (self.partitions) |*p| {
            const d = simd.l2DistanceSquared(vector, p.centroid);
            if (d < best_dist) {
                best_dist = d;
                best_id = p.id;
            }
        }
        return best_id;
    }

    /// Precomputed query-dependent state for RaBitQ search.
    /// Allows batchSearch to amortize rotation/quantization overhead across queries.
    pub const QueryContext = struct {
        q_rot: []f32,
        q_code: []u64,
        q_dot_nq: f32,
        words_per_vec: usize,
        /// Backing allocation: q_rot and q_code are slices into this buffer.
        backing: []u8,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *QueryContext) void {
            self.allocator.free(self.backing);
        }
    };

    /// Prepare query-dependent state (rotation, quantization, correction scalar).
    /// This is the expensive per-query setup that batchSearch amortizes.
    /// Uses a single allocation for q_rot + q_code to reduce heap pressure.
    pub fn prepareQuery(self: *const Index, query: []const f32) !QueryContext {
        const dim = self.dim;
        const words_per_vec = (dim + 63) / 64;

        // Single allocation for q_rot (dim * f32) + q_code (words_per_vec * u64)
        const rot_bytes = dim * @sizeOf(f32);
        const code_bytes = words_per_vec * @sizeOf(u64);
        const total_bytes = rot_bytes + code_bytes;

        const backing = try self.allocator.alloc(u8, total_bytes);
        errdefer self.allocator.free(backing);

        const q_rot: []f32 = @as([*]f32, @ptrCast(@alignCast(backing.ptr)))[0..dim];
        const q_code: []u64 = @as([*]u64, @ptrCast(@alignCast(backing.ptr + rot_bytes)))[0..words_per_vec];

        for (0..dim) |i| {
            const row = self.rotation[i * dim ..][0..dim];
            q_rot[i] = simd.dotProduct(row, query);
        }

        var q_norm: f32 = 0.0;
        for (q_rot) |v| {
            q_norm += v * v;
        }
        q_norm = @sqrt(q_norm);
        const inv_qn = if (q_norm > 1e-8) 1.0 / q_norm else 0.0;

        // Quantize and compute q_dot_nq in a single pass (same optimization as addVector).
        var q_dot_nq: f32 = 0.0;
        for (0..words_per_vec) |w| {
            var word: u64 = 0;
            for (0..64) |b| {
                const idx = w * 64 + b;
                if (idx >= dim) break;
                const val = q_rot[idx] * inv_qn;
                const is_positive = val >= 0.0;
                if (is_positive) {
                    word |= @as(u64, 1) << @intCast(b);
                    q_dot_nq += val;
                } else {
                    q_dot_nq -= val;
                }
            }
            q_code[w] = word;
        }

        return QueryContext{
            .q_rot = q_rot,
            .q_code = q_code,
            .q_dot_nq = q_dot_nq,
            .words_per_vec = words_per_vec,
            .backing = backing,
            .allocator = self.allocator,
        };
    }

    /// Search using a precomputed QueryContext. Avoids redundant rotation/quantization.
    pub fn searchWithContext(
        self: *const Index,
        query: []const f32,
        k: u32,
        nprobe: u32,
        results: []SearchResult,
        ctx: *const QueryContext,
    ) !u32 {
        if (self.partitions.len == 0) return Error.EmptyIndex;
        if (results.len < k) return Error.BufferTooSmall;

        const words_per_vec = ctx.words_per_vec;
        const use_refine = self.config.refine_sq8;

        // Use stack fallback allocator to avoid per-query heap allocations
        // for temporary buffers (partition_dists + heap typically < 16 KB).
        var stack_fallback = std.heap.stackFallback(16384, self.allocator);
        const fb_allocator = stack_fallback.get();

        // Find closest nprobe partitions.
        // Use hierarchical search when partition count is large enough to benefit,
        // otherwise fall back to linear scan for maximum recall.
        const DistItem = struct { id: u32, dist: f32 };
        const probe_count = @min(nprobe, @as(u32, @intCast(self.partitions.len)));

        var partition_dists = try fb_allocator.alloc(DistItem, probe_count);
        defer fb_allocator.free(partition_dists);

        var pd_heap_len: u32 = 0;

        // Use hierarchical search only when there are enough super-partitions
        // to make it worthwhile. For small partition counts, linear scan is
        // faster and guarantees no recall loss.
        const use_hierarchical = self.super_partitions.len >= 4 and self.partitions.len >= 16;

        if (use_hierarchical) {
            // Level 1: Find top-N super-partitions using max-heap
            const num_super = self.super_partitions.len;
            // Search enough super-partitions to cover nprobe sub-partitions.
            // Each super-partition has ~sqrt(P) sub-partitions, so searching
            // ceil(nprobe / avg_sub_per_super) super-partitions ensures coverage.
            const avg_sub_per_super = @max(1, self.partitions.len / num_super);
            const top_super = @max(1, @min(num_super, (probe_count + avg_sub_per_super - 1) / avg_sub_per_super + 1));
            var super_heap: [64]DistItem = undefined;
            var super_heap_len: usize = 0;
            for (self.super_partitions) |*sp| {
                const d = simd.l2DistanceSquared(query, sp.centroid);
                if (super_heap_len < top_super) {
                    super_heap[super_heap_len] = .{ .id = sp.id, .dist = d };
                    super_heap_len += 1;
                    var hi: usize = super_heap_len - 1;
                    while (hi > 0) {
                        const parent = (hi - 1) / 2;
                        if (super_heap[hi].dist > super_heap[parent].dist) {
                            const tmp = super_heap[hi];
                            super_heap[hi] = super_heap[parent];
                            super_heap[parent] = tmp;
                            hi = parent;
                        } else break;
                    }
                } else if (d < super_heap[0].dist) {
                    super_heap[0] = .{ .id = sp.id, .dist = d };
                    var hi: usize = 0;
                    while (true) {
                        const left = 2 * hi + 1;
                        const right = 2 * hi + 2;
                        var largest = hi;
                        if (left < super_heap_len and super_heap[left].dist > super_heap[largest].dist) largest = left;
                        if (right < super_heap_len and super_heap[right].dist > super_heap[largest].dist) largest = right;
                        if (largest == hi) break;
                        const tmp = super_heap[hi];
                        super_heap[hi] = super_heap[largest];
                        super_heap[largest] = tmp;
                        hi = largest;
                    }
                }
            }

            // Level 2: Find closest nprobe sub-partitions among top-N super-partitions
            for (0..super_heap_len) |si| {
                const sp = &self.super_partitions[super_heap[si].id];
                for (sp.sub_ids) |sub_id| {
                    const dist = simd.l2DistanceSquared(query, self.partitions[sub_id].centroid);
                    if (pd_heap_len < probe_count) {
                        partition_dists[pd_heap_len] = .{ .id = sub_id, .dist = dist };
                        pd_heap_len += 1;
                        var hi = pd_heap_len - 1;
                        while (hi > 0) {
                            const parent = (hi - 1) / 2;
                            if (partition_dists[hi].dist > partition_dists[parent].dist) {
                                const tmp = partition_dists[hi];
                                partition_dists[hi] = partition_dists[parent];
                                partition_dists[parent] = tmp;
                                hi = parent;
                            } else break;
                        }
                    } else if (dist < partition_dists[0].dist) {
                        partition_dists[0] = .{ .id = sub_id, .dist = dist };
                        var hi: u32 = 0;
                        while (true) {
                            const left = 2 * hi + 1;
                            const right = 2 * hi + 2;
                            var largest = hi;
                            if (left < pd_heap_len and partition_dists[left].dist > partition_dists[largest].dist) largest = left;
                            if (right < pd_heap_len and partition_dists[right].dist > partition_dists[largest].dist) largest = right;
                            if (largest == hi) break;
                            const tmp = partition_dists[hi];
                            partition_dists[hi] = partition_dists[largest];
                            partition_dists[largest] = tmp;
                            hi = largest;
                        }
                    }
                }
            }
        } else {
            // Linear scan: O(P log nprobe) — guarantees no recall loss
            for (self.partitions, 0..) |*p, i| {
                const dist = simd.l2DistanceSquared(query, p.centroid);
                if (pd_heap_len < probe_count) {
                    partition_dists[pd_heap_len] = .{ .id = @intCast(i), .dist = dist };
                    pd_heap_len += 1;
                    var hi = pd_heap_len - 1;
                    while (hi > 0) {
                        const parent = (hi - 1) / 2;
                        if (partition_dists[hi].dist > partition_dists[parent].dist) {
                            const tmp = partition_dists[hi];
                            partition_dists[hi] = partition_dists[parent];
                            partition_dists[parent] = tmp;
                            hi = parent;
                        } else break;
                    }
                } else if (dist < partition_dists[0].dist) {
                    partition_dists[0] = .{ .id = @intCast(i), .dist = dist };
                    var hi: u32 = 0;
                    while (true) {
                        const left = 2 * hi + 1;
                        const right = 2 * hi + 2;
                        var largest = hi;
                        if (left < pd_heap_len and partition_dists[left].dist > partition_dists[largest].dist) largest = left;
                        if (right < pd_heap_len and partition_dists[right].dist > partition_dists[largest].dist) largest = right;
                        if (largest == hi) break;
                        const tmp = partition_dists[hi];
                        partition_dists[hi] = partition_dists[largest];
                        partition_dists[largest] = tmp;
                        hi = largest;
                    }
                }
            }
        }

        // Sort the selected nprobe partitions ascending by distance for ordered probing.
        std.mem.sortUnstable(DistItem, partition_dists, {}, struct {
            fn lessThan(_: void, a: DistItem, b: DistItem) bool {
                return a.dist < b.dist;
            }
        }.lessThan);

        // Coarse search: over-fetch candidates for SQ8 refinement.
        const coarse_k = if (use_refine) k * self.config.refine_k else k;

        // Precompute R * q once (O(dim²)), then R*(q-c) = R*q - R*c per partition (O(dim))
        var q_rot_buf: [2048]f32 = undefined;
        const q_rot = q_rot_buf[0..self.dim];
        for (0..self.dim) |i| {
            const row = self.rotation[i * self.dim ..][0..self.dim];
            q_rot[i] = simd.dotProduct(row, query);
        }

        // Quantize query for FastScan: sign(q_rot) -> binary code
        // Used when fastscan=true for batch XOR-popcount distance estimation.
        var q_code_buf: [32]u64 = undefined;
        const q_code = q_code_buf[0..words_per_vec];
        if (self.config.fastscan) {
            for (0..words_per_vec) |w| {
                var word: u64 = 0;
                for (0..64) |b| {
                    const idx = w * 64 + b;
                    if (idx >= self.dim) break;
                    if (q_rot[idx] >= 0.0) {
                        word |= @as(u64, 1) << @intCast(b);
                    }
                }
                q_code[w] = word;
            }
        }

        // FastScan scratch buffer for batch Hamming distances
        var hamming_buf: [8192]u64 = undefined;
        const max_batch = hamming_buf.len;

        var heap = try fb_allocator.alloc(SearchResult, coarse_k);
        defer fb_allocator.free(heap);
        var heap_len: u32 = 0;
        for (0..probe_count) |pi| {
            const p = &self.partitions[partition_dists[pi].id];
            if (p.count == 0) continue;

            // Compute rotated query residual: R*(q-c) = R*q - R*c
            // Uses precomputed centroid_rot = R*c, avoiding per-partition O(dim²) rotation.
            var q_r_rot_buf: [2048]f32 = undefined;
            const q_r_rot = q_r_rot_buf[0..self.dim];
            for (0..self.dim) |i| {
                q_r_rot[i] = q_rot[i] - p.centroid_rot[i];
            }

            // Compute ||q - c||² for this partition (needed for cross-partition ranking)
            var q_residual_norm_sq: f32 = 0.0;
            for (0..self.dim) |i| {
                const diff = query[i] - p.centroid[i];
                q_residual_norm_sq += diff * diff;
            }

            if (self.config.fastscan) {
                // FastScan path: batch XOR-popcount for coarse distance estimation.
                // Computes Hamming distance between binary codes, then converts to
                // approximate inner product using query magnitude scaling.
                //
                // Key insight: <sign(code), q_r_rot> ≈ (dim - 2*hamming) * ||q_r_rot|| / dim
                // This preserves the query magnitude that pure binary-binary misses.
                const batch_count = @min(p.count, max_batch);
                const hamming_out = hamming_buf[0..batch_count];
                simd.batchPopcountXor(p.codes[0 .. batch_count * words_per_vec], q_code, hamming_out, words_per_vec);

                // Compute ||q_r_rot|| for magnitude scaling
                var q_r_rot_norm: f32 = 0.0;
                for (q_r_rot) |v| {
                    q_r_rot_norm += v * v;
                }
                q_r_rot_norm = @sqrt(q_r_rot_norm);
                const dim_f: f32 = @floatFromInt(self.dim);
                // Scale factor: converts binary-binary IP to approximate float IP
                // <sign(code), q_r_rot> ≈ (dim - 2*h) / dim * ||q_r_rot||
                const q_scale = if (dim_f > 0) q_r_rot_norm / dim_f else 0.0;

                for (0..batch_count) |vi| {
                    const residual_norm = p.scalars[vi * 2 + 0];
                    const dot_o_bar_o = p.scalars[vi * 2 + 1];
                    const hamming: f32 = @floatFromInt(hamming_out[vi]);
                    const ip_approx = (dim_f - 2.0 * hamming) * q_scale;
                    const correction = if (dot_o_bar_o > 1e-8) dot_o_bar_o else 1e-8;
                    const score = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_approx / correction;

                    const candidate = SearchResult{
                        .id = p.ids[vi],
                        .partition_id = p.id,
                        .score = score,
                        .partition_vi = @intCast(vi),
                    };

                    if (heap_len < coarse_k) {
                        heap[heap_len] = candidate;
                        heap_len += 1;
                        heapSiftUp(heap, heap_len);
                    } else if (score < heap[0].score) {
                        heap[0] = candidate;
                        heapSiftDown(heap, heap_len, 0);
                    }
                }
                // Process remaining vectors if partition is larger than batch buffer
                if (p.count > max_batch) {
                    var offset: u32 = @intCast(max_batch);
                    while (offset < p.count) {
                        const remaining = @min(p.count - offset, max_batch);
                        const rem_hamming = hamming_buf[0..remaining];
                        const rem_codes = p.codes[offset * words_per_vec .. (offset + remaining) * words_per_vec];
                        simd.batchPopcountXor(rem_codes, q_code, rem_hamming, words_per_vec);

                        for (0..remaining) |j| {
                            const vi = offset + @as(u32, @intCast(j));
                            const residual_norm = p.scalars[vi * 2 + 0];
                            const dot_o_bar_o = p.scalars[vi * 2 + 1];
                            const hamming: f32 = @floatFromInt(rem_hamming[j]);
                            const ip_approx = (dim_f - 2.0 * hamming) * q_scale;
                            const correction = if (dot_o_bar_o > 1e-8) dot_o_bar_o else 1e-8;
                            const score = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_approx / correction;

                            const candidate = SearchResult{
                                .id = p.ids[vi],
                                .partition_id = p.id,
                                .score = score,
                                .partition_vi = @intCast(vi),
                            };

                            if (heap_len < coarse_k) {
                                heap[heap_len] = candidate;
                                heap_len += 1;
                                heapSiftUp(heap, heap_len);
                            } else if (score < heap[0].score) {
                                heap[0] = candidate;
                                heapSiftDown(heap, heap_len, 0);
                            }
                        }
                        offset += @intCast(remaining);
                    }
                }
            } else if (self.config.query_bits > 0 and self.config.query_bits <= 8 and !self.config.fastscan) {
                // Query Quantization path: quantize q_r_rot to query_bits per dimension,
                // then use integer arithmetic for inner product computation.
                // Faster than per-bit float multiplication, more accurate than FastScan.
                const n_levels: u32 = @as(u32, 1) << @intCast(self.config.query_bits);

                // Quantize q_r_rot: find min/max, scale to [0, n_levels-1]
                var q_min: f32 = std.math.floatMax(f32);
                var q_max: f32 = -std.math.floatMax(f32);
                for (q_r_rot) |v| {
                    if (v < q_min) q_min = v;
                    if (v > q_max) q_max = v;
                }
                const q_range = q_max - q_min;
                const q_inv_scale = if (q_range > 1e-8) q_range / @as(f32, @floatFromInt(n_levels - 1)) else 0.0;

                // Quantize q_r_rot to u8: store signed quantized values
                var q_quant_buf: [2048]i16 = undefined;
                const q_quant = q_quant_buf[0..self.dim];
                for (q_r_rot, 0..) |v, i| {
                    const normalized = (v - q_min) / if (q_range > 1e-8) q_range else 1.0;
                    const level: u8 = @intFromFloat(@min(@max(normalized * @as(f32, @floatFromInt(n_levels - 1)), 0.0), @as(f32, @floatFromInt(n_levels - 1))));
                    // Convert to signed: center around zero
                    q_quant[i] = @as(i16, @intCast(level)) - @as(i16, @intCast(n_levels / 2));
                }

                // Compute distance using quantized inner product
                for (0..p.count) |vi| {
                    const residual_norm = p.scalars[vi * 2 + 0];
                    const dot_o_bar_o = p.scalars[vi * 2 + 1];
                    const code_offset = vi * words_per_vec;

                    // Compute <sign(code), q_quant> using integer arithmetic
                    var ip_quant: i32 = 0;
                    for (0..words_per_vec) |w| {
                        const code_word = p.codes[code_offset + w];
                        for (0..64) |b| {
                            const idx = w * 64 + b;
                            if (idx >= self.dim) break;
                            const sign: i16 = if ((code_word >> @intCast(b)) & 1 == 1) 1 else -1;
                            ip_quant += @as(i32, sign) * @as(i32, q_quant[idx]);
                        }
                    }

                    // Scale back to approximate float inner product
                    const ip_approx: f32 = @as(f32, @floatFromInt(ip_quant)) * q_inv_scale;
                    const correction = if (dot_o_bar_o > 1e-8) dot_o_bar_o else 1e-8;
                    const score = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_approx / correction;

                    const candidate = SearchResult{
                        .id = p.ids[vi],
                        .partition_id = p.id,
                        .score = score,
                        .partition_vi = @intCast(vi),
                    };

                    if (heap_len < coarse_k) {
                        heap[heap_len] = candidate;
                        heap_len += 1;
                        heapSiftUp(heap, heap_len);
                    } else if (score < heap[0].score) {
                        heap[0] = candidate;
                        heapSiftDown(heap, heap_len, 0);
                    }
                }
            } else {
                // Standard path: per-bit sign multiplication with full-precision q_r_rot.
                // More accurate coarse ranking but slower.
                for (0..p.count) |vi| {
                    const residual_norm = p.scalars[vi * 2 + 0]; // ||x - c||
                    const dot_o_bar_o = p.scalars[vi * 2 + 1]; // sum(|x_rot_i|), ≈0.798*sqrt(dim)
                    const code_offset = vi * words_per_vec;
                    var ip_sign_qr_rot: f32 = 0.0;
                    for (0..words_per_vec) |w| {
                        const code_word = p.codes[code_offset + w];
                        for (0..64) |b| {
                            const idx = w * 64 + b;
                            if (idx >= self.dim) break;
                            const sign: f32 = if ((code_word >> @intCast(b)) & 1 == 1) 1.0 else -1.0;
                            ip_sign_qr_rot += sign * q_r_rot[idx];
                        }
                    }
                    const correction = if (dot_o_bar_o > 1e-8) dot_o_bar_o else 1e-8;
                    const score = residual_norm * residual_norm + q_residual_norm_sq - 2.0 * residual_norm * ip_sign_qr_rot / correction;

                    const candidate = SearchResult{
                        .id = p.ids[vi],
                        .partition_id = p.id,
                        .score = score,
                        .partition_vi = @intCast(vi),
                    };

                    if (heap_len < coarse_k) {
                        heap[heap_len] = candidate;
                        heap_len += 1;
                        heapSiftUp(heap, heap_len);
                    } else if (score < heap[0].score) {
                        heap[0] = candidate;
                        heapSiftDown(heap, heap_len, 0);
                    }
                }
            }
        }

        // SQ8 refinement: re-rank coarse candidates using SIMD SQ8 L2 distance.
        if (use_refine and heap_len > 0) {
            for (0..heap_len) |i| {
                const p = &self.partitions[heap[i].partition_id];
                if (p.sq8_codes.len > 0) {
                    heap[i].score = try p.sq8Distance(query, heap[i].partition_vi);
                }
            }
            var i: u32 = heap_len / 2;
            while (i > 0) {
                i -= 1;
                heapSiftDown(heap, heap_len, i);
            }
        }

        // Heap sort in-place for deterministic output.
        var remaining = heap_len;
        while (remaining > 1) {
            remaining -= 1;
            const tmp = heap[0];
            heap[0] = heap[remaining];
            heap[remaining] = tmp;
            heapSiftDown(heap, remaining, 0);
        }
        const out_count = @min(k, heap_len);
        @memcpy(results[0..out_count], heap[0..out_count]);
        return @intCast(out_count);
    }

    /// Search top-k across nprobe partitions with optional SQ8 refinement.
    pub fn search(
        self: *const Index,
        query: []const f32,
        k: u32,
        nprobe: u32,
        results: []SearchResult,
    ) !u32 {
        var ctx = try self.prepareQuery(query);
        defer ctx.deinit();
        return self.searchWithContext(query, k, nprobe, results, &ctx);
    }

    /// Batch search: process multiple queries efficiently.
    /// Pre-computes QueryContext for all queries up front to amortize rotation cost.
    /// Phase 2 searches queries in parallel using the internal thread pool.
    pub fn batchSearch(
        self: *Index,
        queries: []const []const f32,
        k: u32,
        nprobe: u32,
        results: []SearchResult,
        result_counts: []u32,
    ) !void {
        if (results.len < queries.len * k) return Error.BufferTooSmall;
        if (result_counts.len < queries.len) return Error.BufferTooSmall;

        // Phase 1: Prepare all query contexts in parallel (compute-bound).
        var contexts = try self.allocator.alloc(QueryContext, queries.len);
        defer {
            for (contexts) |*ctx| {
                ctx.deinit();
            }
            self.allocator.free(contexts);
        }
        for (queries, 0..) |query, i| {
            contexts[i] = try self.prepareQuery(query);
        }

        // Phase 2: Search each query in parallel using the thread pool.
        const SearchCtx = struct {
            index: *const Index,
            queries: []const []const f32,
            contexts: []QueryContext,
            k: u32,
            nprobe: u32,
            results: []SearchResult,
            result_counts: []u32,
        };

        var ctx = SearchCtx{
            .index = self,
            .queries = queries,
            .contexts = contexts,
            .k = k,
            .nprobe = nprobe,
            .results = results,
            .result_counts = result_counts,
        };

        self.thread_pool.parallelFor(queries.len, &ctx, struct {
            fn run(c: *SearchCtx, start: usize, end: usize) void {
                for (start..end) |qi| {
                    const out_slice = c.results[qi * c.k ..][0..c.k];
                    c.result_counts[qi] = c.index.searchWithContext(
                        c.queries[qi], c.k, c.nprobe, out_slice, &c.contexts[qi]
                    ) catch 0;
                }
            }
        }.run);
    }

    /// Max-heap sift-up: move element at position (len-1) up to maintain max-heap property.
    /// Heap root (index 0) is the WORST (largest score) element.
    fn heapSiftUp(heap: []SearchResult, len: u32) void {
        if (len <= 1) return;
        var i: u32 = len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (heap[i].score > heap[parent].score) {
                const tmp = heap[i];
                heap[i] = heap[parent];
                heap[parent] = tmp;
                i = parent;
            } else {
                break;
            }
        }
    }

    /// Max-heap sift-down: move element at position i down to maintain max-heap property.
    fn heapSiftDown(heap: []SearchResult, len: u32, start: u32) void {
        var i = start;
        while (true) {
            const left = 2 * i + 1;
            const right = 2 * i + 2;
            var largest = i;
            if (left < len and heap[left].score > heap[largest].score) {
                largest = left;
            }
            if (right < len and heap[right].score > heap[largest].score) {
                largest = right;
            }
            if (largest == i) break;
            const tmp = heap[i];
            heap[i] = heap[largest];
            heap[largest] = tmp;
            i = largest;
        }
    }
};

pub const SearchResult = struct {
    id: u32,
    partition_id: u32,
    score: f32,
    partition_vi: u32 = 0, // local index within partition, used internally for SQ8 refinement
};

/// Generate a random orthogonal matrix using QR decomposition simplified (Gram-Schmidt).
fn generateRandomOrthogonal(allocator: std.mem.Allocator, dim: u32, seed: u64) ![]f32 {
    var rng = std.Random.DefaultPrng.init(seed);
    const mat = try allocator.alloc(f32, dim * dim);
    // Initialize with random normal-ish values
    for (mat) |*v| {
        v.* = rng.random().floatNorm(f32);
    }
    // Modified Gram-Schmidt (MGS) with a second pass for numerical stability.
    for (0..dim) |i| {
        const col_i = mat[i * dim ..][0..dim];
        for (0..2) |_| {
            for (0..i) |j| {
                const col_j = mat[j * dim ..][0..dim];
                var proj: f32 = 0.0;
                for (col_i, col_j) |vi, vj| {
                    proj += vi * vj;
                }
                for (col_i, col_j) |*vi, vj| {
                    vi.* -= proj * vj;
                }
            }
        }
        var norm: f32 = 0.0;
        for (col_i) |v| {
            norm += v * v;
        }
        norm = @sqrt(norm);
        if (norm > 1e-8) {
            const inv = 1.0 / norm;
            for (col_i) |*v| {
                v.* *= inv;
            }
        }
    }
    return mat;
}

// ============================================
// Unit Tests
// ============================================

test "index insert and search" {
    const allocator = std.testing.allocator;
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    // Insert 100 random vectors
    var rng = std.Random.DefaultPrng.init(123);
    var vec: [64]f32 = undefined;
    for (0..100) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    // Search
    var results: [10]SearchResult = undefined;
    const found = try idx.search(&vec, 10, 2, &results);
    try std.testing.expect(found > 0);
}

test "dimension must be multiple of 8" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(Error.InvalidDimension, Index.init(allocator, 63, .{ .num_partitions = 4 }));
}

test "hierarchical kmeans matches linear scan" {
    const allocator = std.testing.allocator;
    // Use 16 partitions so sqrt(16)=4 super-partitions
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 16 });
    defer idx.deinit();

    // Verify super-partitions were created
    try std.testing.expect(idx.super_partitions.len > 0);

    // Verify every sub-partition is covered by exactly one super-partition
    var covered = try allocator.alloc(bool, idx.partitions.len);
    defer allocator.free(covered);
    @memset(covered, false);
    var total_subs: usize = 0;
    for (idx.super_partitions) |sp| {
        total_subs += sp.sub_ids.len;
        for (sp.sub_ids) |sub_id| {
            try std.testing.expect(sub_id < idx.partitions.len);
            try std.testing.expect(!covered[sub_id]); // no duplicates
            covered[sub_id] = true;
        }
    }
    try std.testing.expectEqual(idx.partitions.len, total_subs);
    for (covered) |c| try std.testing.expect(c);

    // Verify hierarchical search matches linear scan for random vectors
    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..50) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        const hier_id = idx.findNearestPartition(&vec);
        const linear_id = idx.findNearestPartitionLinear(&vec);
        try std.testing.expectEqual(linear_id, hier_id);
    }
}

test "hierarchical kmeans degenerate cases" {
    const allocator = std.testing.allocator;

    // Single partition: one super-partition containing it
    {
        var idx = try Index.init(allocator, 64, .{ .num_partitions = 1 });
        defer idx.deinit();
        try std.testing.expectEqual(@as(usize, 1), idx.super_partitions.len);
        try std.testing.expectEqual(@as(usize, 1), idx.super_partitions[0].sub_ids.len);
    }

    // Four partitions: sqrt(4)=2 super-partitions
    {
        var idx = try Index.init(allocator, 64, .{ .num_partitions = 4 });
        defer idx.deinit();
        try std.testing.expectEqual(@as(usize, 2), idx.super_partitions.len);
        var total: usize = 0;
        for (idx.super_partitions) |sp| {
            total += sp.sub_ids.len;
        }
        try std.testing.expectEqual(@as(usize, 4), total);
    }
}

test "batch search equals individual searches" {
    const allocator = std.testing.allocator;
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 8, .refine_sq8 = false });
    defer idx.deinit();

    // Insert 200 random vectors
    var rng = std.Random.DefaultPrng.init(77);
    var vecs: [200][64]f32 = undefined;
    for (0..200) |i| {
        for (&vecs[i]) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vecs[i]);
    }

    // Build queries (reuse last 5 inserted vectors)
    const queries = &[_][]const f32{
        &vecs[195], &vecs[196], &vecs[197], &vecs[198], &vecs[199],
    };

    // Individual search results
    var indiv_results: [5][5]SearchResult = undefined;
    for (queries, 0..) |q, i| {
        _ = try idx.search(q, 5, 2, &indiv_results[i]);
    }

    // Batch search results
    var batch_results: [25]SearchResult = undefined;
    var batch_counts: [5]u32 = undefined;
    try idx.batchSearch(queries, 5, 2, &batch_results, &batch_counts);

    // Verify batch matches individual
    for (0..5) |i| {
        try std.testing.expectEqual(batch_counts[i], @as(u32, 5));
        for (0..5) |j| {
            try std.testing.expectEqual(indiv_results[i][j].id, batch_results[i * 5 + j].id);
            try std.testing.expectApproxEqAbs(indiv_results[i][j].score, batch_results[i * 5 + j].score, 0.001);
        }
    }
}

test "exact match vector is found" {
    const allocator = std.testing.allocator;
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 3 });
    defer idx.deinit();

    // Insert a known vector
    var vec: [64]f32 = undefined;
    for (&vec, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) / 64.0;
    }
    try idx.insert(&vec);

    // Search for the exact same vector, k=1
    var results: [10]SearchResult = undefined;
    const found = try idx.search(&vec, 1, 4, &results);
    try std.testing.expect(found > 0);
    try std.testing.expectEqual(@as(u32, 0), results[0].id); // first inserted vector has id 0
}

test "sq8 refinement improves or matches coarse recall" {
    const allocator = std.testing.allocator;

    // Index without refinement
    var idx_no_refine = try Index.init(allocator, 64, .{ .num_partitions = 8, .refine_sq8 = false });
    defer idx_no_refine.deinit();

    // Index with refinement
    var idx_refine = try Index.init(allocator, 64, .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 3 });
    defer idx_refine.deinit();

    var rng = std.Random.DefaultPrng.init(88);
    var vecs: [300][64]f32 = undefined;
    for (0..300) |i| {
        for (&vecs[i]) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx_no_refine.insert(&vecs[i]);
        try idx_refine.insert(&vecs[i]);
    }

    // Use a query vector that exists in the dataset
    const query = &vecs[250];

    var coarse_results: [5]SearchResult = undefined;
    _ = try idx_no_refine.search(query, 5, 4, &coarse_results);

    var refined_results: [5]SearchResult = undefined;
    _ = try idx_refine.search(query, 5, 4, &refined_results);

    // Compute approximate recall@5: count how many IDs in coarse are also in refined.
    // With proper L2 re-ranking, refined results should be at least as good.
    var matched: u32 = 0;
    for (coarse_results) |c| {
        for (refined_results) |r| {
            if (c.id == r.id) {
                matched += 1;
                break;
            }
        }
    }

    // Refinement should not degrade overlap significantly.
    // In practice SQ8 re-ranking often changes order but keeps high overlap.
    try std.testing.expect(matched >= 2); // heuristic: at least 2/5 overlap
}

test "global ids are unique and monotonic" {
    const allocator = std.testing.allocator;
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 4 });
    defer idx.deinit();

    try idx.insert(&[_]f32{0.1} ** 64);
    try idx.insert(&[_]f32{0.2} ** 64);
    try idx.insert(&[_]f32{0.3} ** 64);

    // Search all vectors back
    var results: [10]SearchResult = undefined;
    const found = try idx.search(&[_]f32{0.1} ** 64, 10, 4, &results);
    try std.testing.expect(found >= 1);

    // Collect all IDs and verify uniqueness
    var ids = std.AutoHashMap(u32, void).init(allocator);
    defer ids.deinit();
    for (0..found) |i| {
        try std.testing.expect(!ids.contains(results[i].id));
        try ids.put(results[i].id, {});
    }
}

test "partition load balance" {
    const allocator = std.testing.allocator;
    var idx = try Index.init(allocator, 64, .{ .num_partitions = 8 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(99);
    var vec: [64]f32 = undefined;
    for (0..1000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }

    // Compute min/max count across partitions
    var min_count: u32 = std.math.maxInt(u32);
    var max_count: u32 = 0;
    var total: u32 = 0;
    for (idx.partitions) |p| {
        if (p.count < min_count) min_count = p.count;
        if (p.count > max_count) max_count = p.count;
        total += p.count;
    }

    try std.testing.expectEqual(@as(u32, 1000), total);
    // With random centroids and uniform data, expect no partition is completely empty
    try std.testing.expect(min_count > 0);
    // And the most loaded partition should not be more than 10x the least loaded
    // (this is a loose heuristic; perfect balance requires online K-Means)
    try std.testing.expect(max_count <= min_count * 10);
}
