const std = @import("std");
const index_mod = @import("index_ivf_rq");

/// Columnar disk storage for IVF-RaBitQ index.
/// Layout inspired by LanceDB: partition-oriented row groups with columnar data inside.
///
/// File layout (little-endian):
///   Header (64 bytes)
///   Rotation matrix (dim * dim * sizeof(f32) bytes)
///   SuperPartition directory
///   Partition directory
///   Column data blobs (codes, scalars, ids, sq8, centroids, etc.)
///
/// All offsets are absolute from the start of the file.
pub const StorageFormat = struct {
    pub const magic = "VDBCOL\x00\x00";
    pub const version: u32 = 2;

    pub const Header = extern struct {
        magic: [8]u8,
        version: u32,
        num_partitions: u32,
        dim: u32,
        num_bits: u32,
        epsilon_0: f32,
        rotation_seed: u64,
        refine_sq8: u8,
        refine_k: u32,
        fastscan: u8,
        query_bits: u32,
        num_super_partitions: u32,
        next_id: u32,
        reserved: [10]u8,
    };

    pub const PartitionEntry = extern struct {
        codes_offset: u64,
        codes_len_bytes: u64,
        scalars_offset: u64,
        scalars_len_bytes: u64,
        ids_offset: u64,
        ids_len_bytes: u64,
        sq8_codes_offset: u64,
        sq8_codes_len_bytes: u64,
        sq8_min_offset: u64,
        sq8_min_len_bytes: u64,
        sq8_scale_offset: u64,
        sq8_scale_len_bytes: u64,
        sq8_max_offset: u64,
        sq8_max_len_bytes: u64,
        centroid_offset: u64,
        centroid_len_bytes: u64,
        count: u32,
        capacity: u32,
        reserved: [8]u8,
    };

    pub const SuperPartitionEntry = extern struct {
        sub_ids_offset: u64,
        sub_ids_len_bytes: u64,
        centroid_offset: u64,
        centroid_len_bytes: u64,
        reserved: [16]u8,
    };
};

pub const Error = error{
    InvalidMagic,
    UnsupportedVersion,
    IoError,
    CorruptData,
};

/// Save an IVF-RaBitQ index to a columnar file.
pub fn saveIndex(index: *const index_mod.Index, path: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.Io.Dir.cwd();
    const file = std.Io.Dir.createFile(dir, io, path, .{}) catch return Error.IoError;
    defer file.close(io);

    const num_partitions = @as(u32, @intCast(index.partitions.len));
    const num_super = @as(u32, @intCast(index.super_partitions.len));
    const dim = index.dim;

    var offset: u64 = 0;

    // === Header ===
    var header = StorageFormat.Header{
        .magic = StorageFormat.magic.*,
        .version = StorageFormat.version,
        .num_partitions = num_partitions,
        .dim = dim,
        .num_bits = index.config.num_bits,
        .epsilon_0 = index.config.epsilon_0,
        .rotation_seed = index.config.rotation_seed,
        .refine_sq8 = if (index.config.refine_sq8) 1 else 0,
        .refine_k = index.config.refine_k,
        .fastscan = if (index.config.fastscan) 1 else 0,
        .query_bits = index.config.query_bits,
        .num_super_partitions = num_super,
        .next_id = index.next_id.load(.monotonic),
        .reserved = std.mem.zeroes([10]u8),
    };
    try file.writeStreamingAll(io, std.mem.asBytes(&header));
    offset += @sizeOf(StorageFormat.Header);

    // === Rotation matrix ===
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(index.rotation));
    offset += index.rotation.len * @sizeOf(f32);

    // === SuperPartition directory placeholder ===
    const super_dir_offset = offset;
    const super_dir_size = num_super * @sizeOf(StorageFormat.SuperPartitionEntry);
    var super_entries = try index.allocator.alloc(StorageFormat.SuperPartitionEntry, num_super);
    defer index.allocator.free(super_entries);
    @memset(super_entries, std.mem.zeroes(StorageFormat.SuperPartitionEntry));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(super_entries));
    offset += super_dir_size;

    // === Partition directory placeholder ===
    const part_dir_offset = offset;
    const part_dir_size = num_partitions * @sizeOf(StorageFormat.PartitionEntry);
    var part_entries = try index.allocator.alloc(StorageFormat.PartitionEntry, num_partitions);
    defer index.allocator.free(part_entries);
    @memset(part_entries, std.mem.zeroes(StorageFormat.PartitionEntry));
    try file.writeStreamingAll(io, std.mem.sliceAsBytes(part_entries));
    offset += part_dir_size;

    // === Column data ===
    for (index.partitions, 0..) |p, pi| {
        const entry = &part_entries[pi];
        entry.count = p.count;
        entry.capacity = p.capacity;

        // centroid
        entry.centroid_offset = offset;
        entry.centroid_len_bytes = p.centroid.len * @sizeOf(f32);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.centroid));
        offset += entry.centroid_len_bytes;

        // codes
        entry.codes_offset = offset;
        entry.codes_len_bytes = p.codes.len * @sizeOf(u64);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.codes));
        offset += entry.codes_len_bytes;

        // scalars
        entry.scalars_offset = offset;
        entry.scalars_len_bytes = p.scalars.len * @sizeOf(f32);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.scalars));
        offset += entry.scalars_len_bytes;

        // ids
        entry.ids_offset = offset;
        entry.ids_len_bytes = p.ids.len * @sizeOf(u32);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.ids));
        offset += entry.ids_len_bytes;

        // sq8 (optional)
        if (p.sq8_codes.len > 0) {
            entry.sq8_codes_offset = offset;
            entry.sq8_codes_len_bytes = p.sq8_codes.len * @sizeOf(u8);
            try file.writeStreamingAll(io, p.sq8_codes);
            offset += entry.sq8_codes_len_bytes;

            entry.sq8_min_offset = offset;
            entry.sq8_min_len_bytes = p.sq8_min.len * @sizeOf(f32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.sq8_min));
            offset += entry.sq8_min_len_bytes;

            entry.sq8_scale_offset = offset;
            entry.sq8_scale_len_bytes = p.sq8_scale.len * @sizeOf(f32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.sq8_scale));
            offset += entry.sq8_scale_len_bytes;

            entry.sq8_max_offset = offset;
            entry.sq8_max_len_bytes = p.sq8_max.len * @sizeOf(f32);
            try file.writeStreamingAll(io, std.mem.sliceAsBytes(p.sq8_max));
            offset += entry.sq8_max_len_bytes;
        }
    }

    // Write super-partition data
    for (index.super_partitions, 0..) |sp, si| {
        const entry = &super_entries[si];

        entry.centroid_offset = offset;
        entry.centroid_len_bytes = sp.centroid.len * @sizeOf(f32);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(sp.centroid));
        offset += entry.centroid_len_bytes;

        entry.sub_ids_offset = offset;
        entry.sub_ids_len_bytes = sp.sub_ids.len * @sizeOf(u32);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(sp.sub_ids));
        offset += entry.sub_ids_len_bytes;
    }

    // Ensure data is persisted before rewriting directory
    try file.sync(io);

    // Rewrite directories with correct offsets using positional writes
    try file.writePositionalAll(io, std.mem.sliceAsBytes(super_entries), super_dir_offset);
    try file.writePositionalAll(io, std.mem.sliceAsBytes(part_entries), part_dir_offset);
}

/// Load an IVF-RaBitQ index from a columnar file.
/// The returned index owns all memory and must be deinit'd.
pub fn loadIndex(allocator: std.mem.Allocator, path: []const u8) !index_mod.Index {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.Io.Dir.cwd();
    const file = std.Io.Dir.openFile(dir, io, path, .{}) catch return Error.IoError;
    defer file.close(io);

    const file_size = file.length(io) catch return Error.IoError;

    if (file_size < @sizeOf(StorageFormat.Header)) {
        return Error.CorruptData;
    }

    // === Header ===
    var header: StorageFormat.Header = undefined;
    _ = try file.readPositionalAll(io, std.mem.asBytes(&header), 0);
    if (!std.mem.eql(u8, &header.magic, &StorageFormat.magic.*)) {
        return Error.InvalidMagic;
    }
    if (header.version != StorageFormat.version) {
        return Error.UnsupportedVersion;
    }

    const config = index_mod.RaBitQConfig{
        .num_partitions = header.num_partitions,
        .num_bits = header.num_bits,
        .epsilon_0 = header.epsilon_0,
        .rotation_seed = header.rotation_seed,
        .refine_sq8 = header.refine_sq8 != 0,
        .refine_k = header.refine_k,
        .fastscan = header.fastscan != 0,
        .query_bits = header.query_bits,
    };

    const dim = header.dim;
    const rot_len = dim * dim;

    var offset: u64 = @sizeOf(StorageFormat.Header);

    // === Rotation matrix ===
    const rotation = try allocator.alloc(f32, rot_len);
    errdefer allocator.free(rotation);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(rotation), offset);
    offset += rot_len * @sizeOf(f32);

    // === SuperPartition directory ===
    const num_super = header.num_super_partitions;
    const super_entries = try allocator.alloc(StorageFormat.SuperPartitionEntry, num_super);
    defer allocator.free(super_entries);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(super_entries), offset);
    offset += num_super * @sizeOf(StorageFormat.SuperPartitionEntry);

    // === Partition directory ===
    const num_partitions = header.num_partitions;
    const part_entries = try allocator.alloc(StorageFormat.PartitionEntry, num_partitions);
    defer allocator.free(part_entries);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(part_entries), offset);
    offset += num_partitions * @sizeOf(StorageFormat.PartitionEntry);

    // === Partitions ===
    const partitions = try allocator.alloc(index_mod.Partition, num_partitions);
    var partitions_initialized: usize = 0;
    errdefer {
        for (partitions[0..partitions_initialized]) |*p| p.deinit();
        allocator.free(partitions);
    }

    for (0..num_partitions) |pi| {
        const entry = part_entries[pi];
        const p = &partitions[pi];
        p.id = @intCast(pi);
        p.allocator = allocator;
        p.count = entry.count;
        p.capacity = entry.capacity;

        // Validate offsets are within file bounds
        if (entry.centroid_offset + entry.centroid_len_bytes > file_size or
            entry.codes_offset + entry.codes_len_bytes > file_size or
            entry.scalars_offset + entry.scalars_len_bytes > file_size or
            entry.ids_offset + entry.ids_len_bytes > file_size)
        {
            return Error.CorruptData;
        }

        // Validate sq8 offsets if present
        if (entry.sq8_codes_len_bytes > 0) {
            if (entry.sq8_codes_offset + entry.sq8_codes_len_bytes > file_size or
                entry.sq8_min_offset + entry.sq8_min_len_bytes > file_size or
                entry.sq8_scale_offset + entry.sq8_scale_len_bytes > file_size or
                entry.sq8_max_offset + entry.sq8_max_len_bytes > file_size)
            {
                return Error.CorruptData;
            }
        }

        // Validate centroid dimension consistency
        const expected_centroid_words = dim;
        if (entry.centroid_len_bytes != expected_centroid_words * @sizeOf(f32)) return Error.CorruptData;

        // centroid
        const centroid_words = entry.centroid_len_bytes / @sizeOf(f32);
        p.centroid = try allocator.alloc(f32, centroid_words);
        errdefer allocator.free(p.centroid);
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.centroid), entry.centroid_offset);

        // centroid_rot: allocated here, computed after rotation is loaded
        p.centroid_rot = try allocator.alloc(f32, centroid_words);
        errdefer allocator.free(p.centroid_rot);
        @memset(p.centroid_rot, 0.0);

        // codes
        const code_words = entry.codes_len_bytes / @sizeOf(u64);
        p.codes = try allocator.alloc(u64, code_words);
        errdefer allocator.free(p.codes);
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.codes), entry.codes_offset);

        // scalars
        const scalar_words = entry.scalars_len_bytes / @sizeOf(f32);
        p.scalars = try allocator.alloc(f32, scalar_words);
        errdefer allocator.free(p.scalars);
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.scalars), entry.scalars_offset);

        // ids
        const id_words = entry.ids_len_bytes / @sizeOf(u32);
        p.ids = try allocator.alloc(u32, id_words);
        errdefer allocator.free(p.ids);
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.ids), entry.ids_offset);

        // sq8 (optional)
        if (entry.sq8_codes_len_bytes > 0) {
            const sq8_words = entry.sq8_codes_len_bytes;
            p.sq8_codes = try allocator.alloc(u8, sq8_words);
            errdefer allocator.free(p.sq8_codes);
            _ = try file.readPositionalAll(io, p.sq8_codes, entry.sq8_codes_offset);

            const min_words = entry.sq8_min_len_bytes / @sizeOf(f32);
            p.sq8_min = try allocator.alloc(f32, min_words);
            errdefer allocator.free(p.sq8_min);
            _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.sq8_min), entry.sq8_min_offset);

            const scale_words = entry.sq8_scale_len_bytes / @sizeOf(f32);
            p.sq8_scale = try allocator.alloc(f32, scale_words);
            errdefer allocator.free(p.sq8_scale);
            _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.sq8_scale), entry.sq8_scale_offset);

            const max_words = entry.sq8_max_len_bytes / @sizeOf(f32);
            p.sq8_max = try allocator.alloc(f32, max_words);
            errdefer allocator.free(p.sq8_max);
            _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(p.sq8_max), entry.sq8_max_offset);
        } else {
            p.sq8_codes = &[_]u8{};
            p.sq8_min = &[_]f32{};
            p.sq8_scale = &[_]f32{};
            p.sq8_max = &[_]f32{};
        }

        // All allocations succeeded — mark as initialized so outer errdefer calls deinit()
        partitions_initialized += 1;
    }

    // === SuperPartitions ===
    const super_partitions = try allocator.alloc(index_mod.SuperPartition, num_super);
    var super_partitions_initialized: usize = 0;
    errdefer {
        for (super_partitions[0..super_partitions_initialized]) |*sp| sp.deinit();
        allocator.free(super_partitions);
    }

    for (0..num_super) |si| {
        const entry = super_entries[si];

        // Validate SuperPartition offsets
        if (entry.centroid_offset + entry.centroid_len_bytes > file_size or
            entry.sub_ids_offset + entry.sub_ids_len_bytes > file_size)
        {
            return Error.CorruptData;
        }

        const centroid_words = entry.centroid_len_bytes / @sizeOf(f32);
        const sub_id_words = entry.sub_ids_len_bytes / @sizeOf(u32);

        const sub_ids = try allocator.alloc(u32, sub_id_words);
        errdefer allocator.free(sub_ids);
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(sub_ids), entry.sub_ids_offset);

        const centroid = try allocator.alloc(f32, centroid_words);
        errdefer allocator.free(centroid);

        super_partitions[si] = index_mod.SuperPartition{
            .id = @intCast(si),
            .centroid = centroid,
            .sub_ids = sub_ids,
            .allocator = allocator,
        };
        _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(super_partitions[si].centroid), entry.centroid_offset);

        super_partitions_initialized += 1;
    }

    const thread_count = @max(1, std.Thread.getCpuCount() catch 4);
    const tp = try @import("thread_pool").ThreadPool.create(allocator, thread_count);
    errdefer tp.destroy();

    // Precompute centroid_rot = R * centroid for each partition
    const simd_mod = @import("simd");
    for (partitions) |*p| {
        for (0..dim) |r| {
            const row = rotation[r * dim ..][0..dim];
            p.centroid_rot[r] = simd_mod.dotProduct(row, p.centroid);
        }
    }

    return index_mod.Index{
        .dim = dim,
        .config = config,
        .rotation = rotation,
        .partitions = partitions,
        .super_partitions = super_partitions,
        .next_id = std.atomic.Value(u32).init(header.next_id),
        .allocator = allocator,
        .thread_pool = tp,
    };
}

// ============================================
// Unit Tests
// ============================================

fn cleanupFile(path: []const u8) void {
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.deleteFile(std.Io.Dir.cwd(), io, path) catch {};
}

test "storage roundtrip: empty index" {
    const allocator = std.testing.allocator;
    const path = "test_storage_empty.bin";
    defer cleanupFile(path);

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = false });
    defer idx.deinit();

    try saveIndex(&idx, path);

    var loaded = try loadIndex(allocator, path);
    defer loaded.deinit();

    try std.testing.expectEqual(idx.dim, loaded.dim);
    try std.testing.expectEqual(idx.partitions.len, loaded.partitions.len);
    try std.testing.expectEqual(idx.super_partitions.len, loaded.super_partitions.len);
    try std.testing.expectEqual(idx.config.num_bits, loaded.config.num_bits);
}

test "storage roundtrip: with vectors" {
    const allocator = std.testing.allocator;
    const path = "test_storage_vectors.bin";
    defer cleanupFile(path);

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 4, .refine_sq8 = true, .refine_k = 3 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    var vec: [64]f32 = undefined;
    for (0..50) |_| {
        for (&vec) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vec);
    }

    try saveIndex(&idx, path);

    var loaded = try loadIndex(allocator, path);
    defer loaded.deinit();

    try std.testing.expectEqual(idx.dim, loaded.dim);
    try std.testing.expectEqual(idx.partitions.len, loaded.partitions.len);
    try std.testing.expectEqual(idx.config.refine_sq8, loaded.config.refine_sq8);
    try std.testing.expectEqual(idx.next_id.load(.monotonic), loaded.next_id.load(.monotonic));

    // Verify partition counts and data
    var total_orig: u32 = 0;
    var total_loaded: u32 = 0;
    for (idx.partitions, loaded.partitions) |po, pl| {
        total_orig += po.count;
        total_loaded += pl.count;
        try std.testing.expectEqual(po.count, pl.count);
        try std.testing.expectEqual(po.capacity, pl.capacity);
        try std.testing.expectEqual(po.codes.len, pl.codes.len);
        try std.testing.expectEqual(po.scalars.len, pl.scalars.len);
        try std.testing.expectEqual(po.ids.len, pl.ids.len);
        try std.testing.expectEqual(po.sq8_codes.len, pl.sq8_codes.len);

        if (po.count > 0) {
            try std.testing.expectEqualSlices(f32, po.centroid, pl.centroid);
            try std.testing.expectEqualSlices(u64, po.codes, pl.codes);
            try std.testing.expectEqualSlices(f32, po.scalars, pl.scalars);
            try std.testing.expectEqualSlices(u32, po.ids, pl.ids);
        }
    }
    try std.testing.expectEqual(total_orig, total_loaded);

    // Verify search works after load
    var results: [10]index_mod.SearchResult = undefined;
    const found = try loaded.search(&vec, 5, 2, &results);
    try std.testing.expect(found > 0);
}

test "storage roundtrip: search results identical before/after" {
    const allocator = std.testing.allocator;
    const path = "test_storage_identical.bin";
    defer cleanupFile(path);

    var idx = try index_mod.Index.init(allocator, 64, .{ .num_partitions = 8, .refine_sq8 = true, .refine_k = 2 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(77);
    var vecs: [100][64]f32 = undefined;
    for (0..100) |i| {
        for (&vecs[i]) |*v| v.* = rng.random().float(f32);
        try idx.insert(&vecs[i]);
    }

    const query = &vecs[99];
    var before: [10]index_mod.SearchResult = undefined;
    const before_found = try idx.search(query, 5, 4, &before);

    try saveIndex(&idx, path);

    var loaded = try loadIndex(allocator, path);
    defer loaded.deinit();

    var after: [10]index_mod.SearchResult = undefined;
    const after_found = try loaded.search(query, 5, 4, &after);

    try std.testing.expectEqual(before_found, after_found);
    for (0..before_found) |i| {
        try std.testing.expectEqual(before[i].id, after[i].id);
        try std.testing.expectApproxEqAbs(before[i].score, after[i].score, 0.001);
    }
}

test "storage corrupt data: too small" {
    const allocator = std.testing.allocator;
    const path = "test_storage_corrupt.bin";
    defer cleanupFile(path);

    const io = std.Io.Threaded.global_single_threaded.io();
    const file = std.Io.Dir.createFile(std.Io.Dir.cwd(), io, path, .{}) catch return;
    defer file.close(io);
    try file.writeStreamingAll(io, "short");

    try std.testing.expectError(Error.CorruptData, loadIndex(allocator, path));
}

test "storage corrupt data: bad magic" {
    const allocator = std.testing.allocator;
    const path = "test_storage_magic.bin";
    defer cleanupFile(path);

    const io = std.Io.Threaded.global_single_threaded.io();
    const file = std.Io.Dir.createFile(std.Io.Dir.cwd(), io, path, .{}) catch return;
    defer file.close(io);
    var bad_header = std.mem.zeroes(StorageFormat.Header);
    @memcpy(bad_header.magic[0..8], "BADMAGIC");
    bad_header.version = 1;
    try file.writeStreamingAll(io, std.mem.asBytes(&bad_header));

    try std.testing.expectError(Error.InvalidMagic, loadIndex(allocator, path));
}
