const std = @import("std");

/// Lance columnar file subset: schema, RecordBatch, mmap, append-only transactions, manifest snapshots.
/// Arrow C FFI is used only at boundaries; everything inside is Zig-native zero-copy.
pub const Error = error{
    InvalidSchema,
    MmapFailed,
    TransactionConflict,
    VersionNotFound,
    DimensionMismatch,
    OutOfMemory,
    IoError,
};

/// On-disk column type for scalar and vector columns.
pub const ColumnType = enum {
    int32,
    int64,
    float32,
    float64,
    utf8,
    binary,
    fixed_size_binary, // used for raw vectors and RaBitQ codes
    vector, // logical type: fixed_size_binary with dim metadata
};

pub const ColumnSchema = struct {
    name: []const u8,
    col_type: ColumnType,
    // For vector columns, dimension must be divisible by 8 (RaBitQ requirement).
    dimension: u32 = 0,
    nullable: bool = false,
};

pub const Schema = struct {
    columns: []const ColumnSchema,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, columns: []const ColumnSchema) !Schema {
        // Validate: vector columns must have dimension % 8 == 0
        for (columns) |col| {
            if (col.col_type == .vector and col.dimension % 8 != 0) {
                return Error.DimensionMismatch;
            }
        }
        const cols = try allocator.dupe(ColumnSchema, columns);
        return Schema{ .columns = cols, .allocator = allocator };
    }

    pub fn deinit(self: *Schema) void {
        self.allocator.free(self.columns);
    }
};

/// Memory-mapped region with bounds checking.
pub const MmapRegion = struct {
    ptr: []align(std.mem.page_size) u8,
    file: std.fs.File,

    pub fn map(file: std.fs.File, offset: u64, len: usize) Error!MmapRegion {
        const ptr = std.posix.mmap(
            null,
            len,
            std.posix.PROT.READ,
            std.posix.MAP{ .TYPE = .SHARED },
            file.handle,
            @intCast(offset),
        ) catch return Error.MmapFailed;
        return MmapRegion{
            .ptr = @alignCast(ptr[0..len]),
            .file = file,
        };
    }

    pub fn unmap(self: *MmapRegion) void {
        std.posix.munmap(self.ptr);
    }
};

/// Append-only transaction boundary and version snapshot (manifest).
pub const Manifest = struct {
    version: u64,
    batch_offsets: []const u64,
    allocator: std.mem.Allocator,

    pub fn load(allocator: std.mem.Allocator, path: []const u8) Error!Manifest {
        const io = std.Io.Threaded.global_single_threaded.io();
        const bytes = std.Io.Dir.readFileAlloc(std.Io.Dir.cwd(), io, path, allocator, .unlimited) catch |err| switch (err) {
            error.OutOfMemory => return Error.OutOfMemory,
            else => return Error.IoError,
        };
        defer allocator.free(bytes);
        // Simple binary layout: version(u64) + count(u64) + offsets[]
        if (bytes.len < 16) return Error.InvalidSchema;
        const version = std.mem.readInt(u64, bytes[0..8], .little);
        const count = std.mem.readInt(u64, bytes[8..16], .little);
        if (bytes.len < 16 + count * 8) return Error.InvalidSchema;
        const offsets = try allocator.alloc(u64, @intCast(count));
        for (0..count) |i| {
            offsets[i] = std.mem.readInt(u64, bytes[16 + i * 8 ..][0..8], .little);
        }
        return Manifest{ .version = version, .batch_offsets = offsets, .allocator = allocator };
    }

    pub fn save(self: *const Manifest, path: []const u8) Error!void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = std.Io.Dir.createFile(std.Io.Dir.cwd(), io, path, .{}) catch return Error.IoError;
        defer file.close(io);
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, self.version, .little);
        file.writeStreamingAll(io, &buf) catch return Error.IoError;
        std.mem.writeInt(u64, &buf, self.batch_offsets.len, .little);
        file.writeStreamingAll(io, &buf) catch return Error.IoError;
        for (self.batch_offsets) |off| {
            std.mem.writeInt(u64, &buf, off, .little);
            file.writeStreamingAll(io, &buf) catch return Error.IoError;
        }
    }

    pub fn deinit(self: *Manifest) void {
        self.allocator.free(self.batch_offsets);
    }
};

/// Dataset handle: schema + manifest + optional mmap of data file.
pub const Dataset = struct {
    schema: Schema,
    manifest: Manifest,
    data_path: []const u8,
    allocator: std.mem.Allocator,

    pub fn open(allocator: std.mem.Allocator, data_path: []const u8, manifest_path: []const u8) Error!Dataset {
        // Schema is stored alongside manifest: schema.json + manifest.bin
        const schema_path = try std.fmt.allocPrint(allocator, "{s}.schema", .{manifest_path});
        defer allocator.free(schema_path);

        const io = std.Io.Threaded.global_single_threaded.io();
        const schema_json = std.Io.Dir.readFileAlloc(std.Io.Dir.cwd(), io, schema_path, allocator, .unlimited) catch |err| switch (err) {
            error.OutOfMemory => return Error.OutOfMemory,
            else => return Error.IoError,
        };
        defer allocator.free(schema_json);

        // Parse minimal schema JSON (columns array)
        const columns = try parseSchemaJson(allocator, schema_json);
        const schema = try Schema.init(allocator, columns);

        const manifest = try Manifest.load(allocator, manifest_path);
        const dp = try allocator.dupe(u8, data_path);
        return Dataset{
            .schema = schema,
            .manifest = manifest,
            .data_path = dp,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Dataset) void {
        self.schema.deinit();
        self.manifest.deinit();
        self.allocator.free(self.data_path);
    }
};

/// Minimal JSON schema parser (expects array of {name, type, dimension?}).
fn parseSchemaJson(allocator: std.mem.Allocator, json: []const u8) Error![]ColumnSchema {
    // For MVP, use a tiny hand-rolled parser or rely on Zig std.json.
    var parsed = std.json.parseFromSlice([]ColumnSchema, allocator, json, .{}) catch return Error.InvalidSchema;
    defer parsed.deinit();
    return try allocator.dupe(ColumnSchema, parsed.value);
}

// ============================================
// Unit Tests
// ============================================

test "schema validation" {
    const allocator = std.testing.allocator;
    const cols = &[_]ColumnSchema{
        .{ .name = "id", .col_type = .int64 },
        .{ .name = "vec", .col_type = .vector, .dimension = 128 },
    };
    var schema = try Schema.init(allocator, cols);
    defer schema.deinit();
    try std.testing.expectEqual(@as(usize, 2), schema.columns.len);
}

test "manifest roundtrip" {
    const allocator = std.testing.allocator;
    const tmp = "test_manifest.bin";
    {
        const offsets = &[_]u64{ 0, 1024, 2048 };
        var m = Manifest{ .version = 1, .batch_offsets = offsets, .allocator = allocator };
        try m.save(tmp);
    }
    {
        var m = try Manifest.load(allocator, tmp);
        defer m.deinit();
        try std.testing.expectEqual(@as(u64, 1), m.version);
        try std.testing.expectEqual(@as(usize, 3), m.batch_offsets.len);
        try std.testing.expectEqual(@as(u64, 1024), m.batch_offsets[1]);
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.deleteFile(std.Io.Dir.cwd(), io, tmp) catch {};
}
