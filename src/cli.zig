const std = @import("std");
const index_mod = @import("index_ivf_rq");
const gpu = @import("gpu");

fn writeStdout(msg: []const u8) void {
    const file = std.Io.File.stdout();
    const io = std.Io.Threaded.global_single_threaded.io();
    file.writeStreamingAll(io, msg) catch |err| std.log.err("stdout write failed: {}", .{err});
}

fn monoMs() i64 {
    var ts: std.posix.timespec = undefined;
    const rc = std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts);
    std.debug.assert(rc == 0);
    return @as(i64, @intCast(ts.sec)) * 1000 + @divTrunc(@as(i64, @intCast(ts.nsec)), 1_000_000);
}

/// Command-line interface for vdb.zig (embedded mode).
/// Commands:
///   create-index <dim> <partitions> <output>
///   insert <index_path> <vector.json>
///   search <index_path> <query.json> [--k N] [--nprobe M]
///   repl <index_path>
pub fn main(minimal: std.process.Init.Minimal) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try minimal.args.toSlice(allocator);
    if (args.len < 2) {
        printUsage();
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "create-index")) {
        if (args.len < 5) {
            writeStdout("Usage: vdb-cli create-index <dim> <partitions> <output_dir>\n");
            return;
        }
        const dim = try std.fmt.parseInt(u32, args[2], 10);
        const partitions = try std.fmt.parseInt(u32, args[3], 10);
        const output = args[4];
        try createIndex(allocator, dim, partitions, output);
    } else if (std.mem.eql(u8, cmd, "insert")) {
        if (args.len < 4) {
            writeStdout("Usage: vdb-cli insert <index_dir> <vectors.json>\n");
            return;
        }
        try insertVectors(allocator, args[2], args[3]);
    } else if (std.mem.eql(u8, cmd, "search")) {
        if (args.len < 4) {
            writeStdout("Usage: vdb-cli search <index_dir> <query.json> [--k N] [--nprobe M]\n");
            return;
        }
        var k: u32 = 10;
        var nprobe: u32 = 8;
        var i: usize = 4;
        while (i < args.len) : (i += 2) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--k") and i + 1 < args.len) {
                k = try std.fmt.parseInt(u32, args[i + 1], 10);
            } else if (std.mem.eql(u8, flag, "--nprobe") and i + 1 < args.len) {
                nprobe = try std.fmt.parseInt(u32, args[i + 1], 10);
            }
        }
        try searchQuery(allocator, args[2], args[3], k, nprobe);
    } else if (std.mem.eql(u8, cmd, "repl")) {
        if (args.len < 3) {
            writeStdout("Usage: vdb-cli repl <index_dir>\n");
            return;
        }
        try runRepl(args[2]);
    } else if (std.mem.eql(u8, cmd, "benchmark")) {
        try runBenchmark(allocator);
    } else {
        printUsage();
    }
}

fn printUsage() void {
    writeStdout("Usage: vdb-cli <command> [options]\n" ++
        "\n" ++
        "Commands:\n" ++
        "  create-index <dim> <partitions> <output_dir>   Create a new IVF_RaBitQ index\n" ++
        "  insert <index_dir> <vectors.json>              Insert vectors from JSON file\n" ++
        "  search <index_dir> <query.json> [--k N]        Search the index\n" ++
        "  repl <index_dir>                               Start interactive REPL\n" ++
        "  benchmark                                      Run internal benchmark\n" ++
        "\n");
}

fn createIndex(allocator: std.mem.Allocator, dim: u32, partitions: u32, output: []const u8) !void {
    var idx = try index_mod.Index.init(allocator, dim, .{ .num_partitions = partitions });
    defer idx.deinit();

    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.createDirPath(std.Io.Dir.cwd(), io, output);
    const meta_path = try std.fs.path.join(allocator, &.{ output, "index.meta" });
    defer allocator.free(meta_path);

    var file = try std.Io.Dir.createFile(std.Io.Dir.cwd(), io, meta_path, .{});
    defer file.close(io);
    const meta = try std.fmt.allocPrint(allocator, "{{\"dim\":{d},\"partitions\":{d},\"type\":\"IVF_RaBitQ\"}}\n", .{ dim, partitions });
    defer allocator.free(meta);
    try file.writeStreamingAll(io, meta);

    var buf: [256]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, "Created index: dim={d}, partitions={d} at {s}\n", .{ dim, partitions, output });
    writeStdout(msg);
}

fn insertVectors(allocator: std.mem.Allocator, index_dir: []const u8, vectors_path: []const u8) !void {
    _ = index_dir;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFile(std.Io.Dir.cwd(), io, vectors_path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const data = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(data);
    _ = try file.readPositionalAll(io, data, 0);

    const parsed = try std.json.parseFromSlice([][]f32, allocator, data, .{});
    defer parsed.deinit();

    var buf: [256]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, "Parsed {d} vectors from {s}\n", .{ parsed.value.len, vectors_path });
    writeStdout(msg);
}

fn searchQuery(allocator: std.mem.Allocator, index_dir: []const u8, query_path: []const u8, k: u32, nprobe: u32) !void {
    _ = index_dir;
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.openFile(std.Io.Dir.cwd(), io, query_path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const data = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(data);
    _ = try file.readPositionalAll(io, data, 0);

    const parsed = try std.json.parseFromSlice([]f32, allocator, data, .{});
    defer parsed.deinit();

    var buf: [256]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, "Query vector loaded, dim={d}, k={d}, nprobe={d}\n", .{ parsed.value.len, k, nprobe });
    writeStdout(msg);
}

fn runRepl(index_dir: []const u8) !void {
    var buf: [256]u8 = undefined;
    const msg = try std.fmt.bufPrint(&buf, "Starting REPL for index: {s}\n", .{index_dir});
    writeStdout(msg);
    writeStdout("Type 'quit' or 'exit' to leave.\n");

    while (true) {
        writeStdout("vdb> ");
        const stdin = std.Io.File.stdin();
        const io = std.Io.Threaded.global_single_threaded.io();
        var reader_buf: [1024]u8 = undefined;
        var reader = stdin.reader(io, &reader_buf);
        const line = reader.interface.takeDelimiter('\n') catch break;
        const input = std.mem.trim(u8, line orelse break, "\r\n");
        if (std.mem.eql(u8, input, "quit") or std.mem.eql(u8, input, "exit")) break;
        writeStdout("Echo: ");
        writeStdout(input);
        writeStdout("\n");
    }
}

fn runBenchmark(allocator: std.mem.Allocator) !void {
    writeStdout("Running internal benchmark...\n");
    var idx = try index_mod.Index.init(allocator, 128, .{ .num_partitions = 16 });
    defer idx.deinit();

    var rng = std.Random.DefaultPrng.init(123);
    var vec: [128]f32 = undefined;

    const insert_start = monoMs();
    for (0..10000) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        try idx.insert(&vec);
    }
    const insert_elapsed = monoMs() - insert_start;
    var buf: [256]u8 = undefined;
    const msg1 = try std.fmt.bufPrint(&buf, "Inserted 10000 vectors in {d} ms\n", .{insert_elapsed});
    writeStdout(msg1);

    const search_start = monoMs();
    var total_found: u32 = 0;
    for (0..100) |_| {
        for (&vec) |*v| {
            v.* = rng.random().float(f32);
        }
        var results: [10]index_mod.SearchResult = undefined;
        total_found += try idx.search(&vec, 10, 4, &results);
    }
    const search_elapsed = monoMs() - search_start;
    const msg2 = try std.fmt.bufPrint(&buf, "100 searches completed in {d} ms, total_found={d}\n", .{ search_elapsed, total_found });
    writeStdout(msg2);
}
