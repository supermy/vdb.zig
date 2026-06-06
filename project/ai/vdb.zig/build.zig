const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Core library module
    const vdb_mod = b.addModule("vdb", .{
        .root_source_file = b.path("src/vdb.zig"),
        .target = target,
        .optimize = optimize,
    });

    // SIMD module
    const simd_mod = b.addModule("simd", .{
        .root_source_file = b.path("src/simd.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Thread pool module
    const thread_pool_mod = b.addModule("thread_pool", .{
        .root_source_file = b.path("src/thread_pool.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Index module
    const index_mod = b.addModule("index_ivf_rq", .{
        .root_source_file = b.path("src/index_ivf_rq.zig"),
        .target = target,
        .optimize = optimize,
    });
    index_mod.addImport("simd", simd_mod);
    index_mod.addImport("vdb", vdb_mod);
    index_mod.addImport("thread_pool", thread_pool_mod);

    // Search module
    const search_mod = b.addModule("search", .{
        .root_source_file = b.path("src/search.zig"),
        .target = target,
        .optimize = optimize,
    });
    search_mod.addImport("index_ivf_rq", index_mod);
    search_mod.addImport("simd", simd_mod);
    search_mod.addImport("vdb", vdb_mod);

    // GPU module
    const gpu_mod = b.addModule("gpu", .{
        .root_source_file = b.path("src/gpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    gpu_mod.addImport("simd", simd_mod);

    // Storage module (columnar disk format)
    const storage_mod = b.addModule("storage", .{
        .root_source_file = b.path("src/storage.zig"),
        .target = target,
        .optimize = optimize,
    });
    storage_mod.addImport("index_ivf_rq", index_mod);
    storage_mod.addImport("simd", simd_mod);
    storage_mod.addImport("thread_pool", thread_pool_mod);

    // CLI executable (embedded mode)
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    const cli = b.addExecutable(.{
        .name = "vdb-cli",
        .root_module = cli_mod,
    });
    cli_mod.addImport("vdb", vdb_mod);
    cli_mod.addImport("simd", simd_mod);
    cli_mod.addImport("index_ivf_rq", index_mod);
    cli_mod.addImport("search", search_mod);
    cli_mod.addImport("gpu", gpu_mod);
    cli_mod.addImport("storage", storage_mod);
    b.installArtifact(cli);

    // HTTP Server executable
    const server_mod = b.createModule(.{
        .root_source_file = b.path("src/server.zig"),
        .target = target,
        .optimize = optimize,
    });
    const server = b.addExecutable(.{
        .name = "vdb-server",
        .root_module = server_mod,
    });
    server_mod.addImport("vdb", vdb_mod);
    server_mod.addImport("simd", simd_mod);
    server_mod.addImport("index_ivf_rq", index_mod);
    server_mod.addImport("search", search_mod);
    server_mod.addImport("gpu", gpu_mod);
    server_mod.addImport("storage", storage_mod);
    b.installArtifact(server);

    // NNG High-performance server executable
    const nng_mod = b.createModule(.{
        .root_source_file = b.path("src/nng_server.zig"),
        .target = target,
        .optimize = optimize,
    });
    const nng_server = b.addExecutable(.{
        .name = "vdb-nng-server",
        .root_module = nng_mod,
    });
    nng_mod.addImport("vdb", vdb_mod);
    nng_mod.addImport("simd", simd_mod);
    nng_mod.addImport("index_ivf_rq", index_mod);
    nng_mod.addImport("search", search_mod);
    nng_mod.addImport("gpu", gpu_mod);
    nng_mod.addImport("storage", storage_mod);
    b.installArtifact(nng_server);

    // Benchmark executable
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });
    const benchmark = b.addExecutable(.{
        .name = "vdb-benchmark",
        .root_module = bench_mod,
    });
    bench_mod.addImport("vdb", vdb_mod);
    bench_mod.addImport("simd", simd_mod);
    bench_mod.addImport("index_ivf_rq", index_mod);
    bench_mod.addImport("search", search_mod);
    bench_mod.addImport("gpu", gpu_mod);
    bench_mod.addImport("storage", storage_mod);
    b.installArtifact(benchmark);

    // Run CLI
    const run_cli = b.addRunArtifact(cli);
    if (b.args) |args| {
        run_cli.addArgs(args);
    }
    const run_step = b.step("run", "Run the CLI");
    run_step.dependOn(&run_cli.step);

    // Run server
    const run_server = b.addRunArtifact(server);
    if (b.args) |args| {
        run_server.addArgs(args);
    }
    const run_server_step = b.step("run-server", "Run the HTTP server");
    run_server_step.dependOn(&run_server.step);

    // Run NNG server
    const run_nng = b.addRunArtifact(nng_server);
    if (b.args) |args| {
        run_nng.addArgs(args);
    }
    const run_nng_step = b.step("run-nng", "Run the NNG high-performance server");
    run_nng_step.dependOn(&run_nng.step);

    // ============================================
    // Test Suite
    // ============================================

    // Unit tests
    const unit_mod = b.createModule(.{
        .root_source_file = b.path("tests/unit/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const unit_tests = b.addTest(.{
        .name = "unit",
        .root_module = unit_mod,
    });
    unit_mod.addImport("vdb", vdb_mod);
    unit_mod.addImport("simd", simd_mod);
    unit_mod.addImport("index_ivf_rq", index_mod);
    unit_mod.addImport("search", search_mod);
    unit_mod.addImport("gpu", gpu_mod);
    unit_mod.addImport("thread_pool", thread_pool_mod);
    unit_mod.addImport("storage", storage_mod);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const unit_test_step = b.step("test-unit", "Run unit tests");
    unit_test_step.dependOn(&run_unit_tests.step);

    // Server tests
    const server_tests = b.addTest(.{
        .name = "server",
        .root_module = server_mod,
    });
    const run_server_tests = b.addRunArtifact(server_tests);
    const server_test_step = b.step("test-server", "Run server tests");
    server_test_step.dependOn(&run_server_tests.step);

    // Integration tests
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const integration_tests = b.addTest(.{
        .name = "integration",
        .root_module = integration_mod,
    });
    integration_mod.addImport("vdb", vdb_mod);
    integration_mod.addImport("simd", simd_mod);
    integration_mod.addImport("index_ivf_rq", index_mod);
    integration_mod.addImport("search", search_mod);
    integration_mod.addImport("gpu", gpu_mod);
    integration_mod.addImport("storage", storage_mod);
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Smoke tests (with verbose logging)
    const smoke_mod = b.createModule(.{
        .root_source_file = b.path("tests/smoke/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const smoke_tests = b.addTest(.{
        .name = "smoke",
        .root_module = smoke_mod,
    });
    smoke_mod.addImport("vdb", vdb_mod);
    smoke_mod.addImport("simd", simd_mod);
    smoke_mod.addImport("index_ivf_rq", index_mod);
    smoke_mod.addImport("search", search_mod);
    smoke_mod.addImport("gpu", gpu_mod);
    smoke_mod.addImport("storage", storage_mod);
    const run_smoke_tests = b.addRunArtifact(smoke_tests);
    const smoke_test_step = b.step("test-smoke", "Run smoke tests with debug logging");
    smoke_test_step.dependOn(&run_smoke_tests.step);

    // Regression tests
    const regression_mod = b.createModule(.{
        .root_source_file = b.path("tests/regression/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const regression_tests = b.addTest(.{
        .name = "regression",
        .root_module = regression_mod,
    });
    regression_mod.addImport("vdb", vdb_mod);
    regression_mod.addImport("simd", simd_mod);
    regression_mod.addImport("index_ivf_rq", index_mod);
    regression_mod.addImport("search", search_mod);
    regression_mod.addImport("gpu", gpu_mod);
    regression_mod.addImport("storage", storage_mod);
    const run_regression_tests = b.addRunArtifact(regression_tests);
    const regression_test_step = b.step("test-regression", "Run regression tests");
    regression_test_step.dependOn(&run_regression_tests.step);

    // Acceptance tests
    const acceptance_mod = b.createModule(.{
        .root_source_file = b.path("tests/acceptance/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const acceptance_tests = b.addTest(.{
        .name = "acceptance",
        .root_module = acceptance_mod,
    });
    acceptance_mod.addImport("vdb", vdb_mod);
    acceptance_mod.addImport("simd", simd_mod);
    acceptance_mod.addImport("index_ivf_rq", index_mod);
    acceptance_mod.addImport("search", search_mod);
    acceptance_mod.addImport("gpu", gpu_mod);
    acceptance_mod.addImport("storage", storage_mod);
    const run_acceptance_tests = b.addRunArtifact(acceptance_tests);
    const acceptance_test_step = b.step("test-acceptance", "Run acceptance tests");
    acceptance_test_step.dependOn(&run_acceptance_tests.step);

    // System tests
    const system_mod = b.createModule(.{
        .root_source_file = b.path("tests/system/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const system_tests = b.addTest(.{
        .name = "system",
        .root_module = system_mod,
    });
    system_mod.addImport("vdb", vdb_mod);
    system_mod.addImport("simd", simd_mod);
    system_mod.addImport("index_ivf_rq", index_mod);
    system_mod.addImport("search", search_mod);
    system_mod.addImport("gpu", gpu_mod);
    system_mod.addImport("storage", storage_mod);
    const run_system_tests = b.addRunArtifact(system_tests);
    const system_test_step = b.step("test-system", "Run system tests");
    system_test_step.dependOn(&run_system_tests.step);

    // End-to-end tests
    const e2e_mod = b.createModule(.{
        .root_source_file = b.path("tests/e2e/all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const e2e_tests = b.addTest(.{
        .name = "e2e",
        .root_module = e2e_mod,
    });
    e2e_mod.addImport("vdb", vdb_mod);
    e2e_mod.addImport("simd", simd_mod);
    e2e_mod.addImport("index_ivf_rq", index_mod);
    e2e_mod.addImport("search", search_mod);
    e2e_mod.addImport("gpu", gpu_mod);
    e2e_mod.addImport("storage", storage_mod);
    const run_e2e_tests = b.addRunArtifact(e2e_tests);
    const e2e_test_step = b.step("test-e2e", "Run end-to-end tests");
    e2e_test_step.dependOn(&run_e2e_tests.step);

    // All tests combined
    const all_tests = b.step("test", "Run all tests (unit + integration + smoke + regression + acceptance + system + e2e + server)");
    all_tests.dependOn(&run_unit_tests.step);
    all_tests.dependOn(&run_integration_tests.step);
    all_tests.dependOn(&run_smoke_tests.step);
    all_tests.dependOn(&run_regression_tests.step);
    all_tests.dependOn(&run_acceptance_tests.step);
    all_tests.dependOn(&run_system_tests.step);
    all_tests.dependOn(&run_e2e_tests.step);
    all_tests.dependOn(&run_server_tests.step);

    // Coverage step (placeholder for future kcov integration)
    const coverage_step = b.step("coverage", "Generate test coverage report");
    coverage_step.dependOn(all_tests);

    // Benchmark step
    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run performance benchmarks vs LanceDB & FAISS");
    benchmark_step.dependOn(&run_benchmark.step);
}
