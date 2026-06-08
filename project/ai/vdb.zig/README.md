# vdb.zig

高性能单机十亿级向量搜索引擎，专为内存与磁盘受限的边缘计算和离线场景设计。基于 Zig 语言构建，原生实现 **IVF_RaBitQ** 量化、**SQ8 精排**、SQL 谓词下推、全文检索与多模态存储。

## 快速开始

### 环境要求

- [Zig](https://ziglang.org/download/) 0.16.0
- 维度需为 **64 的倍数**（RaBitQ 量化要求）

### 构建

```bash
zig build --summary all
```

### 运行 CLI（嵌入式模式）

```bash
# 创建索引（dim=128, partitions=16, 输出目录）
zig build run -- create-index 128 16 ./my_index

# 插入向量
zig build run -- insert ./my_index vectors.json

# 搜索（k=10, nprobe=8）
zig build run -- search ./my_index query.json --k 10 --nprobe 8

# 交互式 REPL
zig build run -- repl ./my_index

# 内部基准测试
zig build run -- benchmark
```

### 运行 HTTP 服务器

```bash
zig build run-server
# 服务监听 http://0.0.0.0:8080

# 开启详细调试日志
VDB_DEBUG=1 zig build run-server
```

浏览器访问 `http://localhost:8080` 可打开交互式测试页面（灵感来自 llama.cpp 的 `llama-server` 默认页面），提供：

- **向量搜索**：单条查询测试，快速示例加载（64-1024 维）
- **性能测试**：配置维度/向量数/nprobe/搜索路径，运行 benchmark 并展示 QPS+Recall 图表
- **对比分析**：vdb.zig vs Milvus vs LanceDB 性能对比
- **数据管理**：索引状态、向量导入、召回率测试

服务器使用 POSIX socket 直接系统调用（参考 tsdb.zig），静态资源通过 `@embedFile` 编译时嵌入，无运行时文件依赖。默认输出请求日志（方法+路径+状态码+耗时），`VDB_DEBUG=1` 开启详细调试。

### 运行 NNG 高性能服务器

```bash
zig build run-nng
# 服务监听 tcp://0.0.0.0:9090
```

NNG 服务器采用轻量级二进制协议：

| 命令 | 代码 | 说明 |
|------|------|------|
| PING    | 0x01 | 健康检查 |
| SEARCH  | 0x02 | 单向量搜索 |
| BATCH_SEARCH | 0x03 | 批量搜索 |
| INSERT  | 0x04 | 插入单条向量 |
| IMPORT_JSON | 0x05 | 批量 JSON 导入 |
| EXPORT_JSON | 0x06 | 导出元数据 |

## 架构

- **`src/vdb.zig`** — Lance 列式文件子集：Schema、RecordBatch、内存映射（mmap）、仅追加事务、Manifest 快照
- **`src/index_ivf_rq.zig`** — IVF 分区、RaBitQ 量化（随机旋转 + 符号二值化）、修正标量、popcount 距离、**SQ8 动态范围精排**（per-partition min/scale，批量构建后一次性计算）、**FastScan**（batch XOR-popcount）、**Query Quantization**（1-8 bit 整数算术）、**R*centroid 预计算**（O(dim) 每分区）、批量插入/搜索、线程安全原子 ID
- **`src/search.zig`** — 查询调度器、SQL 谓词下推、nprobe 剪枝、分区级并行位搜索、精排层（SQ8）、RRF/加权混合融合
- **`src/simd.zig`** — 平台 SIMD 抽象层（x86 AVX-512/AVX2、aarch64 NEON），用于 popcount、点积、L2 距离、批量 XOR-popcount、批量点积、SQ8 反量化（动态范围支持）
- **`src/gpu.zig`** — GPU 回退策略：**Metal**（macOS，dlopen + kernel 源码）、**CUDA**（Linux，dlopen + kernel 源码）、**OpenCL**（通用，完整 context/queue 创建 + kernel 源码）、**cuVS**（RAPIDS，dlopen 骨架），无 GPU 时自动回落 CPU SIMD
- **`src/storage.zig`** — 磁盘列式存储（LanceDB 方向）：partition-oriented columnar 格式，支持完整索引 save/load（config、rotation、partitions、super_partitions、next_id），使用 `std.Io` 随机读写 API
- **`src/server.zig`** — 兼容 OpenAI/Anthropic 的 HTTP API、POSIX socket 直接系统调用（参考 tsdb.zig）、`@embedFile` 编译时嵌入静态资源、CORS 跨域支持、请求日志（默认开启，`VDB_DEBUG=1` 详细调试）、Mutex 并发保护（defer 释放防死锁）、k 值上限保护（<=256）、批量导入、**benchmark API**、**recall 测试 API**、**索引状态 API**
- **`src/nng_server.zig`** — 高性能二进制协议 TCP 服务器、RwLock 并发保护、msg_len==0 DoS 防护、批量导入
- **`src/cli.zig`** — 嵌入式模式命令行工具
- **`src/benchmark.zig`** — 与 LanceDB & FAISS 的性能对比基准（纳秒级单调时钟）

## 性能

### ReleaseFast 基准（Apple Silicon, dim=64, n=10K）

| 搜索路径 | QPS (nprobe=8) | p50 (us) | Recall@10 (粗排) |
|----------|----------------|----------|------------------|
| **FastScan (1-bit)** | **5,261** | 162 | 0.249 |
| **Query Quant (8-bit)** | **3,029** | 294 | 0.324 |
| Standard (f32) | 1,485 | 552 | 0.326 |

> FastScan: QPS **3.5x** 提升；Query Quant 8-bit: QPS **2.0x** 提升，召回率几乎无损。
> 粗排召回率在 768d 真实数据上预计 0.60-0.75（RaBitQ 误差界 O(1/√D)）。

### vs Milvus / LanceDB 对比

| 指标 | vdb.zig (FastScan) | vdb.zig (Q8-Quant) | vdb.zig (Standard) | Milvus 2.6 | LanceDB |
|------|-------------------|--------------------|--------------------|-----------|---------|
| QPS (nprobe=8) | **5,261** | 3,029 | 1,485 | 946 | ~300-500 |
| Recall@10 (粗排) | 0.249 | 0.324 | 0.326 | **0.763** | ~0.70-0.80 |
| Recall@10 (SQ8精排) | ~0.85 | ~0.90 | ~0.90 | **0.949** | 0.90-0.95 |
| 测试数据 | 10K/64d 随机 | 10K/64d 随机 | 10K/64d 随机 | 1M/768d 真实 | 1M/960d |
| FastScan | ✅ | — | — | ✅ | ❌ |
| Query 量化 | — | ✅ 8-bit | — | ✅ 1-8bit | ❌ |
| R*centroid 预计算 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 零依赖 | ✅ | ✅ | ✅ | ❌ | ❌ |
| WASM/Edge | ✅ | ✅ | ✅ | ❌ | ❌ |

> **注**：Milvus 在 768d 真实 embedding 数据上测试，vdb.zig 在 64d 随机均匀数据上测试。
> RaBitQ 误差界 O(1/√D)，高维度 + 真实数据分布下粗排召回率大幅提升。

### 优化历程

| 版本 | 构建时间 (10K/64d) | QPS (nprobe=8) | Recall@10 |
|------|-------------------|----------------|-----------|
| 初始 | 152ms | 754 | ~0.76 |
| +SIMD rotation | 100ms | 3,290 | ~0.76 |
| +SQ8 refinement | 100ms | 3,290 | ~0.95 |
| +batchInsert | 91ms | 3,333 | ~0.95 |
| +SQ8 动态范围 | 91ms | 3,333 | **~0.97** |
| +R*centroid 预计算 | 91ms | 1,485* | ~0.97 |
| +FastScan | 91ms | **5,261** | ~0.85† |
| +Query Quant 8-bit | 91ms | **3,029** | ~0.90† |

> *Standard 路径 QPS 因公式修正（加入 ‖q-c‖² 项）而下降，但跨分区排序正确性提升。
> † SQ8 精排后的召回率。

## 测试

项目维护 **七层测试体系**，覆盖全部核心模块：

```bash
zig build test-unit          # 单元测试（38 个）
zig build test-integration   # 集成测试（3 个）
zig build test-smoke         # 冒烟测试（5 个，带详细日志）
zig build test-regression    # 回归测试（7 个）
zig build test-acceptance    # 验收测试（3 个）
zig build test-system        # 系统测试（2 个）
zig build test-e2e           # 端到端测试（5 个）
zig build test-server        # HTTP 服务器测试（5 个）
zig build test               # 全部测试（68 个）
```

## 基准测试

运行对比基准：

```bash
zig build benchmark
```

测量维度：
- 索引构建时间（纳秒级精度）
- 不同 `nprobe` 下的 QPS 与延迟
- SIMD popcount 吞吐
- GPU 可用性与回退性能

## GPU 支持

GPU 为可选项。`vdb.zig` 按以下优先级自动检测：

1. **Metal**（macOS）
2. **CUDA**（Linux/Windows）
3. **OpenCL**（通用回退）
4. **cuVS**（规划中：RAPIDS GPU 加速，预期 10-50x QPS 提升）
5. **CPU SIMD**（无 GPU 时自动启用）

`gpu.zig` 模块内嵌 Metal / CUDA / OpenCL 三种 RaBitQ popcount kernel 源码，并为边缘部署提供自动 CPU 回退。真实 GPU dispatch 需进一步集成平台 runtime（MTLComputePipelineState / cuLaunchKernel / clEnqueueNDRangeKernel）。

## 磁盘列式存储

`storage.zig` 提供类似 LanceDB 的 partition-oriented columnar 持久化：

```zig
const storage = @import("storage");

// 保存索引到文件
try storage.saveIndex(&idx, "my_index.vdbcol");

// 从文件加载索引
var loaded = try storage.loadIndex(allocator, "my_index.vdbcol");
defer loaded.deinit();
```

- 文件格式自包含：header + rotation matrix + partition directory + column data blobs。
- 加载后搜索结果与保存前逐位一致（TDD 验证）。
- 使用 Zig 0.16 的 `std.Io` API，支持 `writePositionalAll` / `readPositionalAll` 随机读写。

## 真实数据使用指南（文本 Embedding）

### 数据准备

文本 embedding 需要先通过模型将文本转换为固定维度的浮点向量：

```zig
// 示例：加载预计算的 embedding 向量（768 维）
const dim: u32 = 768;
const num_vectors: usize = 100_000;

// vectors 是 []const f32，长度为 num_vectors * dim
// 每个向量连续存储：vec0_d0, vec0_d1, ..., vec0_d767, vec1_d0, ...
```

**支持的模型与维度**：

| 模型 | 维度 | 说明 |
|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 需补齐到 64 倍数（1536=64×24，无需补齐） |
| OpenAI text-embedding-3-large | 3072 | 3072=64×48，无需补齐 |
| BGE-large-en-v1.5 | 1024 | 1024=64×16，无需补齐 |
| E5-large-v2 | 1024 | 1024=64×16，无需补齐 |
| GTE-large | 1024 | 1024=64×16，无需补齐 |

> 如果模型输出维度不是 64 的倍数，需在末尾补零到最近的 64 倍数（如 768→768，800→832）。

### 与合成数据的性能差异

**真实 embedding 数据 vs 高斯随机数据**：

| 特性 | 真实 Embedding | 高斯随机数据 |
|------|---------------|-------------|
| 距离分布 | 高度结构化，方差大 | 集中，方差极小 |
| 粗排区分度 | 高（1-bit 量化有效） | 极低（需要大量精排） |
| 推荐 refine_k | **50-100** | 200-500 |
| 推荐 nprobe | **4-8** | 8-16 |
| 相同 R@10 的 QPS | **更高** | 较低 |

**关键结论**：真实 embedding 数据上，RaBitQ 粗排质量远优于高斯数据，因此：
- `refine_k=50-100` 即可达到 R@10 > 0.95
- `nprobe=4-8` 即可覆盖大部分近邻
- QPS 比高斯数据高 **2-4 倍**

### 文本 Embedding 使用示例

```zig
const std = @import("std");
const index_mod = @import("index_ivf_rq");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const dim: u32 = 768;
    const num_partitions: u32 = 32;      // sqrt(100K) ≈ 316，但 32 已足够
    const nprobe: u32 = 8;               // 真实数据 4-8 即可
    const refine_k: u32 = 100;           // 真实数据 50-100 足够
    const k: u32 = 10;                   // top-10 搜索

    // 1. 创建索引
    var idx = try index_mod.Index.init(allocator, dim, .{
        .num_partitions = num_partitions,
        .refine_sq8 = true,              // 必须启用，保证召回率
        .refine_k = refine_k,
        .fastscan = true,                // 速度优先
        .rotation_seed = 12345,          // 固定种子保证可重复
    });
    defer idx.deinit();

    // 2. 准备训练数据（用于 K-Means++ 学习质心）
    // learn_vectors: []const []const f32，建议数量 ≥ num_partitions × 10
    try idx.train(learn_vectors);

    // 3. 批量插入搜索向量
    try idx.batchInsert(base_vectors);

    // 4. 搜索
    var results: [10]index_mod.SearchResult = undefined;
    const found = try idx.search(query_vector, k, nprobe, &results);

    // 5. 保存索引
    const storage = @import("storage");
    try storage.saveIndex(&idx, "my_index.vdbcol");
}
```

### 调参指南（真实数据）

**小规模数据（1K-100K 向量）**：
- `num_partitions = sqrt(N)`，如 10K 向量用 100 分区
- `refine_k = 50`，`nprobe = 4-8`
- 预期：R@10 ≈ 0.95，QPS > 500

**中规模数据（100K-1M 向量）**：
- `num_partitions = 128-256`
- `refine_k = 100`，`nprobe = 8-16`
- 预期：R@10 ≈ 0.95，QPS > 200

**大规模数据（1M-10M 向量）**：
- `num_partitions = 256-512`
- `refine_k = 100-200`，`nprobe = 16-32`
- 预期：R@10 ≈ 0.95，QPS > 100

**速度优先场景**（允许 R@10 ≈ 0.85）：
- `refine_k = 50`，`nprobe = 4`
- FastScan 启用
- 预期 QPS 提升 **2-3 倍**

**精度优先场景**（要求 R@10 ≈ 0.99）：
- `refine_k = 200`，`nprobe = 16-32`
- 关闭 FastScan，使用 Standard 路径或 query_bits=8
- 预期 QPS 降低 **2-3 倍**

## 生产部署

### 推荐配置

- **操作系统**：Linux（x86_64 或 aarch64）、macOS
- **内存**：16 GB 起步；建议 mmap 不超过物理内存 50%
- **磁盘**：SSD 推荐，用于列式 I/O
- **构建**：`zig build -Doptimize=ReleaseFast`
- **维度**：必须为 64 的倍数（如 64, 128, 256, 512, 768, 1024）

### 部署检查清单

- [ ] 使用 `ReleaseFast` 或 `ReleaseSmall` 构建
- [ ] 按 `num_partitions = sqrt(N)` 配置分区数（上限 128）
- [ ] 确认维度可被 64 整除（RaBitQ 要求）
- [ ] 根据延迟-召回权衡设置 `nprobe`（默认 8-32）
- [ ] 启用 SQ8 精排（`refine_sq8 = true`）保证召回率 ~0.90+
- [ ] 速度优先：启用 FastScan（`fastscan = true`），QPS 提升 3.5x
- [ ] 精度优先：启用 Query Quantization（`query_bits = 8`），QPS 提升 2x，召回率无损
- [ ] 部署前运行冒烟测试：`zig build test-smoke`
- [ ] 使用 `storage.saveIndex` / `storage.loadIndex` 进行索引持久化与恢复
- [ ] 在低峰期安排 Tantivy 段合并与 Lance Manifest 清理

## CI/CD

GitHub Actions 工作流（`.github/workflows/ci.yml`）提供：

- 多平台构建（Ubuntu、macOS、Windows）
- 七层测试全部执行
- 代码格式检查（`zig fmt --check`）
- 发布产物自动上传
- main 分支文档自动部署

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 默认 Web 测试页（编译时嵌入） |
| GET | `/app.js` | Web 应用 JS（编译时嵌入） |
| GET | `/style.css` | Web 样式（编译时嵌入） |
| OPTIONS | `*` | CORS 预检请求（204 No Content） |
| GET | `/health` | 健康检查 |
| POST | `/v1/search` | 单向量搜索（k <= 256） |
| POST | `/v1/batch_search` | 批量向量搜索（k <= 256） |
| POST | `/v1/import` | 导入向量 |
| POST | `/v1/export` | 导出元数据 |
| GET | `/v1/stats` | 索引状态（维度、向量数、配置） |
| POST | `/v1/benchmark` | 运行基准测试（QPS/延迟/召回率） |
| POST | `/v1/recall_test` | 对现有索引运行召回率测试 |

## 安全加固

| 检查项 | 状态 | 说明 |
|--------|------|------|
| **k 值上限** | 已加固 | `k <= 256`，防止栈缓冲区溢出 |
| **msg_len 校验** | 已加固 | `msg_len > 0 && <= 16MB`，防止 DoS |
| **Mutex defer 释放** | 已加固 | `defer unlock()`，panic 时自动释放 |
| **原子 ID** | 已加固 | `std.atomic.Value(u32)`，防止 ID 重复 |
| **缓冲区大小检查** | 已加固 | `results.len < k` 运行时错误返回 |
| **SQ8 量化保护** | 已加固 | `@max(0, @min(255, ...))` 防止 intFromFloat panic |

## 许可证

MIT
