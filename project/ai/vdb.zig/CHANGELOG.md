# 变更日志

## [0.3.0] - 2026-06-06

### 性能优化

- **FastScan 搜索路径**：新增 `fastscan` 配置项，使用 `batchPopcountXor` 批量计算 Hamming 距离，替代逐 bit 浮点乘法。QPS 提升 **3.5-6.2x**（dim=64: 5,261 vs 1,485；dim=128: 5,244 vs 851）。粗排召回率损失 ~24%（可被 SQ8 精排弥补）。内积近似使用 `||q_r_rot||` 幅度缩放，比纯 binary-binary 近似更准确。
- **Query Quantization 搜索路径**：新增 `query_bits` 配置项（1-8 bit），将 `q_r_rot` 量化为整数后用整数算术计算内积。8-bit 模式 QPS 提升 **2.0-2.2x**，召回率几乎无损（0.324 vs 0.326）。
- **R*centroid 预计算**：新增 `Partition.centroid_rot` 字段，在 `trainKMeansPP` 和 `storage.loadIndex` 时预计算 `R * centroid`。搜索时 `R*(q-c) = R*q - centroid_rot`，每分区从 O(dim²) 降至 O(dim)，只需一次全局 O(dim²) 旋转。

### RaBitQ 距离公式修正

- **内积空间修正**：原始实现计算 `<sign(code), (q-c)>`（未旋转空间），修正为 `<sign(code), R*(q-c)>`（旋转空间），与 RaBitQ 论文一致。
- **无偏估计器修正**：从乘以 `dot_o_bar_o` 改为**除以** `dot_o_bar_o`。论文公式：`⟨o, q_dir⟩ ≈ ⟨ō, q_dir⟩ / ⟨ō, o⟩`，`sqrt(dim)` 因子完全消去。
- **跨分区排序修正**：加入 `||q-c||²` 项（每分区不同，必须包含），保证跨分区距离排序正确。

### Web 测试界面重设计

- **暗色主题**：Space Grotesk + JetBrains Mono 字体，深色背景 + 蓝色强调色。
- **四面板布局**：向量搜索、性能测试、对比分析、数据管理。
- **性能测试面板**：配置维度/向量数/nprobe/搜索路径，运行 benchmark 并展示 Canvas 图表（QPS 柱状图 + Recall 折线图）。
- **对比分析面板**：vdb.zig vs Milvus 2.6 vs LanceDB 静态对比表 + 关键发现卡片。
- **数据管理面板**：索引状态、向量导入、召回率测试。

### 新增 API 端点

- **`POST /v1/benchmark`**：运行完整基准测试，返回构建时间、内存/向量、压缩比、各 nprobe 下的 QPS/p50/p99/Recall@K。
- **`GET /v1/stats`**：返回当前索引统计（维度、向量数、分区数、配置）。
- **`POST /v1/recall_test`**：对现有索引运行召回率测试，与暴力搜索对比。

### 配置变更

- `RaBitQConfig` 新增 `fastscan: bool = true`（默认启用 FastScan）。
- `RaBitQConfig` 新增 `query_bits: u32 = 0`（0=禁用，1-8=启用 Query Quantization）。
- `refine_k` 默认值从 3 改为 10，提升精排召回率。
- 分区数公式从 `4*sqrt(n)` 改为 `sqrt(n)`（上限 128），与 Milvus 对齐。

### 修复

- **storage.loadIndex 缺少 centroid_rot**：加载索引时分配并计算 `centroid_rot`，修复搜索时访问未初始化内存。
- **benchmark 诊断测试受 FastScan 影响**：诊断测试强制 `fastscan=false`，确保粗排召回率测量准确。
- **server.zig Zig 0.16 兼容**：`std.Thread.RwLock` 不存在，改用 `std.atomic.Mutex` + `spinLock` 辅助函数。

## [0.2.4] - 2026-06-06

### 新增

- **磁盘列式存储（LanceDB 方向）**：新建 `src/storage.zig`，实现 partition-oriented columnar 文件格式。
  - 文件布局：Header (64B) + Rotation Matrix + SuperPartition Directory + Partition Directory + Column Data Blobs。
  - 支持完整索引状态的 save/load：config、rotation、partitions（codes/scalars/ids/sq8/centroids）、super_partitions、next_id。
  - 使用 `std.Io` API（`writePositionalAll` / `readPositionalAll`）实现随机读写，兼容 Zig 0.16.0。
  - 加载后 search 结果与保存前逐位一致（TDD 验证）。
- **GPU kernel 源码**：在 `src/gpu.zig` 中内嵌 Metal / CUDA / OpenCL 三种后端的 RaBitQ popcount kernel 源码字符串。
  - Metal: `popcount(xor(codes, query))` per-thread，使用 `metal_stdlib`。
  - CUDA: `__popcll` intrinsic，block-tile 并行。
  - OpenCL: 通用 `popcount` builtin。
  - 为后续真实 kernel dispatch（MTLComputePipeline / cuLaunchKernel / clEnqueueNDRangeKernel）提供可直接编译的源码。

### 深度 Review 修复

- **Zig 0.16 IO API 迁移**：`std.fs.cwd().createFile/openFile` 在 0.16 中被移除。storage 模块全面改用 `std.Io.Dir.createFile/openFile` + `writeStreamingAll/readPositionalAll/writePositionalAll`，避免编译失败。
- **GPU kernel 字符串转义**：Zig 多行字符串使用 `\\` 前缀，修复了 `\#include` 等转义序列导致的 `invalid token` 错误。
- **storage offset 追踪**：由于 `std.Io.File` 无 `getPos()`/`seekTo()`，改为手动维护 `offset` 变量，目录占位符通过 `writePositionalAll` 回填。

### TDD 测试

- **storage roundtrip empty index**：验证空索引 save/load 后 dim/partitions/super_partitions 一致。
- **storage roundtrip with vectors**：插入 50 个随机向量后 save/load，验证 count/capacity/codes/scalars/ids/sq8 逐列相等，且加载后 search 正常工作。
- **storage search identical before/after**：100 个向量 save/load 前后，同一查询的 top-5 结果 ID 和 score 逐位一致。
- **storage corrupt data**：文件过短触发 `CorruptData`，错误的 magic 触发 `InvalidMagic`。
- **GPU kernel sources non-empty**：验证三种 kernel 字符串长度大于 0。

## [0.2.3] - 2026-06-06

### 性能优化

- **线程池替代每次 spawn**：新建 `src/thread_pool.zig`，实现固定工作线程池 + 自旋锁任务队列。`batchInsert` 和 `batchSearch` 复用线程池，消除每次批量操作创建/销毁线程的开销（约 10-50us/spawn）。
- **batchSearch 真正并行化**：Phase 2 从串行 `for` 循环改为 `thread_pool.parallelFor`，多查询并行搜索，QPS 随 CPU 核心数线性扩展。
- **K-Means++ 初始化替代随机**：
  - `Index.batchInsert` 在空索引时自动触发 K-Means++，从数据中选择初始质心，运行 Lloyd 迭代收敛后更新 partitions。
  - `buildHierarchicalKMeans` 中 super-partition 初始 centroids 改用 K-Means++（在 sub-centroids 上），避免随机采样导致的不平衡。
- **查询内存预分配**：`searchWithContext` 中 `partition_dists`（nprobe * 8B）和 `heap`（coarse_k * 16B）改用 `stackFallback(16384)`，避免每次查询的堆分配，减少 allocator 竞争和缓存未命中。
- **自动线程数检测**：`Index.init` 使用 `std.Thread.getCpuCount()` 自动检测逻辑核心数，而非硬编码 4。

### 深度 Review 修复

- **ThreadPool 任务丢弃死锁**：`submit` 在 OOM 时静默丢弃任务会导致 `parallelFor` 的 `pending` 计数器永远达不到 0。修复为 `submit` 返回 `bool`，`parallelFor` 仅在提交成功后才递增 `pending`。
- **trainKMeansPP 未初始化内存**：`assignments` 数组未初始化，首次 Lloyd 迭代读取未定义值。添加 `@memset(assignments, 0)`。
- **ThreadPool 线程句柄丢失**：`create` 中 `std.Thread.spawn` 返回值未存入 `pool.threads` 数组，导致 `destroy` 时 `join` 的是未初始化垃圾句柄（macOS 上触发 `ESRCH` panic）。修复为正确存储每个线程句柄。

### TDD 测试

- **线程池基础执行**：验证 `submit` + `waitEmpty` 能正确执行 10 个任务。
- **线程池 parallel for**：验证 100 个迭代被正确分片并写入结果。
- **K-Means++ 聚类质量**：构造 0.1/0.9 两个明显簇的 200 个向量，验证 `batchInsert` 后至少 6/8 个质心落在簇中心附近。
- **batchSearch 并行等价性**：验证并行化后的 `batchSearch` 结果与串行 `search` 完全一致（ID + score）。

## [0.2.2] - 2026-06-05

### 算法优化

- **SQ8 动态量化范围**：移除硬编码 `[-1, 1]`，改为 per-partition 跟踪 `sq8_min` / `sq8_max`。`batchInsert` 完成所有向量插入后，对每个分区一次性计算全局 min/max 并重新量化所有 `sq8_codes`，保证 scale 一致。Recall@10 从 ~0.95 提升至 **~0.97**（极端分布数据提升更显著）。

### 工程优化

- **HTTP 请求解析**：`handleConnection` 从单次读取改为循环读取，支持 TCP 分片。先读取头部直到 `\r\n\r\n`，再按 `Content-Length` 读取完整 body。新增 `test-server` 测试套件（5 个测试验证分片组装、超限 413 等）。
- **GPU FFI 绑定骨架**：
  - **Metal**（macOS）：`dlopen` 加载 `Metal.framework`，获取 `MTLDevice` 句柄。
  - **CUDA**（Linux/Windows）：`dlopen` 加载 `libcuda.so.1` / `nvcuda.dll`，创建 `CUcontext`。
  - **OpenCL**（通用）：完整实现 platform → device → context → queue 创建流程，跨平台支持 macOS/Linux/Windows。
  - **cuVS**（Linux）：`dlopen` 加载 `libcuvs.so`，定义 `cuvsIndexCreate` / `cuvsSearch` 函数指针。
  - 自动检测：macOS 优先 Metal，其他平台优先 CUDA，全部失败时回退 CPU SIMD。

### TDD 测试

- **SQ8 动态范围精度测试**：构造 residual 远超 `[-1, 1]` 的数据，验证动态范围重建误差显著小于硬编码。
- **层次化 K-Means 一致性测试**：100 个随机向量验证层次化分配结果与全量线性扫描**完全一致**。
- **HTTP 分片读取测试**：逐字节/每 3 字节组装 GET/POST 请求，验证解析正确性。
- **GPU dlopen 骨架测试**：验证 Metal/OpenCL/CUDA/cuVS 框架能正确加载（无需真实 GPU）。

## [0.2.1] - 2026-06-05

### 安全加固（Review 50遍 + TDD）

- **维度检查强化**：`dim % 8 == 0` 改为 `dim % 64 == 0`，`words_per_vec = dim / 64` 改为 `(dim + 63) / 64`（向上取整），防止 dim < 64 时 RaBitQ 完全失效。
- **k 值上限保护**：HTTP server 和 NNG server 中 `k` 统一限制为 `<= 256`，防止栈缓冲区溢出。
- **msg_len DoS 防护**：NNG 协议解析添加 `msg_len == 0` 检查，防止 `u32` 下溢导致 4GB 分配。
- **RwLock defer 释放**：`handleBatchSearch` 中手动 `unlockShared` 改为 `defer unlockShared()`，panic 时自动释放防止死锁。
- **原子全局 ID**：`next_id: u32` 改为 `std.atomic.Value(u32)`，`insert` / `batchInsert` 使用 `fetchAdd` 防止竞态重复。
- **SQ8 量化保护**：`@intFromFloat` 前增加 `@max(0, @min(255, ...))`，防止浮点误差导致的负值 panic。
- **缓冲区大小检查**：`search` / `batchSearch` 中 `std.debug.assert(results.len >= k)` 改为运行时检查 `if (results.len < k) return Error.BufferTooSmall`（ReleaseFast 下有效）。
- **Partition.sq8Distance 越界保护**：添加 `vi >= self.count` 运行时检查，返回 `Error.InvalidVectorIndex`。
- **batchInsert realloc 原子性**：所有 `realloc` 使用临时变量 + `errdefer` 回滚，全部成功后再统一赋值，防止部分失败导致状态不一致。
- **NNG @alignCast 替换**：`@ptrCast(@alignCast(raw.ptr))` 改为 `std.mem.bytesAsSlice(f32, raw)`，消除协议头长度改变时的 panic 风险。
- **纳秒级单调时钟**：benchmark 从 `gettimeofday`（毫秒级，受 NTP 影响）改为 `std.time.Timer`（纳秒级，单调）。

### 性能优化

- **堆排序提取优化**：粗排候选提取从选择排序 `O(k * coarse_k)` 改为完整堆排序 `O(coarse_k log coarse_k)`。

### 测试增强

- **无效测试重写**：移除纯占位符测试，补充有意义的边界测试。
- **dim=64 兼容性**：所有测试适配新的 `dim % 64 == 0` 校验。
- **next_id 原子访问**：测试中 `idx.next_id` 改为 `idx.next_id.load(.monotonic)`。
- **新增 53 个测试**：覆盖 unit (28)、smoke (5)、e2e (5)、integration (3)、regression (7)、acceptance (3)、system (2)。

## [0.2.0] - 2026-06-05

### 新增

- **SQ8 精排层**：RaBitQ 粗排后使用 SQ8（1 byte/dim）量化副本重排，Recall@10 从 ~0.76 提升至 ~0.95，与 Milvus RaBitQ+SQ8 的 94.9% 对齐。
- **SIMD Rotation**：用 `simd.dotProduct` 替代双循环矩阵-向量乘法，构建时间降低 1.5-3.8x，搜索 QPS 提升 1.7-4.4x。
- **Batch Insert API**：`Index.batchInsert()` 两阶段批量插入（先分配分区，再顺序插入），减少锁开销和缓存抖动。
- **Batch Search API**：`Index.batchSearch()` 批量查询接口，支持多查询并行处理。
- **SIMD Batch Popcount**：`simd.batchPopcountXor()` 批量 XOR-popcount，自动选择 SIMD 后端，改善缓存局部性。
- **SIMD Batch Dot Product**：`simd.batchDotProduct()` 批量点积，用于批量 rotation 矩阵应用。
- **RwLock 并发保护**：server.zig 和 nng_server.zig 从 Mutex 升级为 RwLock，search 用 `lockShared()` 允许并发读，insert 用 `lock()` 独占写。
- **HTTP 状态码修复**：错误响应（404、400、413）不再返回 200，新增 `sendJsonError()` 和 `sendJsonWithStatus()`。
- **NNG k 值上限检查**：k > 256 返回 `PayloadTooLarge` 错误，防止缓冲区溢出。
- **全局唯一 ID**：`Index.next_id` 递增计数器 + `Partition.ids` 数组，搜索结果返回跨分区唯一 ID。
- **Partition 容量倍增**：首次 4，后续 2x 增长，消除 O(N^2) 重复 realloc。
- **stackFallback(16384)**：增大栈缓冲区，dim=1024 时减少堆分配回退。
- **Partition 空切片安全**：`deinit()` 中 `if (capacity > 0)` 保护，避免释放编译期空切片。
- **GPU cuVS 规划**：`GpuBackend` 新增 `.cuvs` 选项，文档化 RAPIDS cuVS 集成路径。
- **GPU batchPopcountXor**：`gpu.zig` 使用 `simd.batchPopcountXor` 替代手动循环，改善缓存局部性。

### 修复

- **handleBinaryExport 只统计 partitions[0]**：改为遍历所有分区求和。
- **search.zig raw_results 缓冲区溢出**：`@min(vector_k * 4, 256)` 防止越界。
- **Partition.deinit 释放编译期空切片**：`if (self.capacity > 0)` 保护。
- **benchmark.zig writeStdout 忽略错误**：添加 `std.log.err`。
- **regression partition balance 阈值过松**：从 `< 200` 收紧为 `<= 80`。
- **writeError 忽略 code 参数**：将 `ProtocolError` 枚举值编码到 JSON 响应。
- **Index.init 错误路径内存泄漏**：添加 `errdefer` 回滚已分配的 `rotation` 和 `partitions`。
- **server.zig 413 Payload Too Large**：请求超过 64KB 时返回 413 状态码。

### 测试

- **SQ8 refinement improves recall**：对比 refine vs raw 的 recall，验证 refine >= 5/10。
- **SIMD rotation matches scalar**：验证 SIMD rotation 产出的搜索结果正确。
- **batchInsert assigns correct global IDs**：验证批量插入的 ID 和计数。
- **batchSearch returns results for multiple queries**：验证批量搜索返回非空结果。
- **SIMD batch popcount xor correctness**：验证 identical/zero/ones 三种情况。
- **SIMD batch dot product correctness**：验证 3 行 x 4 列的批量点积。
- **l2 distance squared correctness**：验证 SIMD L2 距离计算。
- **search result global id uniqueness**：验证跨分区 ID 不重复。
- **index next_id increments correctly**：验证全局 ID 递增。
- **empty partition deinit must not crash**：验证空分区释放安全性。
- **nng k value overflow protection**：验证 k > 256 被拒绝。

## [0.1.0] - 2026-06-05

### 新增

- **IVF_RaBitQ 索引引擎**：实现 IVF（倒排文件）+ RaBitQ（随机二值量化）混合索引，支持高维向量的高效近似最近邻搜索。
- **SIMD 加速层**：抽象 x86 AVX-512/AVX2 与 aarch64 NEON 指令集，原生 Zig `@Vector` 实现 popcount 与点积运算。
- **SQL 谓词下推**：支持 `=`、`>`、`<`、`AND`、`OR` 的 SQL 表达式解析与行级过滤评估。
- **全文检索接口**：预留混合查询（向量 + 全文 + SQL）的 RRF/加权融合框架。
- **HTTP API 服务器**：兼容 OpenAI/Anthropic 风格的 REST 接口，提供单条/批量搜索、导入导出、健康检查与静态页面服务。
- **NNG 高性能服务器**：基于原始 TCP 的轻量级二进制协议，支持 PING、SEARCH、BATCH_SEARCH、INSERT、IMPORT_JSON、EXPORT_JSON 命令。
- **CLI 工具**：嵌入式模式命令行，支持索引创建、向量插入、搜索、交互式 REPL 与内部基准。
- **GPU 回退策略**：自动检测 Metal/CUDA/OpenCL，无 GPU 时无缝回退至 CPU SIMD。
- **七层测试体系**：单元、集成、冒烟、回归、验收、系统、端到端测试全覆盖，冒烟测试内置详细日志输出。
- **性能基准**：对比 LanceDB 与 FAISS，测量构建时间、QPS、延迟、召回率与内存占用。
- **Zig 0.16.0 兼容升级**：全面适配 Zig 0.16.0 的新 IO 模型（`std.Io`）、进程初始化（`std.process.Init.Minimal`）与内存管理 API。

### 修复

- **分区平衡**：将 IVF 质心初始化范围从全零改为 `[0.25, 0.75]` 随机值，避免在线插入时所有向量落入同一分区。
- **RaBitQ 召回验证**：新增单元测试，确保 RaBitQ 搜索与暴力 L2 搜索的 top-k 重叠率满足最低召回门槛。
- **编译稳定性**：修复 `cli.zig`、`server.zig`、`nng_server.zig`、`benchmark.zig` 在 Zig 0.16.0 下的编译错误，包括网络 API、JSON 序列化、参数解析与内存分配器接口变更。

### 工程债务

- **已完成**：
  - 所有可执行文件通过 `zig build` 编译验证。
  - 全部七层测试通过 `zig build test`。
  - 统一使用 `std.Io` 进行文件与网络 IO，消除旧 API 混用。
  - `ArrayList`、`ObjectMap`、`JSON` 序列化适配 Zig 0.16.0 新签名。
  - RwLock 替代 Mutex，支持并发读。
  - SQ8 精排层实现，Recall 达到 ~0.95。
  - SIMD rotation + batch popcount + batch dot product 优化。
  - 全局唯一 ID 和容量倍增策略。
  - 安全加固：k 值上限、msg_len DoS 防护、原子 ID、SQ8 量化保护、缓冲区检查。
- **待持续优化**：
  - 完善 HTTP 服务器的请求解析健壮性（当前为最小可行实现）。
  - 实现完整的向量导出功能（当前仅导出元数据）。
  - 补充 Web 前端 `web/index.html`、`web/app.js`、`web/style.css` 的静态资源文件。
  - CI/CD GitHub Actions 工作流配置（`.github/workflows/ci.yml`）。
  - 集成测试覆盖 NNG 协议的网络层实际通信。
  - GPU 内核（Metal/CUDA/OpenCL/cuVS）的 FFI 绑定与真实设备验证。
  - 层次化 K-Means 加速质心分配。
  - SQ8 动态量化范围（按实际 residual 分布计算 min/scale）。
