const std = @import("std");

/// A simple fixed-size worker thread pool with a shared task queue.
/// Uses spin-lock for queue protection and yield-based waiting.
/// Designed for batch parallelization (insert/search) to avoid
/// the overhead of spawning threads per operation.
pub const ThreadPool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    queue: std.ArrayList(Task),
    mutex: std.atomic.Mutex,
    active: std.atomic.Value(usize),
    shutdown: std.atomic.Value(bool),

    pub const Task = struct {
        func: *const fn (?*anyopaque) void,
        ctx: ?*anyopaque,
    };

    /// Create a thread pool on the heap so the returned pointer
    /// remains stable for worker threads.
    pub fn create(allocator: std.mem.Allocator, num_threads: usize) !*ThreadPool {
        const pool = try allocator.create(ThreadPool);
        errdefer allocator.destroy(pool);

        pool.allocator = allocator;
        pool.threads = try allocator.alloc(std.Thread, num_threads);
        errdefer allocator.free(pool.threads);

        pool.queue = std.ArrayList(Task).empty;
        pool.mutex = .unlocked;
        pool.active = std.atomic.Value(usize).init(0);
        pool.shutdown = std.atomic.Value(bool).init(false);

        for (0..num_threads) |i| {
            pool.threads[i] = try std.Thread.spawn(.{}, workerLoop, .{pool});
        }

        return pool;
    }

    pub fn destroy(self: *ThreadPool) void {
        self.shutdown.store(true, .release);

        for (self.threads) |t| {
            t.join();
        }
        self.allocator.free(self.threads);
        self.queue.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn submit(self: *ThreadPool, task: Task) bool {
        spinLock(&self.mutex);
        self.queue.append(self.allocator, task) catch {
            self.mutex.unlock();
            return false; // OOM
        };
        self.mutex.unlock();
        return true;
    }

    /// Wait until the queue is empty and no tasks are currently executing.
    pub fn waitEmpty(self: *ThreadPool) void {
        while (true) {
            spinLock(&self.mutex);
            const empty = self.queue.items.len == 0 and self.active.load(.acquire) == 0;
            self.mutex.unlock();
            if (empty) break;
            std.Thread.yield() catch {};
        }
    }

    /// Parallel for: split `len` iterations among worker threads.
    /// `func(context, start, end)` is called for each slice.
    pub fn parallelFor(
        self: *ThreadPool,
        len: usize,
        context: anytype,
        comptime func: fn (@TypeOf(context), usize, usize) void,
    ) void {
        if (len == 0) return;

        const num_threads = self.threads.len;
        if (len <= 1 or num_threads == 0) {
            func(context, 0, len);
            return;
        }

        const per_thread = (len + num_threads - 1) / num_threads;

        const TaskCtx = struct {
            context: @TypeOf(context),
            start: usize,
            end: usize,
            pending: *std.atomic.Value(usize),
        };

        var contexts = self.allocator.alloc(TaskCtx, num_threads) catch {
            // OOM fallback: sequential execution
            func(context, 0, len);
            return;
        };
        defer self.allocator.free(contexts);

        var pending = std.atomic.Value(usize).init(0);

        var submitted: usize = 0;
        for (0..num_threads) |i| {
            const start = i * per_thread;
            const end = @min(start + per_thread, len);
            if (start >= end) break;

            contexts[submitted] = .{
                .context = context,
                .start = start,
                .end = end,
                .pending = &pending,
            };

            const submitted_ok = self.submit(.{
                .func = struct {
                    fn run(ptr: ?*anyopaque) void {
                        const c: *TaskCtx = @ptrCast(@alignCast(ptr));
                        func(c.context, c.start, c.end);
                        _ = c.pending.fetchSub(1, .release);
                    }
                }.run,
                .ctx = &contexts[submitted],
            });
            if (submitted_ok) {
                _ = pending.fetchAdd(1, .monotonic);
                submitted += 1;
            }
        }

        if (submitted == 0) {
            func(context, 0, len);
            return;
        }

        while (pending.load(.acquire) > 0) {
            std.Thread.yield() catch {};
        }
    }

    fn workerLoop(pool: *ThreadPool) void {
        while (!pool.shutdown.load(.acquire)) {
            spinLock(&pool.mutex);
            if (pool.queue.items.len > 0) {
                const task = pool.queue.pop().?;
                _ = pool.active.fetchAdd(1, .monotonic);
                pool.mutex.unlock();

                task.func(task.ctx);

                _ = pool.active.fetchSub(1, .release);
            } else {
                pool.mutex.unlock();
                std.Thread.yield() catch {};
            }
        }
    }

    fn spinLock(m: *std.atomic.Mutex) void {
        while (!m.tryLock()) {
            std.atomic.spinLoopHint();
        }
    }
};
