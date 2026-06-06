// vdb.zig Web UI — 功能测试与性能测试界面
(function() {
    'use strict';

    // ── Helpers ──
    function $(sel) { return document.querySelector(sel); }
    function $$(sel) { return document.querySelectorAll(sel); }

    function generateVector(dim, seed) {
        const vec = [];
        let x = seed * 12345;
        for (let i = 0; i < dim; i++) {
            x = (x * 16807) % 2147483647;
            vec.push(parseFloat((x / 2147483647).toFixed(6)));
        }
        return vec;
    }

    async function postJson(path, body) {
        const resp = await fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        return resp.json();
    }

    async function getJson(path) {
        const resp = await fetch(path);
        return resp.json();
    }

    function fmtNum(n, decimals) {
        if (typeof n !== 'number') return '—';
        return n.toLocaleString(undefined, { minimumFractionDigits: decimals || 0, maximumFractionDigits: decimals || 0 });
    }

    // ── Tab Navigation ──
    $$('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.nav-btn').forEach(b => b.classList.remove('active'));
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.dataset.tab;
            $(`#tab-${tab}`).classList.add('active');
        });
    });

    // ── Health Check ──
    async function checkHealth() {
        try {
            const data = await getJson('/health');
            $('#status-dot').className = 'status-dot ok';
            $('#status-text').textContent = '已连接';
            return true;
        } catch {
            $('#status-dot').className = 'status-dot err';
            $('#status-text').textContent = '连接失败';
            return false;
        }
    }
    checkHealth();

    // ── Search Tab ──
    function loadExample(dim) {
        const vec = generateVector(dim, 1);
        $('#search-vector').value = JSON.stringify(vec);
    }

    $('#btn-search-example').addEventListener('click', () => loadExample(128));
    $$('.chip').forEach(chip => {
        chip.addEventListener('click', () => loadExample(parseInt(chip.dataset.dim)));
    });

    // Load default example
    loadExample(128);

    $('#btn-search').addEventListener('click', async () => {
        try {
            const vector = JSON.parse($('#search-vector').value);
            const k = parseInt($('#search-k').value);
            const nprobe = parseInt($('#search-nprobe').value);
            const result = await postJson('/v1/search', { vector, k, nprobe });

            const tbody = $('#search-results-table tbody');
            tbody.innerHTML = '';

            if (result.status === 'ok' && result.results) {
                $('#search-count').textContent = result.count;
                $('#search-empty').style.display = 'none';
                $('#search-results-table').style.display = '';

                result.results.forEach((r, i) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${i + 1}</td><td>${r.id}</td><td>${r.partition_id}</td><td>${r.score.toFixed(6)}</td>`;
                    tbody.appendChild(tr);
                });
            } else {
                $('#search-count').textContent = '0';
                $('#search-empty').style.display = '';
                $('#search-results-table').style.display = 'none';
                if (result.message) alert('搜索错误: ' + result.message);
            }
        } catch (e) {
            alert('搜索错误: ' + e.message);
        }
    });

    // ── Benchmark Tab ──
    let benchChart = null;

    $('#btn-benchmark').addEventListener('click', async () => {
        const dim = parseInt($('#bench-dim').value);
        const n = parseInt($('#bench-n').value);
        const queryCount = parseInt($('#bench-qcount').value);
        const k = parseInt($('#bench-k').value);
        const nprobeStr = $('#bench-nprobes').value;
        const nprobeValues = nprobeStr.split(',').map(s => parseInt(s.trim())).filter(x => x > 0);
        const path = $('#bench-path').value;

        // Map path to config
        let fastscan = false, queryBits = 0;
        if (path === 'fastscan') { fastscan = true; }
        else if (path === 'query_quant_4') { queryBits = 4; }
        else if (path === 'query_quant_8') { queryBits = 8; }

        const btn = $('#btn-benchmark');
        btn.disabled = true;
        btn.textContent = '测试中...';
        $('#bench-progress').style.display = '';
        $('#bench-progress-fill').style.width = '30%';

        try {
            const result = await postJson('/v1/benchmark', {
                dim, n, query_count: queryCount, k,
                nprobe_values: nprobeValues,
                fastscan, query_bits: queryBits,
            });

            $('#bench-progress-fill').style.width = '100%';

            if (result.status === 'ok') {
                $('#bench-results-card').style.display = '';

                // Stats
                const statsHtml = `
                    <div class="stat-item"><span class="stat-value">${fmtNum(result.build_time_ms)}</span><span class="stat-label">构建时间 (ms)</span></div>
                    <div class="stat-item"><span class="stat-value">${result.memory_per_vector ? result.memory_per_vector.toFixed(1) : '—'}</span><span class="stat-label">内存/向量 (B)</span></div>
                    <div class="stat-item"><span class="stat-value">${result.compression_ratio ? result.compression_ratio.toFixed(1) + 'x' : '—'}</span><span class="stat-label">压缩比</span></div>
                `;
                $('#bench-stats').innerHTML = statsHtml;

                // Table
                const tbody = $('#bench-results-table tbody');
                tbody.innerHTML = '';
                (result.results || []).forEach(r => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${r.nprobe}</td><td>${fmtNum(r.qps, 0)}</td><td>${r.p50_us ? r.p50_us.toFixed(1) : '—'}</td><td>${r.p99_us ? r.p99_us.toFixed(1) : '—'}</td><td>${r.recall_at_k ? r.recall_at_k.toFixed(3) : '—'}</td>`;
                    tbody.appendChild(tr);
                });

                // Chart
                drawBenchChart(result.results || []);
            } else {
                alert('基准测试错误: ' + (result.message || '未知错误'));
            }
        } catch (e) {
            alert('基准测试错误: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.textContent = '运行基准测试';
            setTimeout(() => { $('#bench-progress').style.display = 'none'; }, 500);
        }
    });

    function drawBenchChart(results) {
        const canvas = $('#bench-chart');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;
        const pad = { top: 30, right: 60, bottom: 40, left: 60 };
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);

        if (results.length === 0) return;

        const nprobes = results.map(r => r.nprobe);
        const qpsValues = results.map(r => r.qps || 0);
        const recallValues = results.map(r => r.recall_at_k || 0);

        const maxQps = Math.max(...qpsValues) * 1.1;
        const maxRecall = 1.0;

        // Grid
        ctx.strokeStyle = '#1e2230';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (plotH / 5) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }

        // QPS bars
        const barW = plotW / results.length * 0.4;
        const gap = plotW / results.length;
        ctx.fillStyle = '#3b82f6';
        results.forEach((r, i) => {
            const x = pad.left + gap * i + (gap - barW * 2 - 4) / 2;
            const barH = (r.qps / maxQps) * plotH;
            ctx.fillRect(x, pad.top + plotH - barH, barW, barH);
        });

        // Recall line
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        results.forEach((r, i) => {
            const x = pad.left + gap * i + gap / 2;
            const y = pad.top + plotH - (r.recall_at_k / maxRecall) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Recall dots
        ctx.fillStyle = '#22c55e';
        results.forEach((r, i) => {
            const x = pad.left + gap * i + gap / 2;
            const y = pad.top + plotH - (r.recall_at_k / maxRecall) * plotH;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        // X axis labels
        ctx.fillStyle = '#6b7394';
        ctx.font = '11px JetBrains Mono';
        ctx.textAlign = 'center';
        results.forEach((r, i) => {
            const x = pad.left + gap * i + gap / 2;
            ctx.fillText('np=' + r.nprobe, x, h - pad.bottom + 20);
        });

        // Y axis labels (QPS)
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (plotH / 5) * i;
            const val = maxQps * (1 - i / 5);
            ctx.fillText(fmtNum(Math.round(val)), pad.left - 8, y + 4);
        }

        // Y axis labels (Recall) - right side
        ctx.textAlign = 'left';
        ctx.fillStyle = '#22c55e';
        for (let i = 0; i <= 5; i++) {
            const y = pad.top + (plotH / 5) * i;
            const val = maxRecall * (1 - i / 5);
            ctx.fillText(val.toFixed(2), w - pad.right + 8, y + 4);
        }

        // Legend
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(pad.left, 8, 12, 12);
        ctx.fillStyle = '#e4e7ef';
        ctx.font = '11px Space Grotesk';
        ctx.textAlign = 'left';
        ctx.fillText('QPS', pad.left + 18, 18);

        ctx.fillStyle = '#22c55e';
        ctx.beginPath();
        ctx.arc(pad.left + 70, 14, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#e4e7ef';
        ctx.fillText('Recall@K', pad.left + 80, 18);
    }

    // ── Data Tab ──
    $('#btn-stats').addEventListener('click', async () => {
        try {
            const data = await getJson('/v1/stats');
            if (data.status === 'ok') {
                $('#stat-dim').textContent = data.dimension;
                $('#stat-vectors').textContent = fmtNum(data.total_vectors);
                $('#stat-partitions').textContent = data.num_partitions;
                $('#stat-fastscan').textContent = data.config?.fastscan ? '✅' : '❌';
                $('#stat-query-bits').textContent = data.config?.query_bits || '0';
                $('#stat-refine').textContent = data.config?.refine_sq8 ? '✅' : '❌';
            }
        } catch (e) {
            alert('获取状态失败: ' + e.message);
        }
    });

    // Auto-load stats on tab switch
    const statsObserver = new MutationObserver(() => {
        if ($('#tab-data').classList.contains('active')) {
            $('#btn-stats').click();
        }
    });
    statsObserver.observe($('#tab-data'), { attributes: true, attributeFilter: ['class'] });

    $('#btn-import').addEventListener('click', async () => {
        try {
            const vectors = JSON.parse($('#import-data').value);
            const result = await postJson('/v1/import', { vectors });
            const el = $('#import-result');
            el.textContent = JSON.stringify(result, null, 2);
            el.style.display = 'block';
        } catch (e) {
            const el = $('#import-result');
            el.textContent = 'Error: ' + e.message;
            el.style.display = 'block';
        }
    });

    $('#btn-recall').addEventListener('click', async () => {
        try {
            const queryCount = parseInt($('#recall-qcount').value);
            const k = parseInt($('#recall-k').value);
            const nprobe = parseInt($('#recall-nprobe').value);
            const result = await postJson('/v1/recall_test', { query_count: queryCount, k, nprobe });
            const el = $('#recall-result');
            el.textContent = JSON.stringify(result, null, 2);
            el.style.display = 'block';
        } catch (e) {
            const el = $('#recall-result');
            el.textContent = 'Error: ' + e.message;
            el.style.display = 'block';
        }
    });

})();
