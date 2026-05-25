/**
 * HRP Benchmark — C implementation
 * Compile: gcc -O3 -o hrp_bench bench.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// ── Timing ──

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// ── Generate synthetic prices ──

static void generate_prices(double *prices, int n, int days) {
    unsigned long long seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 6364136223846793005ULL + 1;
        double start = 100.0 + (seed % 900);
        prices[i * days] = start;
        double vol = 0.01 + (seed % 50) * 0.001;
        for (int t = 1; t < days; t++) {
            seed = seed * 6364136223846793005ULL + 1;
            double u = ((double)seed / (double)0xFFFFFFFFFFFFFFFFULL) - 0.5;
            double ret = vol * u * 0.816;
            prices[i * days + t] = prices[i * days + t - 1] * exp(ret);
        }
    }
}

// ── Log returns ──

static void log_returns(const double *prices, double *rets, int n, int days) {
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < days - 1; t++) {
            rets[i * (days - 1) + t] = log(prices[i * days + t + 1] / prices[i * days + t]);
        }
    }
}

// ── Covariance matrix ──

static void cov_matrix(const double *rets, double *cov, int n, int T) {
    double *means = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < T; t++) means[i] += rets[i * T + t];
        means[i] /= T;
    }
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double s = 0;
            for (int t = 0; t < T; t++)
                s += (rets[i * T + t] - means[i]) * (rets[j * T + t] - means[j]);
            cov[i * n + j] = cov[j * n + i] = s / (T - 1);
        }
    }
    free(means);
}

// ── Correlation & Distance ──

static void corr_matrix(const double *cov, double *corr, int n) {
    double *stds = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) stds[i] = sqrt(cov[i * n + i]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            corr[i * n + j] = (stds[i] > 0 && stds[j] > 0) ? cov[i * n + j] / (stds[i] * stds[j]) : 0;
    free(stds);
}

static void dist_matrix(const double *corr, double *dist, int n) {
    for (int i = 0; i < n * n; i++) {
        double v = (1.0 - corr[i]) / 2.0;
        dist[i] = sqrt(v > 0 ? v : 0);
    }
}

// ── Average linkage — O(n^2) nearest-neighbor chain (Müllner 2011) ──
// Same algorithm SciPy uses for method='average'. Builds raw merges via the
// NN-chain, then reproduces SciPy's post-processing: sort merges by distance
// and relabel cluster ids with a union-find, yielding a standard linkage matrix.

typedef struct { int i, j; double dist; int size; } LinkageRow;

static int uf_find(int *parent, int x) {
    int root = x;
    while (parent[root] != root) root = parent[root];
    while (parent[x] != root) { int nx = parent[x]; parent[x] = root; x = nx; }
    return root;
}

static int cmp_link(const void *a, const void *b) {
    double da = ((const LinkageRow *)a)->dist, db = ((const LinkageRow *)b)->dist;
    return (da > db) - (da < db);
}

static void average_linkage(const double *dist_in, LinkageRow *Z, int n) {
    if (n < 2) return;
    double *D = (double *)malloc((size_t)n * n * sizeof(double));
    memcpy(D, dist_in, (size_t)n * n * sizeof(double));
    int *size = (int *)malloc(n * sizeof(int));
    bool *active = (bool *)malloc(n * sizeof(bool));
    for (int i = 0; i < n; i++) { size[i] = 1; active[i] = true; }

    int *chain = (int *)malloc(n * sizeof(int));
    int clen = 0;
    LinkageRow *raw = (LinkageRow *)malloc((size_t)(n - 1) * sizeof(LinkageRow));
    int m = 0;

    for (int s = 0; s < n - 1; s++) {
        if (clen == 0) {
            int start = -1;
            for (int i = 0; i < n; i++) if (active[i]) { start = i; break; }
            chain[clen++] = start;
        }
        int a, b = -1;
        double mind;
        for (;;) {
            a = chain[clen - 1];
            b = -1; mind = 1e18;
            if (clen >= 2) { b = chain[clen - 2]; mind = D[(size_t)a * n + b]; }
            for (int x = 0; x < n; x++) {
                if (!active[x] || x == a) continue;
                double d = D[(size_t)a * n + x];
                if (d < mind) { mind = d; b = x; }
            }
            if (clen >= 2 && b == chain[clen - 2]) break;  // reciprocal NN
            chain[clen++] = b;
        }
        clen -= 2;  // pop the reciprocal pair a, b
        int x = a < b ? a : b, y = a < b ? b : a;
        int ns = size[x] + size[y];
        raw[m].i = x; raw[m].j = y; raw[m].dist = mind; raw[m].size = ns; m++;
        // Lance-Williams average update; the merged cluster lives at index y.
        for (int k = 0; k < n; k++) {
            if (!active[k] || k == x || k == y) continue;
            double nd = (size[x] * D[(size_t)x * n + k] + size[y] * D[(size_t)y * n + k]) / ns;
            D[(size_t)y * n + k] = nd; D[(size_t)k * n + y] = nd;
        }
        size[y] = ns; active[x] = false;
    }

    // SciPy post-processing: sort by distance, relabel via union-find.
    qsort(raw, n - 1, sizeof(LinkageRow), cmp_link);
    int *parent = (int *)malloc((size_t)(2 * n) * sizeof(int));
    int *usize = (int *)malloc((size_t)(2 * n) * sizeof(int));
    for (int i = 0; i < 2 * n; i++) { parent[i] = i; usize[i] = 1; }
    int next = n;
    for (int k = 0; k < n - 1; k++) {
        int xr = uf_find(parent, raw[k].i), yr = uf_find(parent, raw[k].j);
        int lo = xr < yr ? xr : yr, hi = xr < yr ? yr : xr;
        Z[k].i = lo; Z[k].j = hi; Z[k].dist = raw[k].dist;
        Z[k].size = usize[xr] + usize[yr];
        parent[xr] = next; parent[yr] = next; usize[next] = usize[xr] + usize[yr];
        next++;
    }
    free(D); free(size); free(active); free(chain); free(raw); free(parent); free(usize);
}

// ── Leaf order (iterative) ──

static void leaf_order(const LinkageRow *Z, int n, int *order, int *out_len) {
    int *stack = (int *)malloc(2 * n * sizeof(int));
    int sp = 0, cnt = 0;
    stack[sp++] = n + (n - 2); // root
    while (sp > 0) {
        int node = stack[--sp];
        if (node < n) { order[cnt++] = node; continue; }
        const LinkageRow *r = &Z[node - n];
        stack[sp++] = r->j;
        stack[sp++] = r->i;
    }
    *out_len = cnt;
    free(stack);
}

// ── Quasi-diag ──

static void quasi_diag(const double *cov, const int *order, double *qd, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            qd[i * n + j] = cov[order[i] * n + order[j]];
}

// ── HRP weights ──

static double cluster_var(const double *cov, const int *idx, int m, int n) {
    double v = 0;
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++)
            v += cov[idx[a] * n + idx[b]];
    return v / ((double)m * m);
}

static void hrp_weights(const double *covQ, int n, double *w) {
    for (int i = 0; i < n; i++) w[i] = 1.0;

    // BFS-style bisection
    typedef struct { int *idx; int len; } Segment;
    Segment *queue = (Segment *)malloc(2 * n * sizeof(Segment));
    int qh = 0, qt = 0;
    int *all = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) all[i] = i;
    queue[qt++] = (Segment){ all, n };

    while (qh < qt) {
        Segment seg = queue[qh++];
        if (seg.len <= 1) { free(seg.idx); continue; }
        int mid = seg.len / 2;

        double vL = cluster_var(covQ, seg.idx, mid, n);
        double vR = cluster_var(covQ, seg.idx + mid, seg.len - mid, n);
        double alpha = (1.0 / vL) / (1.0 / vL + 1.0 / vR);

        for (int i = 0; i < mid; i++) w[seg.idx[i]] *= alpha;
        for (int i = mid; i < seg.len; i++) w[seg.idx[i]] *= (1.0 - alpha);

        int *left = (int *)malloc(mid * sizeof(int));
        int *right = (int *)malloc((seg.len - mid) * sizeof(int));
        memcpy(left, seg.idx, mid * sizeof(int));
        memcpy(right, seg.idx + mid, (seg.len - mid) * sizeof(int));

        queue[qt++] = (Segment){ left, mid };
        queue[qt++] = (Segment){ right, seg.len - mid };
        free(seg.idx);
    }
    free(queue);

    double sum = 0;
    for (int i = 0; i < n; i++) sum += w[i];
    for (int i = 0; i < n; i++) w[i] /= sum;
}

// ── Format time ──

static void fmt_time(double us, char *buf) {
    if (us < 1000) sprintf(buf, "%6.0fµs", us);
    else if (us < 1e6) sprintf(buf, "%6.1fms", us / 1e3);
    else sprintf(buf, "%6.2fs ", us / 1e6);
}

// ── Main ──

static void bench(int n, int days) {
    double *prices = (double *)malloc(n * days * sizeof(double));
    generate_prices(prices, n, days);

    int T = days - 1;
    double *rets = (double *)malloc(n * T * sizeof(double));
    double t0 = now_us();
    log_returns(prices, rets, n, days);
    double t_ret = now_us() - t0;

    double *cov = (double *)malloc(n * n * sizeof(double));
    t0 = now_us();
    cov_matrix(rets, cov, n, T);
    double t_cov = now_us() - t0;

    double *corr = (double *)malloc(n * n * sizeof(double));
    corr_matrix(cov, corr, n);

    double *dist = (double *)malloc(n * n * sizeof(double));
    dist_matrix(corr, dist, n);

    LinkageRow *Z = (LinkageRow *)malloc((n - 1) * sizeof(LinkageRow));
    t0 = now_us();
    average_linkage(dist, Z, n);
    double t_link = now_us() - t0;

    int *order = (int *)malloc(n * sizeof(int));
    int olen;
    leaf_order(Z, n, order, &olen);

    double *qd = (double *)malloc(n * n * sizeof(double));
    t0 = now_us();
    quasi_diag(cov, order, qd, n);
    double t_qd = now_us() - t0;

    double *w = (double *)malloc(n * sizeof(double));
    t0 = now_us();
    hrp_weights(qd, n, w);
    double t_w = now_us() - t0;

    double total = t_ret + t_cov + t_link + t_qd + t_w;

    char s1[32], s2[32], s3[32], s4[32], s5[32], s6[32];
    fmt_time(t_ret, s1); fmt_time(t_cov, s2); fmt_time(t_link, s3);
    fmt_time(t_qd, s4); fmt_time(t_w, s5); fmt_time(total, s6);
    printf("  %6d │ %s │ %s │ %s │ %s │ %s │ %s\n", n, s1, s2, s3, s4, s5, s6);

    if (n == 10) {
        double sum = 0; for (int i = 0; i < n; i++) sum += w[i];
        fprintf(stderr, "verify N=10: w0=%.12f w1=%.12f w2=%.12f sum=%.6f\n", w[0], w[1], w[2], sum);
    }

    free(prices); free(rets); free(cov); free(corr);
    free(dist); free(Z); free(order); free(qd); free(w);
}

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║          HRP Benchmark — C (gcc -O3)                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    printf("  365 daily observations per asset\n\n");
    printf("  %6s │ %8s │ %8s │ %8s │ %8s │ %8s │ %8s\n",
        "N", "LogRet", "Cov", "Linkage", "QuasiD", "Weights", "TOTAL");
    printf("  ───────────────────────────────────────────────────────────────────\n");

    int sizes[] = {10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
    for (int i = 0; i < 10; i++) bench(sizes[i], 365);
    return 0;
}
