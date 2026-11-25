// adaptive-termination-repo
// Copyright (c) 2026-present NAVER Corp.
// CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
//
// Benchmark driver for adaptive termination in partitioned (hash-sharded) HNSW.
//
// Why C++: distance-computation counts (ndis) can be obtained from a reimplementation, but latency,
// tail latency and CPU time have to be measured with real clocks on the real library (patched Faiss).
//
// Three safeguards decide "can this run be trusted" while measuring, not afterwards.
//   1) CPU model check (opt-in, --expect_cpu). A generation label alone lets a different SKU slip in
//      silently. The paper's runs used --expect_cpu "EPYC 9354".
//   2) Throttle gate. If cgroup v2 cpu.stat nr_throttled grew during a repetition, that repetition is
//      rejected (accept=false). Skipped when cpu.stat has no throttle counters.
//   3) Noise floor. A fixed kernel is timed around every repetition and its drift is recorded.
//      Differences between arms smaller than this floor mean "no difference".
//
// Subcommands
//   build  : fbin -> HNSW index (one shard)
//   add    : insert more rows into a built HNSW index (held-out calibration rows, after calibration)
//   gt     : fbin -> exact top-k ground truth (IndexFlatL2)
//   search : measurement. Searches each of the S shards and merges, reporting recall, ndis, latency
//   oracle : ndis if each shard stopped the instant its last needed true neighbour arrived (a lower bound)
//   ranks  : measures w_j and p_j of the rank-weighted decomposition (theory check)
//   mask   : filter mask generation (random / cluster)
//   align  : per-query alignment between a filter mask and the query
//
// Filters: with --filter on gt/search, answers are searched only inside that mask. This reproduces a
// hybrid query whose inverted-index condition restricts the ANN search. (Not used by the paper.)
//
// Build: bench/build_bench.sh

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/index_io.h>

namespace {

// ---------------------------------------------------------------- environment checks

std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto p = line.find(':');
            if (p == std::string::npos) break;
            std::string v = line.substr(p + 1);
            size_t a = v.find_first_not_of(" \t");
            size_t b = v.find_last_not_of(" \t\r\n");
            return (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
        }
    }
    return "";
}

bool cpu_has(const char* flag) {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("flags", 0) == 0) {
            return line.find(std::string(" ") + flag + " ") != std::string::npos ||
                   line.find(std::string(" ") + flag) == line.size() - strlen(flag) - 1;
        }
    }
    return false;
}

/// CPU time actually used by this thread. Unlike wall time it does not count time spent descheduled
/// or waiting on page faults. A cpu/wall ratio far from 1 is itself a contamination signal.
double thread_cpu_ms() {
    timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

/// Wall clock. That steady_clock is CLOCK_MONOTONIC is only an implementation promise; the output
/// names the clock, so the clock called and the clock named must be the same. Call it directly.
double wall_ms() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

/// cpu.stat of this process's cgroup v2. Inside a container (cgroup namespace) this is
/// /sys/fs/cgroup/cpu.stat; on a host the process's own cgroup is taken from /proc/self/cgroup
/// ("0::<path>"). Falls back to /sys/fs/cgroup/cpu.stat. If the file is missing (cgroup v1, macOS),
/// the readers below return -1 and the measurement still runs.
std::string cgroup_cpu_stat_path() {
    static std::string path;
    if (!path.empty()) return path;
    path = "/sys/fs/cgroup/cpu.stat";
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("0::", 0) == 0) {
            std::string rel = line.substr(3);
            if (!rel.empty() && rel != "/") {
                std::string p = "/sys/fs/cgroup" + rel + "/cpu.stat";
                struct stat st;
                if (::stat(p.c_str(), &st) == 0) path = p;
            }
            break;
        }
    }
    return path;
}

/// Total CPU usage of the cgroup (whole container). Also catches what the per-thread clock misses
/// (kernel work etc.). -1 when unavailable.
double cgroup_cpu_ms() {
    std::ifstream f(cgroup_cpu_stat_path());
    std::string k;
    long long v;
    while (f >> k >> v)
        if (k == "usage_usec") return double(v) / 1e3;
    return -1;
}

/// cgroup v2 throttle counters. Read before and after each repetition; only repetitions whose
/// increase is 0 are accepted. Fields stay -1 when unavailable (no cpu controller / no cgroup v2).
struct Throttle {
    long long nr_periods = -1, nr_throttled = -1, throttled_usec = -1;
    static Throttle read() {
        Throttle t;
        std::ifstream f(cgroup_cpu_stat_path());
        std::string k;
        long long v;
        while (f >> k >> v) {
            if (k == "nr_periods") t.nr_periods = v;
            else if (k == "nr_throttled") t.nr_throttled = v;
            else if (k == "throttled_usec") t.throttled_usec = v;
        }
        return t;
    }
};

/// Noise floor. It always does the same work regardless of the data, so its variation is the
/// variation of the environment.
/// **It is a random-access kernel.** HNSW base-layer search is memory-latency bound pointer chasing
/// over gigabytes (arithmetic is under 10% of the ~250 ns per distance), so a compute kernel that fits
/// in L2 would not see the contention this workload sees.
double noise_floor_ms() {
    constexpr size_t GB = 2;
    constexpr size_t N = GB * (1u << 30) / sizeof(float);
    static std::vector<float> buf;
    if (buf.empty()) {
        buf.resize(N);
        for (size_t i = 0; i < N; i += 4096) buf[i] = float(i & 0xffff);   // actually fault the pages in
    }
    static uint64_t seed = 88172645463325252ULL;
    const size_t lines = buf.size() / 16;      // 64-byte lines
    static volatile double sink;
    auto pass = [&]() {
        double acc = 0;
        for (size_t i = 0; i < 400000; i++) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            acc += buf[(seed >> 20) % lines * 16];
        }
        sink = acc;  // keep the loop from being optimized away
    };
    // Run once before timing. Right after a measurement the multi-GB index has swept through the TLB
    // and caches and the ruler's buffer is cold; timing it as is would misread "the measurement
    // cooled the ruler" as an environment change.
    pass();
    double t0 = wall_ms();
    pass();
    return wall_ms() - t0;
}

// ---------------------------------------------------------------- fbin I/O
// Format: int32 npts, int32 dim, then float32 rows

struct Fbin {
    int64_t n = 0, d = 0;
    std::vector<float> x;
};

Fbin read_fbin(const std::string& path, int64_t limit = -1, int64_t off = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    int32_t n32, d32;
    f.read((char*)&n32, 4);
    f.read((char*)&d32, 4);
    Fbin r;
    int64_t avail = int64_t(n32) - off;
    r.n = (limit > 0 && limit < avail) ? limit : avail;
    r.d = d32;
    r.x.resize(size_t(r.n) * r.d);
    f.seekg(8 + off * int64_t(d32) * 4, std::ios::beg);      // skip the first off rows (e.g. calibration queries start at #1000)
    f.read((char*)r.x.data(), std::streamsize(r.x.size() * sizeof(float)));
    if (!f) throw std::runtime_error("short read: " + path);
    return r;
}

/// Ground truth is stored as an int32 matrix (npts, k, int32 neighbour ids)
void write_ivec(const std::string& path, const std::vector<int32_t>& v, int32_t n, int32_t k) {
    std::ofstream f(path, std::ios::binary);
    f.write((char*)&n, 4);
    f.write((char*)&k, 4);
    f.write((char*)v.data(), std::streamsize(v.size() * sizeof(int32_t)));
}

std::vector<int32_t> read_ivec(const std::string& path, int32_t& n, int32_t& k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    f.read((char*)&n, 4);
    f.read((char*)&k, 4);
    std::vector<int32_t> v(size_t(n) * k);
    f.read((char*)v.data(), std::streamsize(v.size() * sizeof(int32_t)));
    return v;
}

/// Ground truth row i belongs to query qoff+i, so the file must have at least nq rows and K columns.
/// Reading past it would silently use garbage as ground truth.
void check_gt_shape(int32_t gn, int32_t gk, int64_t nq, int K) {
    if (int64_t(gn) < nq || gk < K)
        throw std::runtime_error("ground truth too small: " + std::to_string(gn) + " x " +
                                 std::to_string(gk) + ", need " + std::to_string(nq) + " x " +
                                 std::to_string(K));
}

/// Angular-distance datasets are normalized and then measured with L2.
/// The shards are already normalized by the shard-split step, so ground truth and queries must match.
void normalize_rows(Fbin& f) {
    for (int64_t i = 0; i < f.n; i++) {
        float* r = f.x.data() + i * f.d;
        double s = 0;
        for (int64_t j = 0; j < f.d; j++) s += double(r[j]) * r[j];
        float inv = 1.0f / std::max(float(std::sqrt(s)), 1e-12f);
        for (int64_t j = 0; j < f.d; j++) r[j] *= inv;
    }
}

bool exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

// ---------------------------------------------------------------- arguments

struct Args {
    std::unordered_map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& dflt = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? dflt : it->second;
    }
    int geti(const std::string& k, int dflt) const {
        auto s = get(k);
        return s.empty() ? dflt : std::stoi(s);
    }
    double getd(const std::string& k, double dflt) const {
        auto s = get(k);
        return s.empty() ? dflt : std::stod(s);
    }
};

Args parse(int argc, char** argv, int from) {
    Args a;
    for (int i = from; i < argc; i++) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) continue;
        auto eq = s.find('=');
        if (eq != std::string::npos) a.kv[s.substr(2, eq - 2)] = s.substr(eq + 1);
        else if (i + 1 < argc) a.kv[s.substr(2)] = argv[++i];
        else a.kv[s.substr(2)] = "1";
    }
    return a;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double i = q * (double(v.size()) - 1);
    size_t lo = size_t(i);
    size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (i - double(lo)) * (v[hi] - v[lo]);
}

// ---------------------------------------------------------------- subcommands

int cmd_build(const Args& a) {
    auto base = a.get("base");
    auto out = a.get("out");
    int M = a.geti("M", 40), efc = a.geti("efc", 200);
    int64_t limit = a.geti("limit", -1);
    auto X = read_fbin(base, limit);
    if (a.geti("normalize", 0)) normalize_rows(X);      // not needed if the shard split already normalized
    fprintf(stderr, "[build] %s n=%lld d=%lld M=%d efC=%d\n", base.c_str(), (long long)X.n, (long long)X.d, M, efc);
    faiss::IndexHNSWFlat idx(int(X.d), M);
    idx.hnsw.efConstruction = efc;
    auto t0 = std::chrono::steady_clock::now();
    idx.add(X.n, X.x.data());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    faiss::write_index(&idx, out.c_str());
    fprintf(stderr, "[build] %.1fs -> %s\n", s, out.c_str());
    printf("{\"build_s\": %.3f, \"n\": %lld, \"d\": %lld, \"M\": %d, \"efc\": %d}\n", s, (long long)X.n, (long long)X.d, M, efc);
    return 0;
}

/// Appends rows to a built index (incremental HNSW insertion). Used for the held-out calibration rows: they are
/// searched against the partial graph first and inserted afterwards, giving the graph that is served.
int cmd_add(const Args& a) {
    std::unique_ptr<faiss::Index> idx(faiss::read_index(a.get("index").c_str()));
    auto* h = dynamic_cast<faiss::IndexHNSWFlat*>(idx.get());
    FAISS_THROW_IF_NOT_MSG(h, "add expects an IndexHNSWFlat");
    auto X = read_fbin(a.get("base"));
    FAISS_THROW_IF_NOT_MSG(X.d == h->d, "index and base dimensions differ");
    if (a.geti("normalize", 0)) normalize_rows(X);      // same rule as build
    const long long n0 = h->ntotal;
    fprintf(stderr, "[add] %s + %lld rows of %s (efC=%d)\n", a.get("index").c_str(), (long long)X.n,
            a.get("base").c_str(), h->hnsw.efConstruction);
    auto t0 = std::chrono::steady_clock::now();
    h->add(X.n, X.x.data());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    faiss::write_index(h, a.get("out").c_str());
    fprintf(stderr, "[add] %.1fs -> %s\n", s, a.get("out").c_str());
    printf("{\"add_s\": %.3f, \"n_before\": %lld, \"n_added\": %lld, \"d\": %lld, \"M\": %d, \"efc\": %d}\n", s, n0,
           (long long)X.n, (long long)X.d, h->hnsw.nb_neighbors(1), h->hnsw.efConstruction);
    return 0;
}

int cmd_gt(const Args& a) {
    auto Q = read_fbin(a.get("query"), a.geti("nq", -1), a.geti("qoff", 0));
    if (a.geti("normalize", 0)) normalize_rows(Q);
    int k = a.geti("k", 300);
    int64_t limit = a.geti("limit", -1);

    // With a filter, the ground truth must be computed **among the passing points**. Measuring
    // against unfiltered ground truth would count points the filter removed as "missed" and make
    // recall meaningless.
    std::unordered_set<int32_t> pass;
    bool use_filter = !a.get("filter").empty();
    if (use_filter) {
        int32_t fn, fone;
        auto f = read_ivec(a.get("filter"), fn, fone);
        pass.insert(f.begin(), f.end());
        fprintf(stderr, "[gt] filter: %zu ids\n", pass.size());
    }

    // Compute the ground truth in chunks. Loading the whole base into RAM and letting IndexFlatL2
    // copy it again reaches 80 GB at N=50M, d=200. Exhaustive search per block while keeping a
    // per-query top-k heap keeps memory constant in the block size.
    const int64_t CH = a.geti("chunk", 2000000);
    int64_t n_file, d;
    {
        std::ifstream f(a.get("base"), std::ios::binary);
        int32_t n32, d32;
        f.read(reinterpret_cast<char*>(&n32), 4);
        f.read(reinterpret_cast<char*>(&d32), 4);
        n_file = n32; d = d32;
    }
    const int64_t N = (limit > 0 && limit < n_file) ? limit : n_file;
    FAISS_THROW_IF_NOT_MSG(d == Q.d, "base and query dimensions differ");

    // per query, a max-heap of k (distance, global id) pairs
    std::vector<std::vector<std::pair<float, int32_t>>> heap(Q.n);
    for (auto& h : heap) h.reserve(k + 1);

    std::ifstream bf(a.get("base"), std::ios::binary);
    bf.seekg(8);
    std::vector<float> X(size_t(CH) * d);
    std::vector<float> D(size_t(Q.n) * k);
    std::vector<int64_t> I(size_t(Q.n) * k);
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t off = 0; off < N; off += CH) {
        const int64_t m = std::min(CH, N - off);
        bf.read(reinterpret_cast<char*>(X.data()), m * d * 4);
        if (a.geti("normalize", 0)) {
            for (int64_t i = 0; i < m; i++) {
                double t = 0;
                for (int64_t j = 0; j < d; j++) t += double(X[i * d + j]) * X[i * d + j];
                const float inv = t > 0 ? float(1.0 / std::sqrt(t)) : 1.0f;
                for (int64_t j = 0; j < d; j++) X[i * d + j] *= inv;
            }
        }
        faiss::IndexFlatL2 flat{int(d)};
        flat.add(m, X.data());
        // pick only the local ids in this block that pass the filter
        std::vector<int64_t> loc;
        std::unique_ptr<faiss::IDSelectorBatch> bsel;
        faiss::SearchParameters bsp;
        if (use_filter) {
            for (int64_t i = 0; i < m; i++) if (pass.count(int32_t(off + i))) loc.push_back(i);
            if (loc.empty()) continue;
            bsel = std::make_unique<faiss::IDSelectorBatch>(loc.size(), loc.data());
            bsp.sel = bsel.get();
        }
        flat.search(Q.n, Q.x.data(), k, D.data(), I.data(), use_filter ? &bsp : nullptr);
        for (int64_t qi = 0; qi < Q.n; qi++) {
            auto& h = heap[qi];
            for (int j = 0; j < k; j++) {
                const int64_t id = I[size_t(qi) * k + j];
                if (id < 0) continue;
                const float dd = D[size_t(qi) * k + j];
                if (int(h.size()) < k) {
                    h.emplace_back(dd, int32_t(off + id));
                    std::push_heap(h.begin(), h.end());
                } else if (dd < h.front().first) {
                    std::pop_heap(h.begin(), h.end());
                    h.back() = {dd, int32_t(off + id)};
                    std::push_heap(h.begin(), h.end());
                }
            }
        }
        fprintf(stderr, "\r[gt] %lld/%lld", (long long)(off + m), (long long)N);
    }
    fprintf(stderr, "\n");
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<int32_t> I32(size_t(Q.n) * k, -1);
    for (int64_t qi = 0; qi < Q.n; qi++) {
        auto h = heap[qi];
        std::sort_heap(h.begin(), h.end());          // ascending distance
        for (size_t j = 0; j < h.size(); j++) I32[size_t(qi) * k + j] = h[j].second;
    }
    write_ivec(a.get("out"), I32, int32_t(Q.n), int32_t(k));
    fprintf(stderr, "[gt] nq=%lld k=%d N=%lld %.1fs -> %s\n",
            (long long)Q.n, k, (long long)N, s, a.get("out").c_str());
    return 0;
}

/// Measurement.
///
/// **Shards are not run in parallel threads.** In a distributed deployment the shards live on
/// different machines, so emulating them with threads in one process would (a) add L3 and memory
/// bandwidth contention that the deployment does not have, and (b) under the same CPU quota use 1
/// thread at S=1 but 15 at S=15, so different S would not be comparable.
///
/// Instead a **shard-major loop** runs each shard alone. In deployment one machine keeps serving its
/// own shard, so the cache stays warm for that shard; hopping between shards per query would break
/// that condition. Deployment numbers are then composed offline from the per-shard latencies:
///
///   deployment latency = max_s(latency[s]) + merge
///   total CPU          = sum_s(latency[s]) + merge
///
/// The max is an **optimistic estimate**: in reality machine loads are correlated and fan-out network
/// and coordinator waits are added. What is isolated here is the ANN search cost.
int cmd_search(const Args& a) {
    // CPU-model check is opt-in: --expect_cpu "EPYC 9354" (the paper's machine) refuses to run on any
    // other model. Empty (default) = no check. The detected model is always printed to stderr.
    const std::string expect_cpu = a.get("expect_cpu", "");
    std::string model = cpu_model();
    fprintf(stderr, "[search] CPU: '%s'\n", model.c_str());
    if (!expect_cpu.empty() && model.find(expect_cpu) == std::string::npos) {
        fprintf(stderr, "!! unexpected CPU: '%s' (expected '%s'). The measurement would be contaminated\n", model.c_str(), expect_cpu.c_str());
        return 2;
    }

    int S = a.geti("S", 1);
    int K = a.geti("K", 20);
    int ki = a.geti("ki", K);
    int ef = a.geti("ef", 200);
    float gamma = float(a.getd("gamma", 0.0));
    int repeat = a.geti("repeat", 7);
    int warmup = a.geti("warmup", 1000);

    std::vector<std::unique_ptr<faiss::Index>> shards;
    std::vector<std::vector<int32_t>> sel(S);
    for (int s = 0; s < S; s++) {
        char p[4096];
        snprintf(p, sizeof p, "%s/shard%d.index", a.get("shards").c_str(), s);
        shards.emplace_back(faiss::read_index(p));
        snprintf(p, sizeof p, "%s/shard%d.sel", a.get("shards").c_str(), s);
        int32_t n, one;
        sel[s] = read_ivec(p, n, one);
    }

    auto Q = read_fbin(a.get("query"), a.geti("nq", 1000), a.geti("qoff", 0));
    if (a.geti("normalize", 0)) normalize_rows(Q);
    int32_t gn, gk;
    auto gt = read_ivec(a.get("gt"), gn, gk);
    check_gt_shape(gn, gk, Q.n, K);

    // Check that normalization really happened. Forgetting the flag is the kind of mistake that is silently wrong
    if (a.geti("check_norm", 1)) {
        auto bad = [](float nrm) { return std::fabs(nrm - 1.0f) > 1e-3f; };
        for (int64_t q = 0; q < std::min<int64_t>(4, Q.n); q++) {
            double t = 0;
            for (int64_t j = 0; j < Q.d; j++) t += double(Q.x[q * Q.d + j]) * Q.x[q * Q.d + j];
            if (bad(float(std::sqrt(t)))) {
                fprintf(stderr, "!! query is not normalized (|q%lld| = %.4f). Missing --normalize 1?\n",
                        (long long)q, std::sqrt(t));
                return 2;
            }
        }
        std::vector<float> v(size_t(Q.d));
        for (int s = 0; s < std::min(S, 2); s++) {
            shards[s]->reconstruct(0, v.data());
            double t = 0;
            for (int64_t j = 0; j < Q.d; j++) t += double(v[j]) * v[j];
            if (bad(float(std::sqrt(t)))) {
                fprintf(stderr, "!! vectors of shard %d are not normalized (|x0| = %.4f)\n", s, std::sqrt(t));
                return 2;
            }
        }
    }

    // --gamma is one value or one per shard (comma-separated list of S values): each shard picks its
    // own gamma, as in deployment. 0 turns the rule off on that shard (the patch's adaptive_gamma > 0 branch).
    std::vector<float> gamma_s(size_t(S), gamma);
    if (a.get("gamma").find(',') != std::string::npos) {
        gamma_s.clear();
        std::string g = a.get("gamma");
        for (size_t at = 0; at <= g.size();) {
            size_t c = g.find(',', at);
            if (c == std::string::npos) c = g.size();
            gamma_s.push_back(float(std::stod(g.substr(at, c - at))));
            at = c + 1;
        }
        if (int(gamma_s.size()) != S) {
            fprintf(stderr, "!! the --gamma list must have one value per shard (%d), got %zu\n", S, gamma_s.size());
            return 2;
        }
        gamma = *std::max_element(gamma_s.begin(), gamma_s.end());
    }
    std::string gamma_json = "[";
    for (int s = 0; s < S; s++) {
        char b[32];
        snprintf(b, sizeof b, "%s%g", s ? ", " : "", gamma_s[s]);
        gamma_json += b;
    }
    gamma_json += "]";

    faiss::SearchParametersHNSW sp;
    sp.efSearch = ef;
    sp.adaptive_gamma = gamma;
    sp.adaptive_k = a.geti("rule_k", 0);
    // The faiss default is the variant whose candidate queue is bounded by ef, which structurally
    // limits the search. 0 selects the unbounded-queue variant (search_from_candidate_unbounded),
    // whose termination is based on the **result heap**, so under a filter it continues until enough
    // passing results are found -- the hnswlib behaviour.
    sp.bounded_queue = a.geti("bounded_queue", 1) != 0;

    // Queue-saturation early termination (Patience, patch 0003), the comparison baseline. --patience_p 0
    // (default) switches it off. --patience_s is parsed as a double (lib/patience_replay.py reads the same
    // string with float()). --patience_queue: ef = the ef nearest points computed so far, ki = the result heap.
    sp.patience_p = a.geti("patience_p", 0);
    sp.patience_s = a.getd("patience_s", 0.995);
    const std::string pq = a.get("patience_queue", "ef");
    if (pq != "ef" && pq != "ki") {
        fprintf(stderr, "!! --patience_queue must be ef or ki\n");
        return 2;
    }
    sp.patience_queue = pq == "ki" ? 1 : 0;
    if (sp.patience_p < 0 || (sp.patience_p > 0 && !sp.bounded_queue)) {
        fprintf(stderr, "!! --patience_p must be >= 0 and needs --bounded_queue 1\n");
        return 2;
    }

    // --trace: run at gamma=0 and record, at every base-layer pop, (frontier, d_{rule_k}, cumulative
    // ndis, result-heap ids). The gamma grid is computed by replaying this record (lib/trace_replay.py).
    // The rule only decides when to stop, so a search with gamma = this record cut at the first pop
    // where the rule fires. Format (little endian):
    //   header  "V31T" int32 S, nq, ki, rule_k, ef
    //   per (shard, query)  int32 s, q, npop, upper_ndis | float d0[npop] | float dk[npop] (NaN = aux heap not full)
    //                       | int32 base_ndis[npop] | int32 gid[npop*ki] (global ids, -1 = empty slot)
    //   The last row (d0 = NaN) is the final state on leaving the loop.
    // --trace_ins 1 adds, after base_ndis, two columns for Patience replay (lib/patience_replay.py):
    //   int32 ins_ef[npop] | int32 ins_ki[npop], the cumulative insertions into the ef queue and the result
    //   heap when the row was written (i.e. after the previous candidate was expanded). The header then reads
    //   "V31Q" instead of "V31T", so a reader of the plain format refuses the file.
    const bool trace_ins = a.geti("trace_ins", 0) != 0;
    FILE* trace_f = nullptr;
    if (trace_ins && a.get("trace").empty()) {
        fprintf(stderr, "!! --trace_ins needs --trace\n");
        return 2;
    }
    if (!a.get("trace").empty()) {
        if (gamma != 0.0f || sp.adaptive_k <= 0 || !sp.bounded_queue || repeat != 1 || !a.get("filter").empty() ||
            sp.patience_p != 0) {
            fprintf(stderr, "!! --trace requires --gamma 0 --patience_p 0 --rule_k >0 --bounded_queue 1 --repeat 1 and no filter\n");
            return 2;
        }
        trace_f = fopen(a.get("trace").c_str(), "wb");
        const int32_t hdr[5] = {S, int32_t(Q.n), ki, sp.adaptive_k, ef};
        fwrite(trace_ins ? "V31Q" : "V31T", 1, 4, trace_f); fwrite(hdr, 4, 5, trace_f);
    }
    std::vector<float> tr_d0, tr_dk; std::vector<int> tr_nd, tr_ie, tr_ik; std::vector<faiss::idx_t> tr_ids;

    // --gt_out: search all shards with this ef and write, as ivec, the global top-K obtained by
    // merging the per-shard top-ki by distance -- an approximate calibration ground truth
    // (paper: ef_gt = min(1000, 4E), K = ki = 300). Evaluation ground truth stays exact.
    const bool want_gtout = !a.get("gt_out").empty();
    std::vector<std::vector<std::pair<float, int32_t>>> gtcand(want_gtout ? Q.n : 0);

    // Filter for cmd_search: map the global mask to a local id set per shard. faiss in-filtering
    // computes every distance and only blocks entry to the result heap, so every visited node pays a
    // selector lookup -- a cost ndis does not see, so this axis has to be measured in real CPU.
    // The selector implementation is selectable. In-filtering asks membership **for every visited
    // node**, so that data structure dominates CPU. Batch is an unordered_set, a cache miss per lookup
    // when it is large; Bitmap is 10M bits = 1.25 MB and fits in cache. ndis is the same for both, so
    // the whole difference is the membership-check cost.
    std::vector<std::vector<int64_t>> flocal(S);
    std::vector<std::unique_ptr<faiss::IDSelector>> fsel(S);
    std::vector<std::vector<uint8_t>> fbits(S);
    const bool use_bitmap = a.get("selector", "batch") == "bitmap";
    if (!a.get("filter").empty()) {
        int32_t fn, fone;
        auto f = read_ivec(a.get("filter"), fn, fone);
        std::unordered_set<int32_t> pass(f.begin(), f.end());
        for (int s2 = 0; s2 < S; s2++) {
            for (size_t i = 0; i < sel[s2].size(); i++)
                if (pass.count(sel[s2][i])) flocal[s2].push_back(int64_t(i));
            if (use_bitmap) {
                fbits[s2].assign((sel[s2].size() + 7) / 8, 0);
                for (int64_t v : flocal[s2]) fbits[s2][v >> 3] |= uint8_t(1u << (v & 7));
                fsel[s2] = std::make_unique<faiss::IDSelectorBitmap>(sel[s2].size(), fbits[s2].data());
            } else {
                fsel[s2] = std::make_unique<faiss::IDSelectorBatch>(
                        flocal[s2].size(), flocal[s2].data());
            }
        }
        size_t tot = 0; for (auto& v : flocal) tot += v.size();
        fprintf(stderr, "[search] filter: %zu ids (%.4f%%), %.0f per shard on average\n",
                pass.size(), 100.0 * double(pass.size()) / double(gn ? 1 : 1) / 1.0, double(tot) / S);
    }

    if (Throttle::read().nr_throttled < 0)
        fprintf(stderr, "[search] no cgroup v2 throttle counters in %s: throttle gate disabled\n",
                cgroup_cpu_stat_path().c_str());

    printf("[\n");
    for (int rep = 0; rep < repeat; rep++) {
        double nf_before = noise_floor_ms();
        Throttle t_before = Throttle::read();

        // lat[s][q] -- filled shard-major
        std::vector<std::vector<double>> lat(S, std::vector<double>(Q.n, 0.0));
        std::vector<std::vector<double>> cpu(S, std::vector<double>(Q.n, 0.0));
        std::vector<long long> ndis_q(Q.n, 0);
        std::vector<int> hits_q(Q.n, 0);
        double cg_before = cgroup_cpu_ms();
        std::vector<std::vector<int32_t>> got(size_t(S) * Q.n);
        // Per-shard ndis. Deployment latency is a max over shards, so how evenly the shards work goes
        // straight into latency. Time would pick up load on shared nodes, so use deterministic ndis.
        std::vector<std::vector<long long>> ndis_sq(S, std::vector<long long>(Q.n, 0));
        size_t ndis_total = 0;

        for (int s = 0; s < S; s++) {
            auto* h = dynamic_cast<faiss::IndexHNSW*>(shards[s].get());
            std::vector<int64_t> ids(ki);
            std::vector<float> dis(ki);
            faiss::SearchParametersHNSW sps = sp;
            sps.adaptive_gamma = gamma_s[s];
            if (fsel[s]) sps.sel = fsel[s].get();
            for (int64_t qi = 0; qi < std::min<int64_t>(warmup, Q.n); qi++)
                h->search(1, Q.x.data() + qi * Q.d, ki, dis.data(), ids.data(), &sps);
            if (trace_f) {
                sps.trace_d0 = &tr_d0; sps.trace_dk = &tr_dk; sps.trace_ndis = &tr_nd;
                sps.trace_ids = &tr_ids; sps.trace_k = ki;
                if (trace_ins) { sps.trace_ins_ef = &tr_ie; sps.trace_ins_ki = &tr_ik; }
            }
            for (int64_t qi = 0; qi < Q.n; qi++) {
                faiss::hnsw_stats.reset();
                if (trace_f) { tr_d0.clear(); tr_dk.clear(); tr_nd.clear(); tr_ids.clear(); tr_ie.clear(); tr_ik.clear(); }
                double c0 = thread_cpu_ms(), t0 = wall_ms();
                h->search(1, Q.x.data() + qi * Q.d, ki, dis.data(), ids.data(), &sps);
                lat[s][qi] = wall_ms() - t0;
                cpu[s][qi] = thread_cpu_ms() - c0;
                if (trace_f) {
                    const int32_t npop = int32_t(tr_d0.size());
                    const int32_t upper = int32_t(faiss::hnsw_stats.ndis) - tr_nd.back();
                    const int32_t rec[4] = {s, int32_t(qi), npop, upper};
                    fwrite(rec, 4, 4, trace_f);
                    fwrite(tr_d0.data(), 4, npop, trace_f);
                    fwrite(tr_dk.data(), 4, npop, trace_f);
                    fwrite(tr_nd.data(), 4, npop, trace_f);
                    if (trace_ins) {
                        if (int32_t(tr_ie.size()) != npop || int32_t(tr_ik.size()) != npop) {
                            fprintf(stderr, "!! trace columns differ in length (npop %d, ins_ef %zu, ins_ki %zu)\n",
                                    npop, tr_ie.size(), tr_ik.size());
                            return 3;
                        }
                        fwrite(tr_ie.data(), 4, npop, trace_f);
                        fwrite(tr_ik.data(), 4, npop, trace_f);
                    }
                    std::vector<int32_t> g(tr_ids.size());
                    for (size_t j = 0; j < tr_ids.size(); j++) g[j] = tr_ids[j] >= 0 ? sel[s][tr_ids[j]] : -1;
                    fwrite(g.data(), 4, g.size(), trace_f);
                }
                ndis_total += faiss::hnsw_stats.ndis;
                ndis_q[qi] += (long long)faiss::hnsw_stats.ndis;
                ndis_sq[s][qi] = (long long)faiss::hnsw_stats.ndis;
                auto& g = got[size_t(s) * Q.n + qi];
                for (int j = 0; j < ki; j++)
                    if (ids[j] >= 0) g.push_back(sel[s][ids[j]]);
                if (want_gtout && rep == 0)
                    for (int j = 0; j < ki; j++)
                        if (ids[j] >= 0) gtcand[qi].emplace_back(dis[j], sel[s][ids[j]]);
            }
        }

        // Time the merge separately. At S=15, k_i=7 it is only 105 candidates, but do not assume it is negligible
        std::vector<double> merge_ms(Q.n, 0.0), merge_cpu(Q.n, 0.0);
        double recall_sum = 0;
        std::vector<int32_t> merged;
        for (int64_t qi = 0; qi < Q.n; qi++) {
            double mc0 = thread_cpu_ms(), t0 = wall_ms();
            merged.clear();
            for (int s = 0; s < S; s++) {
                auto& g = got[size_t(s) * Q.n + qi];
                merged.insert(merged.end(), g.begin(), g.end());
            }
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            merge_ms[qi] = wall_ms() - t0;
            merge_cpu[qi] = thread_cpu_ms() - mc0;

            std::vector<int32_t> truth(gt.begin() + size_t(qi) * gk, gt.begin() + size_t(qi) * gk + K);
            std::sort(truth.begin(), truth.end());
            std::vector<int32_t> hit;
            std::set_intersection(truth.begin(), truth.end(), merged.begin(), merged.end(), std::back_inserter(hit));
            recall_sum += double(hit.size()) / K;
            hits_q[qi] = int(hit.size());
        }

        // Per-query raw data. Aggregates alone cannot draw distributions or build oracles.
        // --dump_shard_time 1: timings vary per repetition, so **every repetition** is written with a
        // rep column. The analysis selects repetitions by the per-repetition accept field in the JSON.
        const bool dump_time = a.geti("dump_shard_time", 0) != 0;
        if (!a.get("dump_shard").empty() && (rep == 0 || dump_time)) {
            FILE* sf = fopen(a.get("dump_shard").c_str(), rep == 0 ? "w" : "a");
            if (sf) {
                if (dump_time) {
                    if (rep == 0) fprintf(sf, "rep\tq\tshard\tndis\tcpu_ms\twall_ms\n");
                    for (int64_t qi = 0; qi < Q.n; qi++)
                        for (int s2 = 0; s2 < S; s2++)
                            fprintf(sf, "%d\t%lld\t%d\t%lld\t%.6f\t%.6f\n", rep, (long long)qi, s2,
                                    ndis_sq[s2][qi], cpu[s2][qi], lat[s2][qi]);
                } else {
                    fprintf(sf, "q\tshard\tndis\n");
                    for (int64_t qi = 0; qi < Q.n; qi++)
                        for (int s2 = 0; s2 < S; s2++)
                            fprintf(sf, "%lld\t%d\t%lld\n", (long long)qi, s2, ndis_sq[s2][qi]);
                }
                fclose(sf);
            }
        }
        if (!a.get("dump").empty() && rep == 0) {
            std::string dp = a.get("dump");
            FILE* df = fopen(dp.c_str(), "w");
            if (df) {
                fprintf(df, "q\twall_sum_ms\tcpu_sum_ms\tmax_ms\tmerge_ms\tndis\thits\n");
                for (int64_t qi = 0; qi < Q.n; qi++) {
                    double sm = 0, mx = 0, cp = 0;
                    for (int s = 0; s < S; s++) {
                        sm += lat[s][qi]; cp += cpu[s][qi]; mx = std::max(mx, lat[s][qi]);
                    }
                    fprintf(df, "%lld\t%.6f\t%.6f\t%.6f\t%.6f\t%lld\t%d\n",
                            (long long)qi, sm, cp, mx, merge_ms[qi],
                            (long long)ndis_q[qi], hits_q[qi]);
                }
                fclose(df);
            }
        }

        // Compose: total CPU = sum, deployment latency = max. Both add the merge time
        std::vector<double> sum_ms(Q.n), max_ms(Q.n), cpu_ms(Q.n);
        for (int64_t qi = 0; qi < Q.n; qi++) {
            double sm = 0, mx = 0, cp = 0;
            for (int s = 0; s < S; s++) {
                sm += lat[s][qi];
                cp += cpu[s][qi];
                mx = std::max(mx, lat[s][qi]);
            }
            sum_ms[qi] = sm + merge_ms[qi];
            max_ms[qi] = mx + merge_ms[qi];
            cpu_ms[qi] = cp;            // the merge belongs to the coordinator, so it is not in shard CPU
        }
        double cg_after = cgroup_cpu_ms();
        double cg_total = (cg_before < 0 || cg_after < 0) ? -1.0 : cg_after - cg_before;   // -1 = unavailable

        Throttle t_after = Throttle::read();
        double nf_after = noise_floor_ms();
        const bool thr_avail = t_before.nr_throttled >= 0 && t_after.nr_throttled >= 0;
        long long thr = thr_avail ? t_after.nr_throttled - t_before.nr_throttled : -1;   // -1 = unavailable
        // If the ruler's scale moved, the measurements in between are suspect too. Run-to-run
        // variation under identical conditions was 3.1% in a pilot, so a generous 10% is the threshold
        double nf_drift = 100.0 * std::fabs(nf_after - nf_before) / std::min(nf_before, nf_after);
        bool ok = (!thr_avail || thr == 0) && (nf_drift < 10.0);
        auto mean = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
        };

        // Patience settings are printed only when it is on, so other runs print exactly what they did before.
        char pat_json[160] = "";
        if (sp.patience_p > 0)
            snprintf(pat_json, sizeof pat_json, " \"patience_p\": %d, \"patience_s\": %.17g, \"patience_queue\": \"%s\",",
                     sp.patience_p, sp.patience_s, pq.c_str());
        printf("  {\"rep\": %d, \"S\": %d, \"K\": %d, \"ki\": %d, \"rule_k\": %d, \"ef\": %d, \"gamma\": %.4f,\n"
               "   \"gamma_shard\": %s,%s\n"
               "   \"recall\": %.5f, \"ndis_per_query\": %.1f,\n"
               "   \"wall_sum_ms\": {\"mean\": %.4f, \"p50\": %.4f, \"p90\": %.4f, \"p99\": %.4f, \"p999\": %.4f},\n"
               "   \"cpu_sum_ms\": {\"mean\": %.4f, \"p50\": %.4f, \"p90\": %.4f, \"p99\": %.4f},\n"
               "   \"cpu_over_wall\": %.4f, \"cgroup_cpu_ms_total\": %.1f,\n"
               "   \"merged_ms\": {\"mean\": %.4f, \"p50\": %.4f, \"p90\": %.4f, \"p99\": %.4f, \"p999\": %.4f},\n"
               "   \"merge_ms\": {\"mean\": %.5f, \"p99\": %.5f}, \"merge_cpu_ms_mean\": %.5f,\n"
               "   \"clocks\": {\"wall\": \"CLOCK_MONOTONIC\", \"cpu\": \"CLOCK_THREAD_CPUTIME_ID\",\n"
               "               \"cgroup_cpu\": \"cgroup v2 cpu.stat usage_usec\"},\n"
               "   \"note\": \"cpu_sum_ms/wall_sum_ms are sums over shards measured one at a time; "
               "merged_ms is max over shards + merge; merge runs on the coordinator so its CPU is "
               "reported separately and excluded from cpu_sum_ms\",\n"
               "   \"noise_floor_ms\": [%.3f, %.3f], \"noise_drift_pct\": %.1f,\n"
               "   \"nr_throttled_delta\": %lld, \"accept\": %s}%s\n",
               rep, S, K, ki, sp.adaptive_k, ef, gamma, gamma_json.c_str(), pat_json,
               recall_sum / double(Q.n), double(ndis_total) / double(Q.n),
               mean(sum_ms), pct(sum_ms, 0.50), pct(sum_ms, 0.90), pct(sum_ms, 0.99), pct(sum_ms, 0.999),
               mean(cpu_ms), pct(cpu_ms, 0.50), pct(cpu_ms, 0.90), pct(cpu_ms, 0.99),
               mean(cpu_ms) / std::max(mean(sum_ms) - mean(merge_ms), 1e-9), cg_total,
               mean(max_ms), pct(max_ms, 0.50), pct(max_ms, 0.90), pct(max_ms, 0.99), pct(max_ms, 0.999),
               mean(merge_ms), pct(merge_ms, 0.99), mean(merge_cpu),
               nf_before, nf_after, nf_drift, thr, ok ? "true" : "false",
               rep + 1 < repeat ? "," : "");
        fflush(stdout);
    }
    printf("]\n");
    if (trace_f) fclose(trace_f);
    if (want_gtout) {
        std::vector<int32_t> out(size_t(Q.n) * K, -1);
        for (int64_t qi = 0; qi < Q.n; qi++) {
            auto& c = gtcand[qi];
            std::sort(c.begin(), c.end());
            for (int j = 0; j < K && j < int(c.size()); j++) out[size_t(qi) * K + j] = c[j].second;
        }
        write_ivec(a.get("gt_out"), out, int32_t(Q.n), int32_t(K));
    }
    return 0;
}


/// Oracle -- a lower bound that no per-query stopping method can beat.
///
/// An oracle of the form "per query, pick the best ef from a grid" is loose in two places: it is
/// confined to the grid, and it keeps searching until that ef's search ends even after all answers
/// are in. The rule does not pick an ef, it stops on the trajectory, so the right lower bound is
/// **the moment the last needed true neighbour entered the result heap**.
///
/// Why one run suffices: a point of the true top-K that enters the result heap and is evicted never
/// re-enters, so **in the final result = entered and survived to the end**. Recording arrival times
/// and taking the max over those in the final result is enough; no re-runs or binary search.
///
/// Counted per shard. Each shard can return only the true neighbours it holds, so the arrival time
/// of the last one it returns is that shard's bound. Upper-layer descent is not subtracted -- even a
/// stopping rule has to reach the base layer first. A shard that returns no true neighbour pays only
/// the descent.
///
/// This is still not an absolute bound: it is a bound **on the ef=E trajectory**, and another ef may
/// reach the answers sooner by luck. Measure per ef and take the minimum to get closer to the true bound.
int cmd_oracle(const Args& a) {
    int S = a.geti("S", 1), K = a.geti("K", 20), ki = a.geti("ki", K), ef = a.geti("ef", 200);

    std::vector<std::unique_ptr<faiss::Index>> shards;
    std::vector<std::vector<int32_t>> sel(S);
    int64_t nmax = 0;
    for (int s = 0; s < S; s++) {
        char p[4096];
        snprintf(p, sizeof p, "%s/shard%d.index", a.get("shards").c_str(), s);
        shards.emplace_back(faiss::read_index(p));
        snprintf(p, sizeof p, "%s/shard%d.sel", a.get("shards").c_str(), s);
        int32_t n, one;
        sel[s] = read_ivec(p, n, one);
        for (int32_t v : sel[s]) nmax = std::max<int64_t>(nmax, v + 1);
    }

    // global id -> (shard, local id). Ground truth comes as global ids, search returns local ids
    std::vector<int16_t> home_shard(nmax, -1);
    std::vector<int32_t> home_local(nmax, -1);
    for (int s = 0; s < S; s++)
        for (size_t i = 0; i < sel[s].size(); i++) {
            home_shard[sel[s][i]] = int16_t(s);
            home_local[sel[s][i]] = int32_t(i);
        }

    auto Q = read_fbin(a.get("query"), a.geti("nq", 1000), a.geti("qoff", 0));
    if (a.geti("normalize", 0)) normalize_rows(Q);
    int32_t gn, gk;
    auto gt = read_ivec(a.get("gt"), gn, gk);
    check_gt_shape(gn, gk, Q.n, K);

    faiss::SearchParametersHNSW sp;
    sp.efSearch = ef;
    // The oracle must run on the same search path. Without this it falls back to the default
    // (bounded) path, where under a filter the work equals the unfiltered work and the bound would
    // measure something else entirely.
    sp.bounded_queue = a.geti("bounded_queue", 1) != 0;
    // With a filter the oracle must run under the same filter. Filtering only the ground truth while
    // searching unfiltered would be the inconsistent "found a point the filter removed, yet missed it".
    std::vector<std::vector<int64_t>> flocal_o(S);
    std::vector<std::unique_ptr<faiss::IDSelectorBatch>> fsel_o(S);
    if (!a.get("filter").empty()) {
        int32_t fn2, fone2;
        auto f2 = read_ivec(a.get("filter"), fn2, fone2);
        std::unordered_set<int32_t> pass(f2.begin(), f2.end());
        for (int s2 = 0; s2 < S; s2++) {
            for (size_t i = 0; i < sel[s2].size(); i++)
                if (pass.count(sel[s2][i])) flocal_o[s2].push_back(int64_t(i));
            fsel_o[s2] = std::make_unique<faiss::IDSelectorBatch>(
                    flocal_o[s2].size(), flocal_o[s2].data());
        }
        fprintf(stderr, "[oracle] filter: %zu ids\n", pass.size());
    }
    std::unordered_set<faiss::idx_t> targets;
    std::vector<int> hit_ids, hit_ndis;
    int base_ndis = 0;
    sp.oracle_targets = &targets;
    sp.oracle_hit_ids = &hit_ids;
    sp.oracle_hit_ndis = &hit_ndis;
    sp.oracle_base_ndis = &base_ndis;

    FILE* df = nullptr;
    if (!a.get("dump").empty()) {
        df = fopen(a.get("dump").c_str(), "w");
        if (df) fprintf(df, "q\tref_ndis\toracle_ndis\thits\n");
    }

    double sum_ref = 0, sum_orc = 0;
    // Split the waste into two kinds.
    //   empty   -- shards holding none of the global top-K. They run at full cost and contribute
    //              nothing. **This waste is created purely by partitioning** and grows quickly with S.
    //              A single node has none of it.
    //   confirm -- the part a shard holding answers keeps running after its last answer arrived.
    //              A single node has this too.
    // The oracle gap (ref - oracle) is the sum of the two, so splitting it separates "waste created
    // by partitioning" from "waste inherent to graph search".
    double ndis_empty = 0, ndis_useful = 0, orc_empty = 0, orc_useful = 0;
    long long n_empty = 0, n_useful = 0;
    long long sum_hits = 0;
    std::vector<int64_t> ids(ki);
    std::vector<float> dis(ki);
    for (int64_t qi = 0; qi < Q.n; qi++) {
        const int32_t* g = gt.data() + size_t(qi) * gk;
        std::unordered_set<int32_t> truth(g, g + K);
        long long ref_q = 0, orc_q = 0;
        int hits_q = 0;
        for (int s = 0; s < S; s++) {
            targets.clear();
            for (int j = 0; j < K; j++)
                if (g[j] < nmax && home_shard[g[j]] == s) targets.insert(home_local[g[j]]);

            hit_ids.clear(); hit_ndis.clear(); base_ndis = 0;
            faiss::hnsw_stats.reset();
            auto* h = dynamic_cast<faiss::IndexHNSW*>(shards[s].get());
            faiss::SearchParametersHNSW spo = sp;
            if (fsel_o[s]) spo.sel = fsel_o[s].get();
            h->search(1, Q.x.data() + qi * Q.d, ki, dis.data(), ids.data(), &spo);
            long long shard_ndis = (long long)faiss::hnsw_stats.ndis;
            ref_q += shard_ndis;

            // Only true neighbours left in the final result are "needed". Those that entered and were evicted do not count
            std::unordered_set<int32_t> kept;
            for (int j = 0; j < ki; j++)
                if (ids[j] >= 0 && truth.count(sel[s][ids[j]])) kept.insert(int32_t(ids[j]));
            hits_q += int(kept.size());

            int stop = 0;
            for (size_t t = 0; t < hit_ids.size(); t++)
                if (kept.count(hit_ids[t])) stop = std::max(stop, hit_ndis[t]);
            const double orc_s = double(shard_ndis - base_ndis) + double(stop);
            orc_q += (long long)orc_s;                  // upper-layer descent + where the base layer stopped
            if (kept.empty()) { ndis_empty += double(shard_ndis); orc_empty += orc_s; n_empty++; }
            else              { ndis_useful += double(shard_ndis); orc_useful += orc_s; n_useful++; }
        }
        sum_ref += double(ref_q); sum_orc += double(orc_q); sum_hits += hits_q;
        if (df) fprintf(df, "%lld\t%lld\t%lld\t%d\n", (long long)qi, ref_q, orc_q, hits_q);
    }
    if (df) fclose(df);

    double n = double(Q.n);
    printf("{\"S\": %d, \"K\": %d, \"ki\": %d, \"ef\": %d, \"nq\": %lld,\n"
           " \"recall\": %.5f,\n"
           " \"ref_ndis_per_query\": %.1f, \"oracle_ndis_per_query\": %.1f, \"ratio\": %.4f,\n"
           " \"empty_shards\": {\"frac\": %.4f, \"ndis_per_query\": %.1f, \"oracle_per_query\": %.1f},\n"
           " \"useful_shards\": {\"frac\": %.4f, \"ndis_per_query\": %.1f, \"oracle_per_query\": %.1f},\n"
           " \"waste_split\": {\"empty\": %.1f, \"confirm\": %.1f},\n"
           " \"note\": \"oracle = stop the instant the last kept ground-truth neighbour entered the "
           "result heap, per shard, on this same ef=%d trajectory; upper-layer descent is not refundable. "
           "waste_split divides (ref - oracle) by whether the shard returned any true neighbour in its "
           "final result: 'empty' contributed nothing (mostly the partitioning tail (1-1/S)^K, plus a "
           "few percent of shards that held an answer but missed it -- at S=1 it is only the latter), "
           "'useful' is confirmation after the last kept answer arrived.\"}\n",
           S, K, ki, ef, (long long)Q.n, double(sum_hits) / (n * K),
           sum_ref / n, sum_orc / n, sum_orc / std::max(sum_ref, 1.0),
           double(n_empty) / std::max<double>(double(n_empty + n_useful), 1.0), ndis_empty / n, orc_empty / n,
           double(n_useful) / std::max<double>(double(n_empty + n_useful), 1.0), ndis_useful / n, orc_useful / n,
           (ndis_empty - orc_empty) / n, (ndis_useful - orc_useful) / n, ef);
    return 0;
}

}  // namespace

/// Measures the rank-weighted decomposition -- a check of the theory (w_j * p_j).
///
/// Observation: the X_i points of the global top-K that fall in shard i are exactly ranks 1..X_i
/// inside that shard, because distance order is preserved on subsets (if a point p of shard s is
/// closer than a true neighbour g, p's global rank is also ahead of g, so p is in the top-K too).
/// Hence **the in-shard rank of every true neighbour is known exactly from the ground truth alone**:
/// counting the ground-truth points of shard s in global order gives 1, 2, 3, ...
///
/// Three quantities are reported.
///   w_j         share of all true neighbours that have in-shard rank j. The theory claims it depends
///               only on (K, S), not on the data; compare with the binomial prediction
///   p_j (cond)  probability that the j-th, given it is a true neighbour, is found.
///               sum_j w_j * p_j(cond) = recall holds as an **identity**
///   p_j (uncond) probability that the shard's actual j-th nearest point is found, whether or not it
///               is a global true neighbour. The theory's bound sum_j w_j * p_j(uncond) <= recall uses this
///
/// The difference between the two p_j is the slack of the independence assumption: shards where answers
/// cluster are also easier to search, so the conditional one is larger. The unconditional p_j needs the
/// **exact** per-shard top-J, so vectors are reconstructed from the index and searched with IndexFlatL2
/// (approximating with HNSW would make the check circular).
int cmd_ranks(const Args& a) {
    int S = a.geti("S", 1), K = a.geti("K", 20), ki = a.geti("ki", K);
    int ef = a.geti("ef", 200), J = a.geti("J", 8);
    float gamma = float(a.getd("gamma", 0.0));

    auto Q = read_fbin(a.get("query"), a.geti("nq", 1000), a.geti("qoff", 0));
    if (a.geti("normalize", 0)) normalize_rows(Q);
    int32_t gn, gk;
    auto gt = read_ivec(a.get("gt"), gn, gk);
    check_gt_shape(gn, gk, Q.n, K);

    faiss::SearchParametersHNSW sp;
    sp.efSearch = ef;
    sp.adaptive_gamma = gamma;
    sp.adaptive_k = a.geti("rule_k", 0);

    const int JM = std::max(J, K) + 1;
    std::vector<long long> n_present(JM, 0), n_found_ans(JM, 0), n_found_true(JM, 0), n_true(JM, 0);

    for (int s = 0; s < S; s++) {
        char p[4096];
        snprintf(p, sizeof p, "%s/shard%d.index", a.get("shards").c_str(), s);
        std::unique_ptr<faiss::Index> idx(faiss::read_index(p));
        snprintf(p, sizeof p, "%s/shard%d.sel", a.get("shards").c_str(), s);
        int32_t n, one;
        auto sel = read_ivec(p, n, one);

        std::unordered_map<int32_t, int32_t> to_local;
        to_local.reserve(sel.size() * 2);
        for (size_t i = 0; i < sel.size(); i++) to_local[sel[i]] = int32_t(i);

        // exact per-shard top-J: reconstruct the vectors from the index and search exhaustively
        std::vector<float> X(size_t(n) * Q.d);
        idx->reconstruct_n(0, n, X.data());
        faiss::IndexFlatL2 flat(int(Q.d));
        flat.add(n, X.data());
        std::vector<float> tdis(size_t(Q.n) * J);
        std::vector<int64_t> tids(size_t(Q.n) * J);
        flat.search(Q.n, Q.x.data(), J, tdis.data(), tids.data());
        std::vector<float>().swap(X);

        auto* h = dynamic_cast<faiss::IndexHNSW*>(idx.get());
        std::vector<int64_t> ids(ki);
        std::vector<float> dis(ki);
        for (int64_t qi = 0; qi < Q.n; qi++) {
            h->search(1, Q.x.data() + qi * Q.d, ki, dis.data(), ids.data(), &sp);
            std::unordered_set<int64_t> got;
            for (int j = 0; j < ki; j++) if (ids[j] >= 0) got.insert(ids[j]);

            int rank = 0;
            const int32_t* g = gt.data() + size_t(qi) * gk;
            for (int m = 0; m < K; m++) {
                auto it = to_local.find(g[m]);
                if (it == to_local.end()) continue;
                rank++;
                if (rank < JM) {
                    n_present[rank]++;
                    if (got.count(it->second)) n_found_ans[rank]++;
                }
            }
            for (int j = 1; j <= J; j++) {
                int64_t t = tids[size_t(qi) * J + (j - 1)];
                if (t < 0) continue;
                n_true[j]++;
                if (got.count(t)) n_found_true[j]++;
            }
        }
        fprintf(stderr, "[ranks] shard %d/%d done\n", s + 1, S);
    }

    double tot = double(Q.n) * K;
    long long found_total = 0;
    for (int j = 1; j < JM; j++) found_total += n_found_ans[j];
    double bound = 0, sumw = 0;
    for (int j = 1; j <= J; j++) {
        double w = double(n_present[j]) / tot;
        sumw += w;
        bound += w * (n_true[j] ? double(n_found_true[j]) / double(n_true[j]) : 0.0);
    }

    printf("{\"S\": %d, \"K\": %d, \"ki\": %d, \"ef\": %d, \"gamma\": %.4f, \"nq\": %lld,\n",
           S, K, ki, ef, gamma, (long long)Q.n);
    printf(" \"recall\": %.6f,\n", double(found_total) / tot);
    printf(" \"rank\": [\n");
    for (int j = 1; j <= J; j++) {
        double w = double(n_present[j]) / tot;
        double pc = n_present[j] ? double(n_found_ans[j]) / double(n_present[j]) : 0.0;
        double pu = n_true[j] ? double(n_found_true[j]) / double(n_true[j]) : 0.0;
        printf("   {\"j\": %d, \"w\": %.6f, \"p_cond\": %.6f, \"p_uncond\": %.6f, \"n_present\": %lld}%s\n",
               j, w, pc, pu, n_present[j], j < J ? "," : "");
    }
    printf(" ],\n \"sum_w\": %.6f, \"bound_uncond\": %.6f,\n", sumw, bound);
    printf(" \"note\": \"sum_j w_j*p_cond_j == recall exactly (identity); bound_uncond uses the "
           "unconditional p_j and should be <= recall -- the gap is the slack from assuming "
           "independence between 'answer landed here' and 'search found it'\"}\n");
    return 0;
}

/// Builds a filter mask -- the subset left by an inverted-index condition in a hybrid query.
///
/// Two kinds are needed.
///   random  : uniform random. An easy baseline, but real filters do not look like this
///   cluster : nearest neighbours of a random seed point. **An inverted-index filter leaves a
///             semantically coherent set** (same brand, same topic), which also clusters in vector
///             space. This is the realistic one
///
/// Output: the list of passing global ids (.ivec format).
int cmd_mask(const Args& a) {
    auto X = read_fbin(a.get("base"), a.geti("limit", -1));
    if (a.geti("normalize", 0)) normalize_rows(X);
    double p = a.getd("sel", 0.01);
    int64_t m = std::max<int64_t>(1, int64_t(double(X.n) * p));
    std::string kind = a.get("kind", "random");
    uint64_t seed = uint64_t(a.geti("seed", 0));

    std::vector<int32_t> ids;
    if (kind == "random") {
        std::vector<int32_t> all(X.n);
        for (int64_t i = 0; i < X.n; i++) all[i] = int32_t(i);
        std::mt19937_64 rng(seed);
        for (int64_t i = 0; i < m; i++)
            std::swap(all[i], all[i + int64_t(rng() % uint64_t(X.n - i))]);
        ids.assign(all.begin(), all.begin() + m);
    } else if (kind == "cluster") {
        std::mt19937_64 rng(seed);
        int64_t c = int64_t(rng() % uint64_t(X.n));
        faiss::IndexFlatL2 flat(int(X.d));
        flat.add(X.n, X.x.data());
        std::vector<float> D(m);
        std::vector<int64_t> I(m);
        flat.search(1, X.x.data() + c * X.d, int(m), D.data(), I.data());
        ids.reserve(m);
        for (int64_t i = 0; i < m; i++) if (I[i] >= 0) ids.push_back(int32_t(I[i]));
    } else {
        fprintf(stderr, "!! --kind must be random|cluster\n");
        return 2;
    }
    std::sort(ids.begin(), ids.end());
    write_ivec(a.get("out"), ids, int32_t(ids.size()), 1);
    fprintf(stderr, "[mask] %s sel=%.4f -> %zu ids (%.4f%%) -> %s\n",
            kind.c_str(), p, ids.size(), 100.0 * double(ids.size()) / double(X.n),
            a.get("out").c_str());
    return 0;
}

/// Measures each query's "alignment with the filter".
///
/// Built to tell whether recall collapsing under a cluster filter is because **the filter is
/// clustered** or because **the filter's cluster lies away from the query**. The implications are
/// opposite: the former means cluster filters themselves do not suit in-filtering; the latter means
/// there is no problem when filter and query point at the same thing, as in hybrid queries.
///
/// No new masks are needed: there is one mask, but **the queries are scattered at different
/// distances from that cluster**, so alignment is already sampled.
///
///   align(q) = d(q, nearest passing point) / d(q, nearest point)
///
/// 1 means the query is inside the cluster (fully aligned); larger means outside. The nearest passing
/// point is rank 1 of the filtered ground truth, the overall nearest is rank 1 of the unfiltered one --
/// only distances need recomputing.
int cmd_align(const Args& a) {
    auto Q = read_fbin(a.get("query"), a.geti("nq", 1000), a.geti("qoff", 0));
    if (a.geti("normalize", 0)) normalize_rows(Q);
    int32_t gn, gk, fn, fk;
    auto gt = read_ivec(a.get("gt"), gn, gk);              // unfiltered ground truth
    auto gtf = read_ivec(a.get("gt_filtered"), fn, fk);    // filtered ground truth

    // Load only the base vectors needed: rank 1 of the two ground truths. No reason to load everything
    std::unordered_set<int32_t> need;
    for (int64_t q = 0; q < Q.n; q++) {
        need.insert(gt[size_t(q) * gk]);
        need.insert(gtf[size_t(q) * fk]);
    }
    int64_t n_file, d;
    {
        std::ifstream f(a.get("base"), std::ios::binary);
        int32_t n32, d32;
        f.read(reinterpret_cast<char*>(&n32), 4);
        f.read(reinterpret_cast<char*>(&d32), 4);
        n_file = n32; d = d32;
    }
    (void)n_file;
    std::unordered_map<int32_t, std::vector<float>> vec;
    {
        std::ifstream f(a.get("base"), std::ios::binary);
        std::vector<float> buf(d);
        for (int32_t id : need) {
            f.seekg(8 + int64_t(id) * d * 4);
            f.read(reinterpret_cast<char*>(buf.data()), d * 4);
            if (a.geti("normalize", 0)) {
                double t = 0;
                for (int64_t j = 0; j < d; j++) t += double(buf[j]) * buf[j];
                const float inv = t > 0 ? float(1.0 / std::sqrt(t)) : 1.0f;
                for (int64_t j = 0; j < d; j++) buf[j] *= inv;
            }
            vec[id] = buf;
        }
    }
    auto dist2 = [&](const float* x, const float* y) {
        double t = 0;
        for (int64_t j = 0; j < d; j++) { double v = double(x[j]) - y[j]; t += v * v; }
        return t;
    };

    FILE* out = a.get("out").empty() ? stdout : fopen(a.get("out").c_str(), "w");
    fprintf(out, "q\talign\td_all\td_pass\n");
    for (int64_t q = 0; q < Q.n; q++) {
        const float* qv = Q.x.data() + q * Q.d;
        double da = std::sqrt(dist2(qv, vec[gt[size_t(q) * gk]].data()));
        double dp = std::sqrt(dist2(qv, vec[gtf[size_t(q) * fk]].data()));
        fprintf(out, "%lld\t%.6f\t%.6f\t%.6f\n", (long long)q, da > 0 ? dp / da : 1.0, da, dp);
    }
    if (out != stdout) fclose(out);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: bench <build|add|gt|search|oracle|ranks|mask|align> [--key value ...]\n"
                "  build  --base X.fbin --out shard0.index [--M 40] [--efc 200] [--limit N] [--normalize 0]\n"
                "  add    --index in.index --base H.fbin --out out.index [--normalize 0]\n"
                "  gt     --base X.fbin --query Q.fbin --out gt.ivec [--k 300] [--nq 1000] [--qoff 0] [--normalize 1]\n"
                "         [--limit N] [--chunk 2000000]\n"
                "  search --shards DIR --S 5 --query Q.fbin --gt gt.ivec [--nq 1000] [--qoff 0]\n"
                "         [--K 20] [--ki 12] [--rule_k 27] [--ef 200] [--gamma 0.04 | g1,g2,...] [--repeat 7] [--warmup 1000]\n"
                "         [--normalize 1] [--check_norm 1] [--bounded_queue 1] [--vt_threshold N] [--expect_cpu STR]\n"
                "         [--qoff 1000]  e.g. calibration queries; gt row i must belong to query qoff+i\n"
                "         [--dump PATH]  per-query TSV (first repetition only)\n"
                "         [--dump_shard PATH [--dump_shard_time 1]]  query x shard ndis (+ cpu/wall for every repetition)\n"
                "         [--trace PATH]  base-layer traversal record for gamma replay (needs --gamma 0 --rule_k >0 --repeat 1)\n"
                "         [--trace_ins 1]  with --trace: add the queue insertion counts for Patience replay (header V31Q)\n"
                "         [--patience_p P] [--patience_s 0.995] [--patience_queue ef|ki]  queue-saturation early termination\n"
                "         [--gt_out PATH]  merged per-shard top-ki as approximate ground truth (ivec, K columns)\n"
                "         [--selector batch|bitmap]  membership data structure for in-filtering\n"
                "  oracle --shards DIR --S 5 --query Q.fbin --gt gt.ivec [--K 20] [--ki 12] [--ef 200]\n"
                "         [--dump PATH]  ndis at the moment the last true neighbour arrived (lower bound)\n"
                "  mask   --base X.fbin --out mask.ivec --sel 0.01 [--kind random|cluster] [--seed 0]\n"
                "  ranks  --shards DIR --S 5 --query Q.fbin --gt gt.ivec [--K 20] [--ki 12] [--ef 200]\n"
                "         [--rule_k 12] [--gamma 0] [--J 8]   measures the rank-weighted decomposition w_j * p_j\n"
                "  align  --base X.fbin --query Q.fbin --gt gt.ivec --gt_filtered gtf.ivec [--out PATH]\n"
                "  [--filter mask.ivec] on gt/search/oracle restricts answers to that mask\n");
        return 1;
    }
    std::string cmd = argv[1];
    Args a = parse(argc, argv, 2);
    // faiss switches the visited set from an array to an unordered_set when the index has at least
    // visited_table_hashset_threshold points (default 10M). The 10M-point dataset at S=1 is exactly
    // 10M, so only S=1 would run with a different data structure. Comparisons must be able to pin
    // that switch (the paper uses --vt_threshold 999999999).
    if (!a.get("vt_threshold").empty())
        faiss::visited_table_hashset_threshold = size_t(a.getd("vt_threshold", 1e7));
    try {
        if (cmd == "build") return cmd_build(a);
        if (cmd == "add") return cmd_add(a);
        if (cmd == "gt") return cmd_gt(a);
        if (cmd == "search") return cmd_search(a);
        if (cmd == "oracle") return cmd_oracle(a);
        if (cmd == "ranks") return cmd_ranks(a);
        if (cmd == "mask") return cmd_mask(a);
        if (cmd == "align") return cmd_align(a);
    } catch (const std::exception& e) {
        fprintf(stderr, "!! %s\n", e.what());
        return 3;
    }
    fprintf(stderr, "unknown subcommand: %s\n", cmd.c_str());
    return 1;
}
