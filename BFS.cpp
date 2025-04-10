#include "solution.hpp"
#include <omp.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include <string>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <algorithm>
#include <cinttypes>
#include <atomic>
#include <random>
#include <limits.h>
#include <map>
#include <unordered_map>
#include <unordered_set>


#include <climits>

#include <parlay/delayed_sequence.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <parlay/utilities.h>
#include <parlay/slice.h>

// direction optimization
// reorder optimization
// quantization optimization
// 

#define PART_SIZE (128 * 1024)
#define PROP_SIZE_8      1
#define PROP_SIZE_16     2
#define PROP_SIZE_32     4
#define SAMPLE_NUM       8
#define CLUSTER_NUM      32
#define KMEANS_ITER     10

typedef std::atomic<int8_t> distType8;
typedef std::atomic<int16_t> distType16;
typedef std::atomic<int> distType;

// for frontorder
typedef long long dgt_type;
typedef unsigned char dist_type;

// 自定义哈希函数
struct VectorHash {
    size_t operator()(const std::vector<unsigned>& vec) const {
        size_t hash = 0;
        for (unsigned num : vec) {
            hash ^= std::hash<unsigned>()(num) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// 自定义比较函数（默认 == 即可，但可以显式声明）
struct VectorEqual {
    bool operator()(const std::vector<unsigned>& a, const std::vector<unsigned>& b) const {
        return a == b;
    }
};

dist_type calculate_dis_v2v(std::vector<unsigned> vec1,std::vector<unsigned> vec2){
    dist_type diff = 0;
    for(size_t i = 0; i < vec1.size();i++){
        diff += ((vec1[i]==vec2[i])?0:1);
    }
    return diff;
}

std::vector<unsigned> digit2vec(int digit, dgt_type number, int feat_size){
    std::vector<unsigned> raw;
    while (number > 0 && raw.size() < feat_size) {
        raw.push_back(number % digit); 
        number /= digit;
    }
    while (raw.size() < feat_size) {
        raw.push_back(0); 
    }
    std::reverse(raw.begin(), raw.end());
    if(raw.size()!=feat_size){
        printf("decompress error!");
        exit(0);
    }
    return raw;
}

// atomic operations
template<typename T, typename U>
T fetch_and_add(T &x, U inc) {
    return __sync_fetch_and_add(&x, inc);
}

template<typename T>
bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
    return __sync_bool_compare_and_swap(&x, old_val, new_val);
}

template <typename T>
inline bool atomic_write_min(std::atomic<T> &atomic_var, T new_val) {
    T current_val = atomic_var.load(std::memory_order_relaxed);
    while (new_val < current_val) {
        if (atomic_var.compare_exchange_weak(current_val, new_val, std::memory_order_relaxed)) {
            return true; // 成功更新为新值
        }
    }
    return false; // 如果 new_val >= current_val，则返回 false
}

// sliding queue
template <typename T>
class QueueBuffer;

template <typename T>
class SlidingQueue {
    T *shared;
    size_t shared_in;
    size_t shared_out_start;
    size_t shared_out_end;
    friend class QueueBuffer<T>;

    public:
        SlidingQueue()
        : shared(nullptr), shared_in(0), shared_out_start(0), shared_out_end(0) {}
        
        explicit SlidingQueue(size_t shared_size) {
            shared = new T[shared_size];
            reset();
        }

        ~SlidingQueue() {
            delete[] shared;
        }

        void push_back(T to_add) {
            shared[shared_in++] = to_add;
        }

        void set_size(size_t shared_size) {
            if (shared != nullptr) {
                delete[] shared;
            }
            shared = new T[shared_size];
            reset();
        }

        bool empty() const {
            return shared_out_start == shared_out_end;
        }

        void reset() {
            shared_out_start = 0;
            shared_out_end = 0;
            shared_in = 0;
        }

        void slide_window() {
            shared_out_start = shared_out_end;
            shared_out_end = shared_in;
        }

        void print() {
            // printf("shared_in = %zu\n", shared_in);
            printf("shared_out(%zu, %zu) -> size = %zu\n",
                shared_out_start, shared_out_end, size());
        }

        typedef T* iterator;

        iterator begin() const {
            return shared + shared_out_start;
        }

        iterator end() const {
            return shared + shared_out_end;
        }

        size_t size() const {
            return end() - begin();
        }
};

template <typename T>
class QueueBuffer {
    size_t in;
    T *local_queue;
    SlidingQueue<T> *sq;
    size_t local_size;
    static SlidingQueue<T> dummy_queue; 

    public:
        QueueBuffer() // default constructor
            : in(0), sq(&dummy_queue), local_size(128) 
        {
            // in = 0;
            local_queue = new T[local_size];
        }
        
        QueueBuffer(size_t buf_size) // default constructor
            : in(0), sq(&dummy_queue) 
        {
            // in = 0;
            local_size = buf_size;
            local_queue = new T[local_size];
            // printf("buf size = %lu\n", buf_size);
        }

        explicit QueueBuffer(SlidingQueue<T> &master, size_t given_size = 16384)
            : in(0), sq(&master), local_size(given_size) 
        {
            // in = 0;
            local_queue = new T[local_size];
        }

        ~QueueBuffer() {
            delete[] local_queue;
        }

        // 初始化 SlidingQueue
        void set_master(SlidingQueue<T> &master) {
            sq = &master;
        }

        void realloc(size_t new_size) {
            if(new_size == local_size)
                return;
            delete[] local_queue;
            local_queue = new T[new_size];
            local_size = new_size;
        }

        void push_back(T to_add) {
            if (in == local_size)
                flush();
            local_queue[in++] = to_add;
        }

        size_t size() {
            return in;
        }

        size_t bag_size() {
            return sq->shared_in;
        }

        void flush() {
            if(in == 0)
                return;
            if(!sq || sq == &dummy_queue){
                printf("buffer map not right");
                return;
            }
            T *shared_queue = sq->shared;
            size_t copy_start = fetch_and_add(sq->shared_in, in);
            std::copy(local_queue, local_queue+in, shared_queue+copy_start);
            in = 0;
        }
};

template <typename T>
SlidingQueue<T> QueueBuffer<T>::dummy_queue(1024);

// thread-safe bitmap
class Bitmap {
    public:
        explicit Bitmap(size_t size) {
            uint64_t num_words = (size + kBitsPerWord - 1) / kBitsPerWord;
            start_ = new uint64_t[num_words];
            end_ = start_ + num_words;
        }

        ~Bitmap() {
            delete[] start_;
        }

        void reset() {
            std::fill(start_, end_, 0);
        }

        void set_bit(size_t pos) {
            start_[word_offset(pos)] |= ((uint64_t) 1l << bit_offset(pos));
        }

        void set_bit_atomic(size_t pos) {
            uint64_t old_val, new_val;
            do {
                old_val = start_[word_offset(pos)];
                new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
            } while (!compare_and_swap(start_[word_offset(pos)], old_val, new_val));
        }

        void clear_bit_atomic(size_t pos) {
            uint64_t old_val, new_val;
            do {
                old_val = start_[word_offset(pos)];
                new_val = old_val & ~((uint64_t)1l << bit_offset(pos));
            } while (!compare_and_swap(start_[word_offset(pos)], old_val, new_val));
        }

        bool get_bit(size_t pos) const {
            return (start_[word_offset(pos)] >> bit_offset(pos)) & 1l;
        }

        void swap(Bitmap &other) {
            std::swap(start_, other.start_);
            std::swap(end_, other.end_);
        }

    private:
        uint64_t *start_;
        uint64_t *end_;

        static const uint64_t kBitsPerWord = 64;
        static uint64_t word_offset(size_t n) { return n / kBitsPerWord; }
        static uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }
};

template <typename T_>
class pvector {
    public:
        typedef T_* iterator;
        pvector() : start_(nullptr), end_size_(nullptr), end_capacity_(nullptr) {}

        explicit pvector(size_t num_elements) {
            start_ = new T_[num_elements];
            end_size_ = start_ + num_elements;
            end_capacity_ = end_size_;
        }

        pvector(size_t num_elements, T_ init_val) : pvector(num_elements) {
            fill(init_val);
        }

        pvector(iterator copy_begin, iterator copy_end)
            : pvector(copy_end - copy_begin) {
            #pragma omp parallel for
            for (size_t i=0; i < capacity(); i++)
                start_[i] = copy_begin[i];
        }

        // don't want this to be copied, too much data to move
        pvector(const pvector &other) = delete;

        // prefer move because too much data to copy
        pvector(pvector &&other)
            : start_(other.start_), end_size_(other.end_size_),
            end_capacity_(other.end_capacity_) {
            other.start_ = nullptr;
            other.end_size_ = nullptr;
            other.end_capacity_ = nullptr;
        }

        // want move assignment
        pvector& operator= (pvector &&other) {
            if (this != &other) {
                ReleaseResources();
                start_ = other.start_;
                end_size_ = other.end_size_;
                end_capacity_ = other.end_capacity_;
                other.start_ = nullptr;
                other.end_size_ = nullptr;
                other.end_capacity_ = nullptr;
            }
            return *this;
        }

        void ReleaseResources(){
            if (start_ != nullptr) {
                delete[] start_;
            }
        }

        ~pvector() {
            ReleaseResources();
        }

        // not thread-safe
        void reserve(size_t num_elements) {
            if (num_elements > capacity()) {
                T_ *new_range = new T_[num_elements];
                #pragma omp parallel for
                for (size_t i=0; i < size(); i++)
                new_range[i] = start_[i];
                end_size_ = new_range + size();
                delete[] start_;
                start_ = new_range;
                end_capacity_ = start_ + num_elements;
            }
        }

        // prevents internal storage from being freed when this pvector is desctructed
        // - used by Builder to reuse an EdgeList's space for in-place graph building
        void leak() {
            start_ = nullptr;
        }

        bool empty() {
            return end_size_ == start_;
        }

        void clear() {
            end_size_ = start_;
        }

        void resize(size_t num_elements) {
            reserve(num_elements);
            end_size_ = start_ + num_elements;
        }

        T_& operator[](size_t n) {
            return start_[n];
        }

        const T_& operator[](size_t n) const {
            return start_[n];
        }

        void push_back(T_ val) {
            if (size() == capacity()) {
                size_t new_size = capacity() == 0 ? 1 : capacity() * growth_factor;
                reserve(new_size);
            }
            *end_size_ = val;
            end_size_++;
        }

        void fill(T_ init_val) {
            #pragma omp parallel for
            for (T_* ptr=start_; ptr < end_size_; ptr++)
                *ptr = init_val;
        }

        size_t capacity() const {
            return end_capacity_ - start_;
        }

        size_t size() const {
            return end_size_ - start_;
        }

        iterator begin() const {
            return start_;
        }

        iterator end() const {
            return end_size_;
        }

        T_* data() const {
            return start_;
        }

        void swap(pvector &other) {
            std::swap(start_, other.start_);
            std::swap(end_size_, other.end_size_);
            std::swap(end_capacity_, other.end_capacity_);
        }

    private:
        T_* start_;
        T_* end_size_;
        T_* end_capacity_;
        static const size_t growth_factor = 2;
};

class Node {
public:
    int id;
    int cluster_id;
    std::vector<unsigned> feat = std::vector<unsigned>(SAMPLE_NUM, 255);
    dgt_type digits;
    int digit_id;
    dist_type min_dis;
};


class Graph : 
public BaseGraph {
    eidType* rowptr;
    vidType* col;
    uint64_t N;
    uint64_t M;
    public:
        /// property
        std::string name;
        uint64_t deg;
        
        // process option
        bool dense;
        bool reorder;
        bool lowd;

        // sparse
        int sparse_queue_size;
        size_t sparse_buffer_size;
        int LOG2N;
        size_t num_bags;
        pvector<SlidingQueue<vidType>> bags;
        pvector<std::atomic<uint8_t>> bag_id;

        // lowd utils
        pvector<vidType> isolated_nodes;
        pvector<vidType> degree_one_nodes;
        pvector<vidType> single_neighbor;
        // old2new and new2old(lowd)
        pvector<int> old2new;
        pvector<vidType> new2old;
        // variable
        vidType* new_id;
        vidType* old_id;
        int hop;
        Graph(eidType* rowptr_, vidType* col_, uint64_t N_, uint64_t M_) :
        rowptr(nullptr), col(nullptr), N(0), M(0){
            deg = M_ / N_;
            dense = true;
            reorder = false;
            if(deg <= 2){
                dense = true;
                lowd = true;
            }
            else if(deg > 2 && deg <= 10){
                dense = false;
            }
            else{
                if(N_ > 9000000)
                    reorder = false;
                else
                    reorder = true;
            }
            sparse_queue_size = 8;
            sparse_buffer_size = 128;

            // printf("[before preprocess]:\nM = %lu, N = %lu, d = %lu\n", M_, N_, deg);
            // preprocessing for lowd
            if(lowd){
                preprocess_lowd(rowptr_, col_, N_, M_ );
            }
            else{
                rowptr = rowptr_;
                col = col_;
                M = M_;
                N = N_;
            }
            // preprocessing for reordering
            if(reorder){
                new_id = new vidType[N];
                old_id = new vidType[N];
                auto start = std::chrono::high_resolution_clock::now();
                // corder();
                FrontOrderFast();
                preprocess_reorder_graph();
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                std::cout << "reorder latency: " << duration.count() << " ms" << std::endl;
            }
            // auto start = std::chrono::high_resolution_clock::now();
            // FrontOrderFast();
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> duration = end - start;
            // std::cout << "reorder latency: " << duration.count() << " ms" << std::endl;
        }
        
        ~Graph() {
            ///////// for reordering
            if(reorder){
                delete[] new_id;
                delete[] old_id;
            }
        }

        void preprocess_lowd(eidType* rowptr_, vidType* col_, uint64_t N_, uint64_t M_){
            uint64_t newN, newM;
            pvector<eidType> degrees(N_);
            #pragma omp parallel for
            for(vidType i = 0;i < N_;i++)
                degrees[i] = rowptr_[i+1] - rowptr_[i];
            
            old2new = pvector<int>(N_, -1);
            single_neighbor = pvector<vidType>(N_, 0);

            vidType valid_count = 0;
            for(vidType i = 0;i < N_;i++){
                if(degrees[i] == 0){
                    isolated_nodes.push_back(i);
                    old2new[i] = -1;
                }
                else if(degrees[i] == 1){
                    degree_one_nodes.push_back(i);
                    old2new[i] = -2;
                }
                else
                    old2new[i] = valid_count++;
            }
            newN = valid_count;
            new2old.resize(newN);
            
            #pragma omp parallel for
            for(vidType i = 0;i < N_;i++){
                int new_id = old2new[i];
                if (new_id >= 0) 
                    new2old[new_id] = i;
            }

            #pragma omp parallel for
            for(vidType i = 0;i < degree_one_nodes.size();i++){
                vidType u = degree_one_nodes[i];
                single_neighbor[u] = col_[rowptr_[u]];
            }

            // new rowptr and col
            pvector<eidType> new_degs(newN, 0);
            #pragma omp parallel for
            for (vidType i = 0; i < N_; i++) {
                int new_i = old2new[i];
                if (new_i < 0) 
                    continue;
                for (eidType e = rowptr_[i]; e < rowptr_[i+1]; e++) {
                    vidType nbr = col_[e];
                    int new_nbr = old2new[nbr];
                    if (new_nbr >= 0) 
                        new_degs[new_i]++;
                }
            }

            eidType* new_rowptr_ = new eidType[newN + 1];
            new_rowptr_[0] = 0;
            for (vidType i = 1; i <= newN; i++) {
                new_rowptr_[i] = new_rowptr_[i - 1] + new_degs[i - 1];
            }
            newM = new_rowptr_[newN];

            vidType* new_col_ = new vidType[newM];
            pvector<eidType> idx_counter(newN, 0);
            #pragma omp parallel for
            for (vidType i = 0; i < N_; i++) {
                int new_i = old2new[i];
                if (new_i < 0) continue;
                for (eidType e = rowptr_[i]; e < rowptr_[i+1]; e++) {
                    vidType nbr = col_[e];
                    int new_nbr = old2new[nbr];
                    if (new_nbr >= 0) {
                        eidType pos = new_rowptr_[new_i] + idx_counter[new_i];
                        new_col_[pos] = new_nbr;
                        idx_counter[new_i]++;
                    }
                }
            }

            rowptr = new_rowptr_;
            col = new_col_;
            N = newN; 
            M = newM;
            // printf("deg0 = %lu, deg1 = %lu, other vertices = %lu, edge num = %lu\n", 
            //         isolated_nodes.size(), degree_one_nodes.size(), N, M);
        }

        void preprocess_reorder_graph(){
            unsigned max_threads = omp_get_max_threads();
            std::vector<unsigned> new_degree(N, 0);
            #pragma omp parallel for schedule(static) num_threads(max_threads)
            for(unsigned i = 0; i < N; i++){
                new_degree[new_id[i]] = rowptr[i+1] - rowptr[i];
            }

            std::vector<eidType> new_row(N + 1, 0);
            for(int i = 0;i < N;i++){
                new_row[i + 1] = new_row[i] + new_degree[i];
            }
            std::vector<vidType> new_col(M, 0);
            #pragma omp parallel for schedule(dynamic) num_threads(max_threads)
            for(vidType i = 0; i < N; i++) {
                unsigned count = 0;
                for(eidType j = rowptr[i]; j < rowptr[i + 1]; j++) {
                    new_col[new_row[new_id[i]] + count] = new_id[col[j]];
                    count++;
                }
                std::sort(new_col.begin() + new_row[new_id[i]], new_col.begin() + new_row[new_id[i]] + count);
            }
            std::copy(new_row.begin(), new_row.end(), rowptr);
            std::copy(new_col.begin(), new_col.end(), col);
        }

        void corder(){
            unsigned partition_size = PART_SIZE;
            vidType num_partitions = (N - 1) / partition_size + 1;

            std::vector<vidType> deg(N);
            #pragma omp parallel for
            for(vidType i = 0;i < N;i++)
                deg[i] = rowptr[i+1] - rowptr[i];

            unsigned max_threads = omp_get_max_threads();
            std::vector<vidType> segment_large;
            segment_large.reserve(N);
            std::vector<vidType> segment_small;
            segment_small.reserve(N/2);
            unsigned average_degree = M / N;
            for(unsigned i = 0; i < N; i++){
                if(deg[i] > 1 * average_degree)
                    segment_large.push_back(i);
                else
                    segment_small.push_back(i);
            }
            unsigned num_large_per_seg = ceil((float) segment_large.size() / num_partitions);
            unsigned num_small_per_seg = partition_size - num_large_per_seg;
            unsigned last_cls = num_partitions - 1;

            while( (num_large_per_seg * last_cls > segment_large.size()) ||
                    (num_small_per_seg * last_cls > segment_small.size())) 
            {
                last_cls -= 1;
            }
                
            #pragma omp parallel for schedule(static) num_threads(max_threads)
            for(unsigned i = 0; i < last_cls; i++) {
                unsigned index = i * partition_size;
                for(unsigned j = 0; j < num_large_per_seg; j++) {
                    old_id[index] = segment_large[i * num_large_per_seg + j];
                    new_id[segment_large[i * num_large_per_seg + j]] = index++;
                }
                for(unsigned j = 0; j < num_small_per_seg; j++){
                    old_id[index] = segment_small[i * num_small_per_seg + j];
                    new_id[segment_small[i * num_small_per_seg + j]] = index++;
                }
            }
            auto last_large = num_large_per_seg * last_cls;
            auto last_small = num_small_per_seg * last_cls;
            unsigned index = last_cls * partition_size;

            for(unsigned i = last_large; i < segment_large.size(); i++) {
                old_id[index] = segment_large[i];
                new_id[segment_large[i]] = index++;
            }
            for(unsigned i = last_small; i < segment_small.size(); i++) {
                old_id[index] = segment_small[i];
                new_id[segment_small[i]] = index++;
            }
        }

        void FrontOrderFast() {
            std::vector<vidType> deg(N);
            #pragma omp parallel for
            for(vidType i = 0;i < N;i++)
                deg[i] = rowptr[i+1] - rowptr[i];

            dist_type codebook[128][262144] = {0}; // a codebook from clusterId to vecId
            std::vector<Node> nodes(N);
            unsigned digit = 0;
            weight_type* distances = new weight_type[N];
            for(int i = 0;i < SAMPLE_NUM;i++){
                std::random_device rd;  // 获取随机种子
                std::mt19937 gen(rd()); // 选择Mersenne Twister 伪随机数生成器
                std::uniform_int_distribution<int> dist(0, N - 1); // 生成 [0, N] 之间的整数

                int source = dist(gen);
                bfs_dense<distType8>(source, distances);
                #pragma omp parallel for
                for(int j = 0;j < N;j++){
                    if(distances[j] == 4294967295)
                        nodes[j].feat[i] = 255;
                    else{
                        nodes[j].feat[i] = distances[j];
                        if(distances[j] > digit){           
                            #pragma omp critical
                                digit = distances[j];
                        }
                    }
                }
            }
            printf("digit = %d\n", digit);
            
            // init nodes
            int tnum = omp_get_max_threads();
            printf("max thread num = %d\n", tnum);
            #pragma omp parallel for
            for(unsigned i = 0;i < N;i++){
                nodes[i].id = i;
                nodes[i].cluster_id = -1;
                nodes[i].min_dis = std::numeric_limits<dist_type>::max();
                nodes[i].digits = 0;
                for (size_t j = 0; j < nodes[i].feat.size(); j++){
                    if (nodes[i].digits > LLONG_MAX / digit) {
                        printf("overflow %lld\n", nodes[i].digits);
                        throw std::overflow_error("Multiplication overflow\n");
                    }
                    nodes[i].digits = nodes[i].digits * digit + nodes[i].feat[j]; // compress to digits
                }
            }
            
            // 去重
            std::vector<std::unordered_set<dgt_type>> thread_sets(tnum);
            #pragma omp parallel for num_threads(tnum)
            for(int i = 0; i < tnum;i++)
            {
                auto& local_set = thread_sets[i];
                int chunk_size = (N + tnum - 1) / tnum;  // 向上取整分配
                int start = i * chunk_size;
                int end = std::min(start + chunk_size, (int)N);
                for (int j = start; j < end; j++) {
                    local_set.insert(nodes[j].digits);
                }
            }
            std::unordered_set<dgt_type> global_set;
            for (const auto& local_set : thread_sets) {
                global_set.insert(local_set.begin(), local_set.end());
            }
            int digit_num = global_set.size();
            printf("Global set size = %d\n", global_set.size());

            // 获得digit到id的映射
            std::unordered_map<dgt_type, int> digit_to_id;
            std::unordered_map<int, dgt_type> id_to_digit;
            int id = 0;
            for(std::unordered_set<dgt_type>::iterator it = global_set.begin(); it != global_set.end(); ++it)
            {
                dgt_type dgt = *it;
                digit_to_id[dgt] = id;
                id_to_digit[id] = dgt;
                id++;
            }
            
            // 对每个节点, 转换digit_id
            std::vector<int> frequency(digit_num, 0);
            #pragma omp parallel for num_threads(tnum)
            for(unsigned i = 0;i < N;i++){
                int did = digit_to_id[nodes[i].digits];
                nodes[i].digit_id = did; // get digit id
                #pragma omp atomic
                    frequency[did] += 1; // 获取频率
            }
            
            
            // K-means初始化
            std::vector<Node> centroids; // 质心数组
            std::vector<Node> active_centroids; // 活跃质心
            unsigned num_clusters = CLUSTER_NUM;
            printf("cluster num = %d\n", CLUSTER_NUM);
            std::vector<int> indices(digit_num);
            for (int i = 0; i < digit_num; ++i) {
                indices[i] = i;
            }
            std::sort(indices.begin(), indices.end(), [&frequency](int a, int b) {
                return (frequency[a] > frequency[b]); // 按值比较，但排序的是索引
            });
            for(int i = 0; i < num_clusters;i++){
                Node center_node;
                center_node.cluster_id = i;
                center_node.digit_id = indices[i];
                center_node.digits = id_to_digit[indices[i]];
                std::vector<unsigned> raw = digit2vec(digit, center_node.digits, nodes[0].feat.size());
                for(int j = 0;j < SAMPLE_NUM;j++)
                    center_node.feat[j] = raw[j];
                // printf("---------\n");
                // printf("center(%d):\n", i);
                // printf("  digit id = %d\n", center_node.digit_id);
                // printf("  digits = %lld\n", center_node.digits);
                // printf("  frequency = %d\n", frequency[indices[i]]);
                // for(int l = 0;l < raw.size();l++)
                //     printf("%u ", raw[l]);
                // printf("\n");
                centroids.push_back(center_node);
                active_centroids.push_back(center_node);
            }

            // 获取digit_id之间的codebook
            for (int i = 0; i < num_clusters;i++){
                dgt_type dg1 = centroids[i].digits;
                std::vector<unsigned> raw1 = digit2vec(digit, dg1, nodes[0].feat.size());
                #pragma omp parallel for num_threads(tnum)
                for(int j = 0; j < digit_num;j++){
                    dgt_type dg2 = id_to_digit[j];
                    std::vector<unsigned> raw2 = digit2vec(digit, dg2, nodes[0].feat.size());
                    codebook[i][j] = calculate_dis_v2v(raw1, raw2);
                }
            }

            // K-means
            unsigned int iter = 0;
            double converge_rate = 0.1;
            std::vector<std::vector<unsigned>> clusters(num_clusters); // 聚类数组
            while (iter < KMEANS_ITER) {
                printf("----------\nKmeans iter(%d)\n", iter);
                for (auto& cluster : clusters)
                    cluster.clear();
                // 计算距离
                #pragma omp parallel for num_threads(tnum)
                for (unsigned i = 0;i < nodes.size();i++) {
                    for (int j = 0; j < active_centroids.size(); j++) {
                        dist_type diff = codebook[active_centroids[j].cluster_id][nodes[i].digit_id];
                        if (diff < nodes[i].min_dis) {
                            nodes[i].min_dis = diff;
                            nodes[i].cluster_id = active_centroids[j].cluster_id;
                        }
                    }
                }
                // 重新分配cluster
                std::vector<std::vector<std::vector<unsigned>>> local_clusters(tnum, 
                                                                std::vector<std::vector<unsigned>>(clusters.size()));
                #pragma omp parallel for num_threads(tnum)
                for(int i = 0; i < tnum;i++)
                {
                    int chunk_size = (nodes.size() + tnum - 1) / tnum;  // 向上取整分配
                    int start = i * chunk_size;
                    int end = std::min(start + chunk_size, (int)nodes.size());
                    for (int j = start; j < end; j++) {
                        local_clusters[i][nodes[j].cluster_id].push_back(nodes[j].id);
                    }
                }
                #pragma omp parallel for num_threads(tnum)
                for(int i = 0;i < clusters.size();i++){
                    for(int j = 0;j < tnum;j++){
                        clusters[i].insert(clusters[i].end(), local_clusters[j][i].begin(), local_clusters[j][i].end());
                    }
                }
                
                // 更新质心位置为聚类内节点的平均值
                active_centroids.clear();
                std::vector<Node> inactive_centroids; // 不活跃质心
                inactive_centroids.clear();
                unsigned num_not_converged = 0;
                #pragma omp parallel for num_threads(tnum)
                for (size_t i = 0; i < centroids.size(); ++i) {
                    Node prev = centroids[i];
                    if (!clusters[i].empty()) {
                        unsigned dim = nodes[0].feat.size();
                        std::vector<unsigned> newCentroid(dim, 0);
                        dgt_type newCentroidDigit = 0;
                        for (size_t j = 0; j < dim; ++j){
                            for (const auto& nodeId : clusters[i])
                                newCentroid[j] += nodes[nodeId].feat[j];
                            double result = static_cast<double>(newCentroid[j]) / clusters[i].size();
                            newCentroid[j] =  static_cast<unsigned>(std::round(result));
                            newCentroidDigit = newCentroidDigit * digit + newCentroid[j];
                        }
                        centroids[i].digits = newCentroidDigit;
                        // centroids[i].feat = newCentroid;
                        for(int j = 0;j < SAMPLE_NUM;j++)
                            centroids[i].feat[j] = newCentroid[j];
                        dist_type dis = calculate_dis_v2v(prev.feat, newCentroid);
                        if(dis > 0){
                            num_not_converged++;
                            active_centroids.push_back(centroids[i]);
                            for(int j = 0; j < digit_num;j++){
                                std::vector<unsigned> raw1 = digit2vec(digit, id_to_digit[j], nodes[0].feat.size());
                                codebook[centroids[i].cluster_id][j] = calculate_dis_v2v(raw1, newCentroid);
                            }
                            printf("cluster(%d): (", centroids[i].cluster_id);
                            for(int l = 0;l < prev.feat.size();l++)
                                printf("%u ", prev.feat[l]);
                            printf(") -> (");
                            for(int l = 0;l < centroids[i].feat.size();l++)
                                printf("%u ", centroids[i].feat[l]);
                            printf(")\n");
                        }
                        else
                            inactive_centroids.push_back(centroids[i]);
                    }
                }
                printf("active set = %d, inactive set = %d\n", active_centroids.size(), inactive_centroids.size());
                // 更新距离列表
                for(int i = 0;i < active_centroids.size();i++){
                    int active_cluster_id = active_centroids[i].cluster_id;
                    std::vector<unsigned> cluster = clusters[active_cluster_id];
                    // printf("cluster(%d): size = %d\n", active_cluster_id, cluster.size());
                    #pragma omp parallel for num_threads(tnum)
                    for(int j = 0;j < cluster.size();j++){
                        int nodeId = cluster[j];
                        nodes[nodeId].min_dis = codebook[active_cluster_id][nodes[nodeId].digit_id];
                        nodes[nodeId].cluster_id = active_cluster_id;
                        for(int k = 0;k < inactive_centroids.size();k++){
                            dist_type diff = codebook[inactive_centroids[k].cluster_id][nodes[nodeId].digit_id];
                            if (diff < nodes[nodeId].min_dis) {
                                nodes[nodeId].min_dis = diff;
                                nodes[nodeId].cluster_id = inactive_centroids[k].cluster_id;
                            }
                        }
                    }
                }
                iter++;
                
                if(num_not_converged <= converge_rate * num_clusters){
                    std::cout << "Finally: num not converged = " << num_not_converged << std::endl;
                    break;
                }

            }
            // for (unsigned i = 0;i < clusters.size();i++) 
            //     printf("cluster(%d) size = %d\n", i, clusters[i].size());

            unsigned partition_size = PART_SIZE;
            vidType num_partitions = (N - 1) / partition_size + 1;

            std::vector<std::vector<std::vector<Node>>> local_parts(tnum,
                std::vector<std::vector<Node>>(num_partitions)
            );

            std::vector<std::vector<Node>> parts(num_partitions);
            const auto average_degree = M / N;
            // 并行处理每个 cluster
            #pragma omp parallel
            {
                int tid = omp_get_thread_num(); // 当前线程 ID

                // 对 clusters 做并行 for 循环
                #pragma omp for
                for(unsigned cid = 0; cid < clusters.size(); cid++) {
                    // 先区分 large_vertex 和 small_vertex
                    std::vector<Node> large_vertex;
                    std::vector<Node> small_vertex;
                    large_vertex.reserve(clusters[cid].size());
                    small_vertex.reserve(clusters[cid].size());

                    for(unsigned nid = 0; nid < clusters[cid].size(); nid++) {
                        Node nd = nodes[clusters[cid][nid]];
                        if(deg[nd.id] > average_degree) {
                            large_vertex.push_back(nd);
                        } else {
                            small_vertex.push_back(nd);
                        }
                    }

                    int local_p_id = 0;
                    // 处理 large_vertex
                    for(const auto &nd : large_vertex) {
                        if(local_parts[tid][local_p_id].size() < partition_size) {
                            local_parts[tid][local_p_id].push_back(nd);
                            local_p_id = (local_p_id + 1) % (num_partitions - 1);
                        } 
                        else {
                            local_p_id = (local_p_id + 1) % (num_partitions - 1);
                            local_parts[tid][num_partitions - 1].push_back(nd);
                        }
                    }

                    // 处理 small_vertex
                    unsigned seg_size = small_vertex.size() / (num_partitions - 1);
                    for(unsigned seg = 0; seg < num_partitions - 1; seg++){
                        for(unsigned sid = seg * seg_size; sid < (seg + 1) * seg_size; sid++){
                            if(local_parts[tid][seg].size() < partition_size) {
                                local_parts[tid][seg].push_back(small_vertex[sid]);
                            } else {
                                local_parts[tid][num_partitions - 1].push_back(small_vertex[sid]);
                            }
                        }
                    }

                    // 处理余下的 small_vertex
                    for(unsigned sid = seg_size * (num_partitions - 1); sid < small_vertex.size(); sid++){
                        if(local_parts[tid][local_p_id].size() < partition_size) {
                            local_parts[tid][local_p_id].push_back(small_vertex[sid]);
                            local_p_id = (local_p_id + 1) % (num_partitions - 1);
                        } else {
                            local_p_id = (local_p_id + 1) % (num_partitions - 1);
                            local_parts[tid][num_partitions - 1].push_back(small_vertex[sid]);
                        }
                    }
                }
            } // 并行区域结束

            // ===== 合并阶段 =====
            // 将所有线程在 local_parts 中的结果合并到全局 parts
            for(int t = 0; t < tnum; t++) {
                for(unsigned p = 0; p < num_partitions; p++) {
                    // 把 local_parts[t][p] 中的节点插入 parts[p] 末尾
                    parts[p].insert(parts[p].end(),
                                    local_parts[t][p].begin(),
                                    local_parts[t][p].end());
                }
            }
            
            unsigned int new_cnt = 0;
            for(unsigned pid = 0; pid < parts.size();pid++){
                for(unsigned nid = 0; nid < parts[pid].size(); nid++){
                    unsigned int oid = parts[pid][nid].id;
                    new_id[oid] = new_cnt;
                    old_id[new_cnt] = oid;
                    new_cnt++;
                }
            }
            printf("[!!]Reorder finished, new count = %d, N = %d\n", new_cnt, N);
            // std::cout << "==========" << std::endl;
            return;
        }

        template <typename AtomicT>
        pvector<AtomicT> InitDistances() {
            pvector<AtomicT> dist(N);
            #pragma omp parallel for
            for (vidType n = 0; n < N; n++) {
                // int outd = rowptr[n + 1] - rowptr[n];
                // dist[n].store((outd != 0) ? -outd : -1, std::memory_order_relaxed);
                dist[n].store(-1, std::memory_order_relaxed);
            }

            return dist;
        }

        template <typename AtomicT>
        int64_t BUStep(pvector<AtomicT> &dist, Bitmap &front, Bitmap &next) {
            int64_t awake_count = 0;
            next.reset();
            #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
            for (vidType u = 0; u < N; u++) {
                if (dist[u].load(std::memory_order_relaxed) < 0) {
                    for (eidType v = rowptr[u]; v < rowptr[u + 1]; v++) {
                        vidType src = col[v];
                        if (front.get_bit(src)) {
                            dist[u].store(hop + 1, std::memory_order_relaxed);
                            awake_count++;
                            next.set_bit(u);
                            break;
                        }
                    }
                }
            }
            return awake_count;
        }

        template <typename AtomicT>
        void TDStep(pvector<AtomicT> &dist, SlidingQueue<vidType> &queue) {
            // int64_t scout_count = 0;
            using ValueType = typename AtomicT::value_type;
            #pragma omp parallel
            {
                QueueBuffer<vidType> lqueue(queue);
                #pragma omp for nowait // reduction(+ : scout_count) 
                for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
                    vidType u = *q_iter;
                    for (eidType v = rowptr[u]; v < rowptr[u + 1]; v++) {
                        vidType dst = col[v];
                        ValueType curr_val = dist[dst].load(std::memory_order_relaxed);
                        if (curr_val < 0) {
                            if (dist[dst].compare_exchange_strong(curr_val, static_cast<ValueType>(hop + 1), std::memory_order_relaxed)) {
                                lqueue.push_back(dst);
                                // scout_count += -curr_val;
                            }
                        }
                    }
                }
                lqueue.flush();
            }
            // return scout_count;
        }

        void QueueToBitmap(const SlidingQueue<vidType> &queue, Bitmap &bm) {
            #pragma omp parallel for
            for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
                vidType u = *q_iter;
                bm.set_bit_atomic(u);
            }
        }

        void BitmapToQueue(const Bitmap &bm, SlidingQueue<vidType> &queue) {
            #pragma omp parallel
            {
                QueueBuffer<vidType> lqueue(queue);
                #pragma omp for nowait
                for (vidType n = 0; n < N; n++){
                    if (bm.get_bit(n))
                        lqueue.push_back(n);
                }
                lqueue.flush();
            }
            queue.slide_window();
        }

        template <typename AtomicT>
        void bfs_dense(vidType source, weight_type * distances){
            int alpha = 15, beta = 18;
            pvector<AtomicT> dist = InitDistances<AtomicT>();
            hop = 0;
            dist[source].store(hop, std::memory_order_relaxed);
            
            SlidingQueue<vidType> queue(N);
            queue.push_back(source);
            queue.slide_window();
            
            Bitmap curr(N);
            curr.reset();
            Bitmap front(N);
            front.reset();
            // int64_t edges_to_check = M;
            // int64_t scout_count = rowptr[source + 1] - rowptr[source];
            while (!queue.empty()) {
                double active_frac = (double)(queue.size()) / N;
                // if (scout_count > edges_to_check / alpha) {
                if (active_frac > 0.02) {
                    int64_t awake_count, old_awake_count;
                    QueueToBitmap(queue, front);
                    awake_count = queue.size();
                    queue.slide_window();
                    do {
                        old_awake_count = awake_count;
                        // printf("hop[%d]: BU, frontier fraction = %.4lf\n", hop, (double)awake_count / N);
                        awake_count = BUStep<AtomicT>(dist, front, curr);
                        hop++;
                        front.swap(curr);
                    } while ((awake_count >= old_awake_count) ||
                            (awake_count > (N / beta)));
                    BitmapToQueue(front, queue);
                    // scout_count = 1;
                }
                else {
                    // edges_to_check -= scout_count;
                    // printf("hop[%d]: TD, frontier = %.4lf\n", hop, (double)(queue.size()) / N);
                    // scout_count = TDStep<AtomicT>(dist, queue);
                    TDStep<AtomicT>(dist, queue);
                    hop++;
                    queue.slide_window();
                }
            }
            #pragma omp parallel for
            for (vidType i = 0; i < N; i++) {
                if (dist[i].load(std::memory_order_relaxed) < 0)
                    distances[i] = 4294967295;
                else{
                    distances[i] = static_cast<weight_type>(dist[i].load(std::memory_order_relaxed));
                }
            }
        }

        template <typename AtomicT>
        void sparse_relax(pvector<AtomicT> &dist, size_t id, Bitmap &in_frontier) {
            int front_bag = id % num_bags;
            using ValueType = typename AtomicT::value_type;
            #pragma omp parallel
            {
                // set buffers
                pvector<QueueBuffer<vidType>> lqueue(num_bags); 
                for (size_t i = 0; i < num_bags; ++i) {
                    lqueue[i].realloc(sparse_buffer_size);
                    lqueue[i].set_master(bags[i]);
                }
                
                #pragma omp for nowait
                for(auto q_iter = bags[front_bag].begin(); q_iter < bags[front_bag].end(); q_iter++) {
                    vidType f = *q_iter;
                    in_frontier.clear_bit_atomic(f);
                    if (id == 0 || id == parlay::log2_up(dist[f].load(std::memory_order_relaxed))) {
                        vidType local_queue[sparse_queue_size];
                        size_t front = 0, rear = 0;
                        local_queue[rear++] = f;
                        while (front < rear) {
                            vidType u = local_queue[front++];
                            for (eidType i = rowptr[u]; i < rowptr[u + 1]; i++) {
                                vidType v = col[i];
                                if (atomic_write_min(dist[v], (ValueType)(dist[u].load(std::memory_order_relaxed) + 1))) {
                                    if (rear < sparse_queue_size) {
                                        local_queue[rear++] = v;
                                    }
                                    else {
                                        uint8_t bg_id = (dist[v] == 0) ? 0 : parlay::log2_up(dist[v].load(std::memory_order_relaxed));
                                        if (in_frontier.get_bit(v) == false) {
                                            in_frontier.set_bit_atomic(v);
                                            atomic_write_min(bag_id[v], bg_id);
                                            lqueue[bg_id % num_bags].push_back(v);
                                        }
                                        else {
                                            if (atomic_write_min(bag_id[v], bg_id)) {
                                                lqueue[bg_id % num_bags].push_back(v);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for(int i = 0;i < num_bags;i++){
                    lqueue[i].flush();
                }
            }
        }

        template <typename AtomicT>
        void bfs_sparse(vidType source, weight_type * distances){
            LOG2N = parlay::log2_up(N);
            num_bags = parlay::log2_up(sparse_queue_size) + 2;
            bag_id = pvector<std::atomic<uint8_t>>(N);
            Bitmap in_frontier(N);
            pvector<AtomicT> dist = pvector<AtomicT>(N);
            // initialization
            using ValueType = typename AtomicT::value_type;
            ValueType DIST_MAX = std::numeric_limits<ValueType>::max();

            #pragma omp parallel for schedule(dynamic, 1024)
            for(vidType i = 0; i < N; i++) {
                if(i == source){
                    dist[i].store(0, std::memory_order_relaxed);
                    bag_id[i].store(0, std::memory_order_relaxed);
                }
                else
                {
                    dist[i].store(DIST_MAX, std::memory_order_relaxed);
                    bag_id[i].store(LOG2N, std::memory_order_relaxed);    
                }
            }
            in_frontier.reset();
            in_frontier.set_bit(source);
            bags = pvector<SlidingQueue<vidType>>(num_bags);
            for(int i = 0;i < num_bags;i++){
                bags[i].set_size(N);
            }
            bags[0].push_back(source);
            for (int i = 0; i <= LOG2N; i++) {
                bags[i % num_bags].slide_window();
                while (!bags[i % num_bags].empty()) {
                    sparse_relax<AtomicT>(dist, i, in_frontier);
                    bags[i % num_bags].slide_window();
                }
            }
            #pragma omp parallel for
            for (vidType i = 0; i < N; i++) {
                if (dist[i].load(std::memory_order_relaxed) == DIST_MAX)
                    distances[i] = 4294967295;
                else
                    distances[i] = static_cast<weight_type>(dist[i].load(std::memory_order_relaxed));
            }
        }


        void BFS(vidType source, weight_type * distances) {
            bool src_deg_1 = false;
            int newsrc = source;
            // low-d optimization
            if(lowd){
                if(old2new[source] == -1){
                    distances[source] = 0;
                    return;
                }
                else if(old2new[source] == -2) {
                    vidType nbr = single_neighbor[source];
                    int new_nbr = old2new[nbr];
                    if (new_nbr < 0) {
                        distances[source] = 0;
                        distances[nbr] = 1; 
                        return;
                    } 
                    else{
                        newsrc = old2new[nbr];
                        src_deg_1 = true;
                    }
                }
                else
                    newsrc = old2new[newsrc];
            }
            // reorder optimization
            if(reorder)
                newsrc = new_id[newsrc];
            if(dense){
                if(lowd)
                    bfs_dense<distType16>(newsrc, distances);
                else
                    bfs_dense<distType8>(newsrc, distances);
            }
            else{
                bfs_sparse<distType16>(newsrc, distances);
            }
            // post processing
            if(reorder){
                auto start = std::chrono::high_resolution_clock::now();
                uint8_t* comp_dist = new uint8_t[N];
                #pragma omp parallel for
                for (vidType i = 0; i < N; i++) {
                    if (distances[i] == 4294967295)
                        comp_dist[i] = 255;
                    else
                        comp_dist[i] = static_cast<uint8_t>(distances[i]);
                }
                #pragma omp parallel for
                for(vidType i = 0; i < N;i++){
                    if(comp_dist[new_id[i]] == 255)
                        distances[i] = 4294967295;
                    else
                        distances[i] = static_cast<weight_type>(comp_dist[new_id[i]]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;
                std::cout << "post reorder latency: " << duration.count() << " ms" << std::endl;
            }
            if(lowd){
                weight_type* comp_dist = new weight_type[N];
                #pragma omp parallel for
                for (vidType i = 0; i < N; i++) {
                    if (distances[i] == 4294967295)
                        comp_dist[i] = 4294967295;
                    else
                        comp_dist[i] = static_cast<weight_type>(distances[i]);
                }
                #pragma omp parallel for
                for(int i = 0;i < isolated_nodes.size();i++){
                    distances[isolated_nodes[i]] = 4294967295;
                }
                if(src_deg_1){
                    distances[source] = 0;
                    #pragma omp parallel for
                    for (vidType i = 0; i < (vidType)new2old.size(); i++) {
                        vidType old_id = new2old[i];
                        if (comp_dist[i] == 4294967295)
                            distances[old_id] = 4294967295; // 大概是 DIST_MAX
                        else
                            distances[old_id] = comp_dist[i]+1;
                    }
                    #pragma omp parallel for
                    for (int i = 0; i < degree_one_nodes.size();i++) {
                        vidType deg1_node = degree_one_nodes[i];
                        if(deg1_node == source)
                            continue;
                        vidType nbr = single_neighbor[deg1_node];
                        int new_nbr = old2new[nbr];
                        if (new_nbr >= 0) {
                            if (comp_dist[new_nbr] == 4294967295)
                                distances[deg1_node] = 4294967295; // unreachable
                            else 
                                distances[deg1_node] = distances[nbr] + 1;
                        } 
                        else
                            distances[deg1_node] = 4294967295; 
                    }
                    #pragma omp parallel for
                    for (int i = 0; i < degree_one_nodes.size();i++) {
                        vidType deg1_node = degree_one_nodes[i];
                        if(deg1_node == source)
                            continue;
                        vidType nbr = single_neighbor[deg1_node];
                        int new_nbr = old2new[nbr];
                        if (new_nbr >= 0) {
                            if (comp_dist[new_nbr] == 4294967295)
                                distances[deg1_node] = 4294967295; // unreachable
                            else 
                                distances[deg1_node] = distances[nbr] + 1;
                        } 
                        else
                            distances[deg1_node] = 4294967295; 
                    }
                }
                else{
                    #pragma omp parallel for
                    for (vidType i = 0; i < (vidType)new2old.size(); i++) {
                        vidType old_id = new2old[i];
                        if (comp_dist[i] == 4294967295)
                            distances[old_id] = 4294967295; // 大概是 DIST_MAX
                        else
                            distances[old_id] = static_cast<weight_type>(comp_dist[i]);
                    }
                    #pragma omp parallel for
                    for (int i = 0; i < degree_one_nodes.size();i++) {
                        vidType deg1_node = degree_one_nodes[i];
                        vidType nbr = single_neighbor[deg1_node];
                        int new_nbr = old2new[nbr];
                        if (new_nbr >= 0) {
                            if (comp_dist[new_nbr] == 4294967295)
                                distances[deg1_node] = 4294967295; // unreachable
                            else 
                                distances[deg1_node] = distances[nbr] + 1;
                        } 
                        else
                            distances[deg1_node] = 4294967295; 
                    }
                }
            }
        }
};

BaseGraph* initialize_graph(eidType* rowptr, vidType* col, 
                            uint64_t N, uint64_t M) {
    Graph* graph = new Graph(rowptr, col, N, M);
    return graph;
}
