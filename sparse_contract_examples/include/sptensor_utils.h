#pragma once

template<typename T>
size_t vector_memory_usage(const std::vector<T>& vec) {
    size_t elements_memory = vec.capacity() * sizeof(T);
    size_t vector_overhead = sizeof(vec);
    return elements_memory + vector_overhead;
}

template<typename K, typename V>
size_t unordered_map_memory_usage(const std::unordered_map<K, V>& map) {
    size_t elements_memory = map.size() * (sizeof(K) + sizeof(V));
    size_t buckets_memory = map.bucket_count() * sizeof(void*);
    size_t map_overhead = sizeof(map);
    return elements_memory + buckets_memory + map_overhead;
}
