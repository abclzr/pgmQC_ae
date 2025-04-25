#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>

#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <set>

typedef float floatType;
typedef uint32_t indexType;
short log2(int64_t number);


struct Entry {
	indexType index, common_index, uncommon_index;
	floatType value;
	Entry(indexType _index, floatType _value);
	Entry(const Entry &other);
	~Entry();

	bool operator<(const Entry& other) const;

	void get_transposed_index(const std::vector<indexType> &bitmask,
		const std::vector<short> &original_index_shift,
		const std::vector<short> &transposed_index_shift,
		const std::vector<bool> &to_common_index);

};



void get_bitmask_and_shift(const std::vector<int32_t> &original_modes,
						const std::vector<int32_t> &transposed_modes_common,
						const std::vector<int32_t> &transposed_modes_uncommon,
						const std::unordered_map<int32_t, short> &log2_extent,
						std::vector<indexType> &bitmask,
						std::vector<short> &original_index_shift,
						std::vector<short> &transposed_index_shift,
						std::vector<bool> &to_common_index);


size_t contract_on_cpu(std::vector<Entry> &sparse_tensor1, std::vector<Entry> &sparse_tensor2,
					const std::vector<int32_t> &modes1, const std::vector<int32_t> &modes2,
					const std::unordered_map<int32_t, short> &log2_extent,
					std::unordered_map<indexType, floatType> &result_map,
					std::vector<int32_t> *result_modes,
					int *returned_flops_cnt);

void check_results(const float *data1, const float *data2, size_t elements);

std::vector<std::vector<Entry>> read_tensor_from_file(std::string tensorfilename, const std::vector<size_t> &elements_list);

std::vector<std::tuple<int, int, int>> read_contraction_path_from_file(std::string contractionPathfilename, size_t contraction_path_length);

void serialize_sparse_tensor_to_file(std::string filename, const std::vector<Entry> &sparse_tensor);

void print_sparse_tensor_in_plain_text_to_file(std::string filename, const std::vector<Entry> &sparse_tensor, std::string from_which_subgraph);

std::vector<std::vector<Entry>> deserialize_sparse_tensor_list_from_file(std::string filename, size_t n);

// Function to clear the binary file
void clear_binary_file(const std::string& filename);