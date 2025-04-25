
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <stdexcept>

#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <set>
#include "sptensor.h"
#include "sptensor_utils.h"

typedef float floatType;
typedef uint32_t indexType;

bool DEBUG = false;

short log2(int64_t number)
{
   	short count = 0;
   	number = number - 1;
   	while (number > 0)
   	{
   	   	count++;
   	   	number = number >> 1;
   	}
   	return count;
}


Entry::Entry(indexType _index, floatType _value) : index(_index), value(_value), common_index(0), uncommon_index(0) {}

Entry::Entry(const Entry &other) : index(other.index), common_index(other.common_index),
                              uncommon_index(other.uncommon_index), value(other.value) {}
Entry::~Entry() {}

bool Entry::operator<(const Entry& other) const {
    return std::abs(this->value) > std::abs(other.value); // Note: > for descending order
}

void Entry::get_transposed_index(const std::vector<indexType> &bitmask,
        const std::vector<short> &original_index_shift,
        const std::vector<short> &transposed_index_shift,
        const std::vector<bool> &to_common_index)
{
    indexType tmp_index = this->index;

    this->common_index = 0;
    this->uncommon_index = 0;
    int length = bitmask.size();
    for (int i = 0; i < length; ++i)
    {
        uint8_t splited_index_original = tmp_index & bitmask[i];
        if (to_common_index[i])
           this->common_index |= splited_index_original << transposed_index_shift[i];
        else
           this->uncommon_index |= splited_index_original << transposed_index_shift[i];
        tmp_index >>= original_index_shift[i];
    }
    return;
}


void get_bitmask_and_shift(const std::vector<int32_t> &original_modes,
        const std::vector<int32_t> &transposed_modes_common,
        const std::vector<int32_t> &transposed_modes_uncommon,
        const std::unordered_map<int32_t, short> &log2_extent,
        std::vector<indexType> &bitmask,
        std::vector<short> &original_index_shift,
        std::vector<short> &transposed_index_shift,
        std::vector<bool> &to_common_index)
{
   	bitmask.clear();
   	original_index_shift.clear();
   	transposed_index_shift.clear();
   	to_common_index.clear();
   	int transposed_modes_common_length = transposed_modes_common.size();
   	int transposed_modes_uncommon_length = transposed_modes_uncommon.size();

   	for (int32_t original_mode : original_modes)
   	{
   	   	bitmask.push_back((1 << log2_extent.at(original_mode)) - 1);
   	   	original_index_shift.push_back(log2_extent.at(original_mode));
   	   	short bitwidth = 0;
   	   	for (int i = 0; i < transposed_modes_common_length; ++i)
   	   	{
   	   	   	if (transposed_modes_common[i] == original_mode)
   	   	   	{
   	   	   	   	transposed_index_shift.push_back(bitwidth);
   	   	   	   	to_common_index.push_back(true);
   	   	   	   	break;
   	   	   	}
   	   	   	bitwidth += log2_extent.at(transposed_modes_common[i]);
   	   	}
   	   	bitwidth = 0;
   	   	for (int i = 0; i < transposed_modes_uncommon_length; ++i)
   	   	{
   	   	   	if (transposed_modes_uncommon[i] == original_mode)
   	   	   	{
   	   	   	   	transposed_index_shift.push_back(bitwidth);
   	   	   	   	to_common_index.push_back(false);
   	   	   	   	break;
   	   	   	}
   	   	   	bitwidth += log2_extent.at(transposed_modes_uncommon[i]);
   	   	}
   	}
   	assert (to_common_index.size() == original_modes.size());
}

size_t contract_on_cpu_double_pointers(std::vector<Entry> &sparse_tensor1, std::vector<Entry> &sparse_tensor2,
                    const std::vector<int32_t> &modes1, const std::vector<int32_t> &modes2,
                    const std::unordered_map<int32_t, short> &log2_extent,
                    std::unordered_map<indexType, floatType> &result_map,
                    std::vector<int32_t> *result_modes,
					int *returned_flops_cnt)
{
   	indexType length1 = sparse_tensor1.size();
   	indexType length2 = sparse_tensor2.size();
   	indexType flops_cnt = 0;
   	std::vector<int32_t> common_modes;
   	std::vector<int32_t> result_modes_;
   	std::set<int32_t> common_modeset;
   	for (const int32_t mode1 : modes1)
   	   	for (const int32_t mode2 : modes2)
   	      	if (mode1 == mode2)
   	      	{
				common_modes.push_back(mode1);
				common_modeset.insert(mode1);
				break;
   	      	}
   	if (result_modes == nullptr || result_modes->size() == 0)
   	{
		if (result_modes == nullptr)
			result_modes = &result_modes_;
		for (int32_t mode1 : modes1)
			if (common_modeset.find(mode1) == common_modeset.end())
   	        	result_modes->push_back(mode1);
		for (int32_t mode2 : modes2)
			if (common_modeset.find(mode2) == common_modeset.end())
				result_modes->push_back(mode2);
   	}
   	else
   	{
		for (const int32_t result_mode : *result_modes)
			assert (common_modeset.find(result_mode) == common_modeset.end());
   	}

   	std::cout<<"Transposing new index..."<<std::endl;
   	// calculate the new index.
   	// The common index are to be contracted.
   	// The uncommon index are to be reserved.
   	auto start = std::chrono::high_resolution_clock::now();

   	std::vector<indexType> bitmask;
   	std::vector<short> original_index_shift;
   	std::vector<short> transposed_index_shift;
   	std::vector<bool> to_common_index;
	
   	get_bitmask_and_shift(modes1, common_modes, *result_modes, log2_extent,
   	                     bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	for (Entry &entry : sparse_tensor1)
   	{
   	   entry.get_transposed_index(bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	}

   	get_bitmask_and_shift(modes2, common_modes, *result_modes, log2_extent,
   	                     bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	for (Entry &entry : sparse_tensor2)
   	{
   		entry.get_transposed_index(bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	}
	
   	auto end = std::chrono::high_resolution_clock::now();
	
   	std::chrono::duration<double, std::milli> duration = end - start;
   	std::cout << "Transposing index time cost: " << duration.count() << " ms" << std::endl;

   	std::cout<<"Create double pointers..."<<std::endl;
   	auto start2 = std::chrono::high_resolution_clock::now();

   	std::unordered_map<indexType, std::vector<Entry>> hash_map1;
   	std::unordered_map<indexType, std::vector<Entry>> hash_map2;
   	indexType tmp1 = 0;
   	indexType tmp2 = 0;
   	if (DEBUG)
   	{
   		for (int i = 0; i < length1; ++i)
   			std::cout << sparse_tensor1[i].index << " "<< sparse_tensor1[i].common_index << " "<< sparse_tensor1[i].uncommon_index << " "<<sparse_tensor1[i].value<<std::endl;	
   		for (int i = 0; i < length2; ++i)
   			std::cout << sparse_tensor2[i].index << " "<< sparse_tensor2[i].common_index << " "<< sparse_tensor2[i].uncommon_index << " "<<sparse_tensor2[i].value<<std::endl;
   	}

   	while (tmp1 < length1 || tmp2 < length2)
   	{
   		if (tmp2 == length2 || (tmp1 < length1 && sparse_tensor1[tmp1] < sparse_tensor2[tmp2]))
   		{
   			indexType key_before_hash = sparse_tensor1[tmp1].common_index;
   			indexType uncommon_index1 = sparse_tensor1[tmp1].uncommon_index;
   			auto it2 = hash_map2.find(key_before_hash);
   			if (it2 != hash_map2.end())
   				for (const Entry &entry : it2->second)
   				{
					indexType key = entry.uncommon_index | uncommon_index1;
   					if (DEBUG) std::cout<< "add " << sparse_tensor1[tmp1].value * entry.value << " to " << key << std::endl;
   					if (result_map.find(key) != result_map.end())
   					{
   						floatType delta = sparse_tensor1[tmp1].value * entry.value;
   						result_map[key] += delta;
   					}
   					else
   						result_map[key] = sparse_tensor1[tmp1].value * entry.value;
   					flops_cnt++;
   				}
   			auto it1 = hash_map1.find(key_before_hash);
   			if (it1 != hash_map1.end())
   				it1->second.push_back(sparse_tensor1[tmp1]);
   			else
   			{
   				// std::vector<Entry> item = {sparse_tensor1[tmp1]};
   				hash_map1[key_before_hash] = {sparse_tensor1[tmp1]};
   			}
   			++tmp1;
   		}
   		else
   		{
   			indexType key_before_hash = sparse_tensor2[tmp2].common_index;
   			indexType uncommon_index2 = sparse_tensor2[tmp2].uncommon_index;
   			auto it1 = hash_map1.find(key_before_hash);
   			if (it1 != hash_map1.end())
   				for (const Entry &entry : it1->second)
   				{
   					indexType key = entry.uncommon_index | uncommon_index2;
   					if (DEBUG) std::cout<< "add " << sparse_tensor2[tmp2].value * entry.value << " to " << key << std::endl;
   					if (result_map.find(key) != result_map.end())
   						result_map[key] += sparse_tensor2[tmp2].value * entry.value;
   					else
   						result_map[key] = sparse_tensor2[tmp2].value * entry.value;
   					flops_cnt++;
   				}
   			auto it2 = hash_map2.find(key_before_hash);
   			if (it2 != hash_map2.end())
   				it2->second.push_back(sparse_tensor2[tmp2]);
   			else
   			{
   				// std::vector<Entry> item = {sparse_tensor2[tmp2]};
   				hash_map2[key_before_hash] = {sparse_tensor2[tmp2]};
   			}
   			++tmp2;
   		}
   	}
   	auto end2 = std::chrono::high_resolution_clock::now();
   	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
   	std::cout << "Double pointer chasing time cost: " << duration2.count() << " ms" << std::endl;


	if (true) printf("Number of flops = %ld\n", flops_cnt);
	*returned_flops_cnt = flops_cnt;


	return vector_memory_usage(common_modes) + vector_memory_usage(result_modes_)
    	+ unordered_map_memory_usage(hash_map1) + unordered_map_memory_usage(hash_map2);
}

size_t contract_on_cpu(std::vector<Entry> &sparse_tensor1, std::vector<Entry> &sparse_tensor2,
                    const std::vector<int32_t> &modes1, const std::vector<int32_t> &modes2,
                    const std::unordered_map<int32_t, short> &log2_extent,
                    std::unordered_map<indexType, floatType> &result_map,
                    std::vector<int32_t> *result_modes,
					int *returned_flops_cnt)
{
	if (sparse_tensor1.size() > sparse_tensor2.size())
	{
		return contract_on_cpu(sparse_tensor2, sparse_tensor1, modes2, modes1, log2_extent, result_map, result_modes, returned_flops_cnt);
	}
   	indexType length1 = sparse_tensor1.size();
   	indexType length2 = sparse_tensor2.size();
   	indexType flops_cnt = 0;
   	std::vector<int32_t> common_modes;
   	std::vector<int32_t> result_modes_;
   	std::set<int32_t> common_modeset;
   	for (const int32_t mode1 : modes1)
   	   	for (const int32_t mode2 : modes2)
   	      	if (mode1 == mode2)
   	      	{
				common_modes.push_back(mode1);
				common_modeset.insert(mode1);
				break;
   	      	}
   	if (result_modes == nullptr || result_modes->size() == 0)
   	{
		if (result_modes == nullptr)
			result_modes = &result_modes_;
		for (int32_t mode1 : modes1)
			if (common_modeset.find(mode1) == common_modeset.end())
   	        	result_modes->push_back(mode1);
		for (int32_t mode2 : modes2)
			if (common_modeset.find(mode2) == common_modeset.end())
				result_modes->push_back(mode2);
   	}
   	else
   	{
		for (const int32_t result_mode : *result_modes)
			assert (common_modeset.find(result_mode) == common_modeset.end());
   	}

   	std::cout<<"Transposing new index..."<<std::endl;
   	// calculate the new index.
   	// The common index are to be contracted.
   	// The uncommon index are to be reserved.
   	auto start = std::chrono::high_resolution_clock::now();

   	std::vector<indexType> bitmask;
   	std::vector<short> original_index_shift;
   	std::vector<short> transposed_index_shift;
   	std::vector<bool> to_common_index;
	
   	get_bitmask_and_shift(modes1, common_modes, *result_modes, log2_extent,
   	                     bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	for (Entry &entry : sparse_tensor1)
   	{
   	   entry.get_transposed_index(bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	}

   	get_bitmask_and_shift(modes2, common_modes, *result_modes, log2_extent,
   	                     bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	for (Entry &entry : sparse_tensor2)
   	{
   		entry.get_transposed_index(bitmask, original_index_shift, transposed_index_shift, to_common_index);
   	}
	
   	auto end = std::chrono::high_resolution_clock::now();
	
   	std::chrono::duration<double, std::milli> duration = end - start;
   	std::cout << "Transposing index time cost: " << duration.count() << " ms" << std::endl;

   	std::cout<<"Create double pointers..."<<std::endl;
   	auto start2 = std::chrono::high_resolution_clock::now();

   	std::unordered_map<indexType, std::vector<Entry>> hash_map1;
   	indexType tmp1 = 0;
   	indexType tmp2 = 0;
   	if (DEBUG)
   	{
   		for (int i = 0; i < length1; ++i)
   			std::cout << sparse_tensor1[i].index << " "<< sparse_tensor1[i].common_index << " "<< sparse_tensor1[i].uncommon_index << " "<<sparse_tensor1[i].value<<std::endl;	
   		for (int i = 0; i < length2; ++i)
   			std::cout << sparse_tensor2[i].index << " "<< sparse_tensor2[i].common_index << " "<< sparse_tensor2[i].uncommon_index << " "<<sparse_tensor2[i].value<<std::endl;
   	}

   	while (tmp1 < length1)
   	{
   		indexType key_before_hash = sparse_tensor1[tmp1].common_index;
   		indexType uncommon_index1 = sparse_tensor1[tmp1].uncommon_index;
   		auto it1 = hash_map1.find(key_before_hash);
   		if (it1 != hash_map1.end())
   			it1->second.push_back(sparse_tensor1[tmp1]);
   		else
   		{
   			// std::vector<Entry> item = {sparse_tensor1[tmp1]};
   			hash_map1[key_before_hash] = {sparse_tensor1[tmp1]};
   		}
   		++tmp1;
	}
	while (tmp2 < length2)
   	{
   		indexType key_before_hash = sparse_tensor2[tmp2].common_index;
   		indexType uncommon_index2 = sparse_tensor2[tmp2].uncommon_index;
   		auto it1 = hash_map1.find(key_before_hash);
   		if (it1 != hash_map1.end())
   			for (const Entry &entry : it1->second)
   			{
   				indexType key = entry.uncommon_index | uncommon_index2;
   				if (DEBUG) std::cout<< "add " << sparse_tensor2[tmp2].value * entry.value << " to " << key << std::endl;
   				if (result_map.find(key) != result_map.end())
   					result_map[key] += sparse_tensor2[tmp2].value * entry.value;
   				else
   					result_map[key] = sparse_tensor2[tmp2].value * entry.value;
   				flops_cnt++;
   			}
   		++tmp2;
   	}
   	auto end2 = std::chrono::high_resolution_clock::now();
   	std::chrono::duration<double, std::milli> duration2 = end2 - start2;
   	std::cout << "Double pointer chasing time cost: " << duration2.count() << " ms" << std::endl;


	if (true) printf("Number of flops = %ld\n", flops_cnt);
	*returned_flops_cnt = flops_cnt;

	return unordered_map_memory_usage(hash_map1) + unordered_map_memory_usage(result_map);
}

void check_results(const float *data1, const float *data2, size_t elements)
{
   	// ANSI escape codes for colors
   	const std::string green = "\033[32m";
   	const std::string red = "\033[31m";
   	const std::string reset = "\033[0m";
   	const std::string bold = "\033[1m";


	std::cout << "Checking the contraction result..." << std::endl;
	for (int i = 0; i < elements; ++i)
	{
		if (abs(data1[i] - data2[i]) > 1e-5)
		{
			std::cerr<<bold<<red << "[Wrong]" << reset << " The contraction result is not correct!" << std::endl;
        	for (int j = 0; j < elements; ++j)
            	std::cerr<< data1[j] << " ";
         	std::cerr<<std::endl;
         	for (int j = 0; j < elements; ++j)
            	std::cerr<< data2[j] << " ";
         	std::cerr<<std::endl;
         	std::cerr << "In the " << i << "th value: " << data1[i] << " not equal to " << data2[i] << " absolute error: " << data1[i] - data2[i] << std::endl;
			return;
		}
	}
	std::cout<<bold<<green << "[Pass]" << reset << " The contraction result is correct." << std::endl;
   // for (int j = 0; j < elements; ++j)
   //    std::cerr<< data1[j] << " ";
   // std::cerr<<std::endl;
   return;
}


std::vector<std::vector<Entry>> read_tensor_from_file(std::string tensorfilename, const std::vector<size_t> &elements_list)
{
	// Open the binary file in binary read mode
	std::ifstream binfile(tensorfilename, std::ios::binary);
    std::vector<std::vector<Entry>> sparse_tensor_list;

	if (!binfile.is_open())
	{
   		std::cerr << "Error: Failed to open file " << tensorfilename << std::endl;
      	return sparse_tensor_list;
   	}

	int num_inputs = elements_list.size();
   	for (int i = 0; i < num_inputs; ++i)
   	{
   	   	size_t elements = elements_list[i];
   	   	floatType data;
   	   	std::vector<Entry> sparse_tensor;
   	   	for (indexType j = 0; j < elements; ++j)
   	   	{
			binfile.read(reinterpret_cast<char*>(&data), sizeof(floatType));
   	   	   	if (std::abs(data) > 1e-15)
   	   	   	{
   	   	   	   	sparse_tensor.push_back(Entry(j, data));
   	   	   	}
   	   	}
   	   	std::sort(sparse_tensor.begin(), sparse_tensor.end());
   	   	sparse_tensor_list.push_back(sparse_tensor);
   	}
   	return sparse_tensor_list;
}

std::vector<std::tuple<int, int, int>> read_contraction_path_from_file(std::string contractionPathfilename, size_t contraction_path_length)
{
	std::ifstream file(contractionPathfilename); // Open the file
    if (!file.is_open()) {
        std::cerr << "Failed to open the file\n";
        return {};
    }
    std::vector<std::tuple<int, int, int>> vec; // Vector to store the tuples
    std::string line;
    // Read each line from the file
    for (int i = 0; i < contraction_path_length; ++i) {
		std::getline(file, line);
        std::istringstream iss(line); // Create a string stream from the line
        int a, b, c;
        
        if (!(iss >> a >> b >> c)) {
            std::cerr << "Failed to parse line: " << line << "\n";
            continue; // Skip lines that don't have exactly 3 integers
        }

        vec.emplace_back(a, b, c); // Add the tuple to the vector
    }
	return vec;
}

void serialize_sparse_tensor_to_file(std::string filename, const std::vector<Entry> &sparse_tensor)
{
    // Open a file in binary write mode
    std::ofstream ofs(filename, std::ios::binary | std::ios::app);

    // Check if the file is opened successfully
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing\n";
        return;
    }
    // Write the length of the vector
    size_t length = sparse_tensor.size();
    ofs.write(reinterpret_cast<const char*>(&length), sizeof(length));

    // Write the vector data
	for (const Entry &entry : sparse_tensor)
	{
    	ofs.write(reinterpret_cast<const char*>(&entry.index), sizeof(entry.index));
    	ofs.write(reinterpret_cast<const char*>(&entry.value), sizeof(entry.value));
	}

    // Close the file
    ofs.close();
	std::cout << "Successfullly written the sparse tensor to " << filename << std::endl;
}


void print_sparse_tensor_in_plain_text_to_file(std::string filename, const std::vector<Entry> &sparse_tensor, std::string from_which_subgraph)
{
    // Open a file in binary write mode
    std::ofstream ofs(filename, std::ios::app);

    // Check if the file is opened successfully
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for writing\n";
        return;
    }

	// Write the name of the subgraph
	ofs << "The following sparse tensor is from " << from_which_subgraph << std::endl;

    // Write the length of the vector
    size_t length = sparse_tensor.size();
    ofs << "There're " << length << " non-zero models" << std::endl;

    // Write the vector data
	for (const Entry &entry : sparse_tensor)
	{
    	ofs << "(" << entry.index << ") = " << entry.value << std::endl;
	}

    // Close the file
    ofs.close();
	std::cout << "Successfullly printing the sparse tensor in plain text mode to " << filename << std::endl;
}


std::vector<std::vector<Entry>> deserialize_sparse_tensor_list_from_file(std::string filename, size_t n) {
    // Open a file in binary read mode
    std::ifstream ifs(filename, std::ios::binary);

    // Check if the file is opened successfully
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file " << filename <<  " for reading\n";
        return {};
    }

    std::vector<std::vector<Entry>> sparse_tensor_list;
	size_t length;
    indexType index;
	floatType value;

	for (int i = 0; i < n - 1; ++i)
	{
		std::vector<Entry> sparse_tensor;
		ifs.read(reinterpret_cast<char*>(&length), sizeof(length));
		for (int j = 0; j < length; ++j)
		{
			ifs.read(reinterpret_cast<char*>(&index), sizeof(index));
			ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
			sparse_tensor.push_back(Entry(index, value));
		}
		sparse_tensor_list.push_back(sparse_tensor);
	}

    // Close the file
    ifs.close();

    return sparse_tensor_list;
}

// Function to clear the binary file
void clear_binary_file(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for clearing\n";
    }
    ofs.close();
}
