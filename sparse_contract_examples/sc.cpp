/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Sphinx: #1

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

// #include <cuda_runtime.h>
// #include <cutensornet.h>
#include "sptensor.h"
#include "sptensor_utils.h"

#define DEBUG false


typedef float floatType;
typedef uint32_t indexType;

int main(int argc, char* argv[])
{
	static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

	bool verbose = true;

    std::string read_mode;
    
    // Loop through each argument
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Check if the argument starts with "-read_mode="
        if (arg.substr(0, 11) == "-read_mode=") {
            // Extract the value after "="
            read_mode = arg.substr(11);
            break; // Exit loop once we find the argument
        }
    }

	assert (read_mode == "sparse" || read_mode == "non_sparse");

	std::string directory_path = argv[2];
	std::string output_directory_path = argv[3];

	std::string descriptionfilename = directory_path + "/description.txt";
	std::string tensorfilename;
	if (read_mode == "non_sparse")
		tensorfilename = directory_path + "/tensors.bin";
	else if (read_mode == "sparse")
		tensorfilename = directory_path + "/sp_tensors.bin";

	std::string contractionPathfilename = directory_path + "/contraction_path.txt";
	
	std::string outputtensorfilename;
	std::string outputtensor_plaintext_filename;
	std::string metric_filename;
	if (read_mode == "non_sparse")
	{
		outputtensorfilename = output_directory_path + "/sp_tensors.bin";
		outputtensor_plaintext_filename = output_directory_path + "/sp_tensors.txt";
		metric_filename = directory_path + "/sp_metric.txt";
	}
	else if (read_mode == "sparse")
	{
		outputtensorfilename = output_directory_path + "/sp_result.bin";
		clear_binary_file(outputtensorfilename);
		metric_filename = output_directory_path + "/sp_metric.txt";
	}
	// Output the file paths to verify correctness
	std::cout << "Description file: " << descriptionfilename << std::endl;
	std::cout << "Tensor file: " << tensorfilename << std::endl;

	// read description file
	std::ifstream file(descriptionfilename);
	if (!file.is_open())
	{
		std::cerr << "Failed to open the file: " << descriptionfilename << std::endl;
		return 1;
	}
	int num_vars;
	file >> num_vars;

	std::vector<std::string> vars;
	std::unordered_map<std::string, int32_t> var2mode;
	for (int i = 0; i < num_vars; ++i)
	{
		std::string var;
		file >> var;
		vars.push_back(var);
		var2mode[var] = i + 100;
	}

	// Set mode extents
	std::unordered_map<int32_t, int64_t> extent;
	std::unordered_map<int32_t, short> log2_extent;

	for (int i = 0; i < num_vars; ++i)
	{
		int64_t ex;
		file >> ex;
		extent[i + 100] = ex;
    	log2_extent[i + 100] = log2(ex);
	}

	int n;
	file >> n;
	std::string line;
	std::getline(file, line);

	// Create vectors of tensor modes
	std::vector<std::vector<int32_t>> modes_list;
	std::vector<int32_t> modesR;

	for (int i = 0; i < n; ++i)
	{
   		std::getline(file, line);
		std::istringstream iss(line);
		std::vector<int32_t> modes;
		std::string str;
		while (iss >> str)
		{
			modes.push_back(var2mode[str]);
    	}
    	std::reverse(modes.begin(), modes.end());
		if (i < n - 1)
    		modes_list.push_back(modes);
		else
			modesR = modes;
	}

	if(verbose)
    	printf("Defined tensor network, modes, and extents\n");

	std::vector<size_t> elements_list;
	for (const auto& modes: modes_list)
	{
		size_t elements = 1;
		for (const auto& mode: modes)
			elements *= extent[mode];
		elements_list.push_back(elements);
	}


	/*******************
	* Initialize data
	*******************/
	std::vector<std::vector<Entry>> sparse_tensor_list;
	if (read_mode == "non_sparse")
		sparse_tensor_list = read_tensor_from_file(tensorfilename, elements_list);
	else if (read_mode == "sparse")
		sparse_tensor_list = deserialize_sparse_tensor_list_from_file(tensorfilename, n);

	size_t num_inputs = n - 1;
	size_t contraction_path_length = num_inputs - 1;
	assert (contraction_path_length = sparse_tensor_list.size() - 1);
	std::vector<std::tuple<int, int, int>> path = read_contraction_path_from_file(contractionPathfilename, contraction_path_length);
	
	size_t memory_usage = 0;

	std::cout<<"=========Start contracting tensors in sparse from on cpu...========="<<std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	int flops_cnt = 0;
	int total_flops = 0;
	for (std::tuple<int, int, int> each_contraction : path)
	{
		int index1 = std::get<0>(each_contraction);
		int index2 = std::get<1>(each_contraction);
		assert(sparse_tensor_list.size() == std::get<2>(each_contraction));

		std::unordered_map<indexType, floatType> result_map;
		if (sparse_tensor_list.size() < num_inputs * 2 - 2)
			modes_list.push_back({});
		else
			// if it will be the final contracted tensor, use the given modes of order
			modes_list.push_back(modesR);

		memory_usage += contract_on_cpu(sparse_tensor_list[index1], sparse_tensor_list[index2],
										modes_list[index1], modes_list[index2],
										log2_extent, result_map, &modes_list.back(), &flops_cnt);
		total_flops += flops_cnt;

		sparse_tensor_list.push_back({});
		for (const auto &pair : result_map)
			if (std::abs(pair.second) > 1e-15)
			{
				sparse_tensor_list.back().push_back(Entry(pair.first, pair.second));
			}
		std::sort(sparse_tensor_list.back().begin(), sparse_tensor_list.back().end());
	}
	serialize_sparse_tensor_to_file(outputtensorfilename, sparse_tensor_list.back());
	// print_sparse_tensor_in_plain_text_to_file(outputtensor_plaintext_filename, sparse_tensor_list.back(), directory_path);

	auto end = std::chrono::high_resolution_clock::now();
	// if (read_mode == "sparse")
	// 	for (auto entry : sparse_tensor_list.back())
	// 		printf("a[%ld] = %.10f\n", entry.index, entry.value);

	// Calculate the duration in milliseconds

	std::chrono::duration<double, std::milli> duration = end - start;
	// Print the duration
	std::cout << "Sparse tensor contraction time taken: " << duration.count() << " ms" << std::endl;

	size_t memory_usage_hashtable = memory_usage;
	for (auto &sparse_tensor : sparse_tensor_list)
		memory_usage += vector_memory_usage(sparse_tensor);

	std::cout<<"Memory usage on cpu: " << memory_usage << " bytes" << std::endl;
	std::cout<<"=========End contraction on cpu.===================================="<<std::endl;

	// Write the metric results to csv file
	if (read_mode == "sparse")
	{
		std::ofstream metric_file(metric_filename);
		metric_file << "Memory usage on cpu: " << memory_usage << " bytes" << std::endl;
		metric_file << "Sparse tensor contraction time taken: " << duration.count() << " ms" << std::endl;
		metric_file << "Total flops: " << total_flops << std::endl;
		metric_file << "Memory usage hashtable: " << memory_usage_hashtable << " bytes" << std::endl;
		metric_file << "Memory usage vector: " << memory_usage - memory_usage_hashtable << " bytes" << std::endl;
		metric_file << "Sparsity of the tensor: " << sparse_tensor_list.back().size() << " / " << (1LL << 2 * modesR.size()) << std::endl;
		metric_file.close();
	}
	else
	{
		std::ofstream metric_file(metric_filename);
		metric_file << "Sparsity of the tensor: " << sparse_tensor_list.back().size() << " / " << (1LL << 2 * modesR.size()) << std::endl;
		metric_file.close();
	}
	// Free Host memory resources


	if(verbose)
    	printf("Freed resources and exited\n");

	return 0;
}