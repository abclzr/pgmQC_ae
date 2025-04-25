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
#include <algorithm>
#include <cassert>
#include <set>

// #include <cuda_runtime.h>
// #include <cutensornet.h>

#define DEBUG false



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

typedef float floatType;
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

struct Entry {
   uint64_t index, common_index, uncommon_index;
   floatType value;
   Entry(uint64_t _index, floatType _value) : index(_index), value(_value), common_index(0), uncommon_index(0) {}
   Entry(const Entry &other) : index(other.index), common_index(other.common_index),
                              uncommon_index(other.uncommon_index), value(other.value) {}
   ~Entry() {}

   bool operator<(const Entry& other) const {
      return std::abs(this->value) > std::abs(other.value); // Note: > for descending order
   }

   void get_transposed_index(const std::vector<uint64_t> &bitmask,
         const std::vector<short> &original_index_shift,
         const std::vector<short> &transposed_index_shift,
         const std::vector<bool> &to_common_index)
   {
      uint64_t tmp_index = this->index;

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
};

void get_bitmask_and_shift(const std::vector<int32_t> &original_modes,
         const std::vector<int32_t> &transposed_modes_common,
         const std::vector<int32_t> &transposed_modes_uncommon,
         const std::unordered_map<int32_t, short> &log2_extent,
         std::vector<uint64_t> &bitmask,
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

size_t contract_on_cpu(std::vector<Entry> &sparse_tensor1, std::vector<Entry> &sparse_tensor2,
                     const std::vector<int32_t> &modes1, const std::vector<int32_t> &modes2,
                     const std::unordered_map<int32_t, short> &log2_extent,
                     std::unordered_map<uint64_t, floatType> &result_map,
                     std::vector<int32_t> *result_modes=nullptr)
{
   uint64_t length1 = sparse_tensor1.size();
   uint64_t length2 = sparse_tensor2.size();
   uint64_t flops_cnt = 0;
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
   if (result_modes == nullptr)
   {
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

   std::vector<uint64_t> bitmask;
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

   std::unordered_map<uint64_t, std::vector<Entry>> hash_map1;
   std::unordered_map<uint64_t, std::vector<Entry>> hash_map2;
   uint64_t tmp1 = 0;
   uint64_t tmp2 = 0;
   if (DEBUG)
   {
      for (int i = 0; i < length1; ++i)
         std::cout << sparse_tensor1[i].index << " "<< sparse_tensor1[i].common_index << " "<< sparse_tensor1[i].uncommon_index << " "<<sparse_tensor1[i].value<<std::endl;

      for (int i = 0; i < length2; ++i)
         std::cout << sparse_tensor2[i].index << " "<< sparse_tensor2[i].common_index << " "<< sparse_tensor2[i].uncommon_index << " "<<sparse_tensor2[i].value<<std::endl;
   }

   while (tmp1 < length1 || tmp2 < length2)
   {
      if (tmp2 == length2 || sparse_tensor1[tmp1] < sparse_tensor2[tmp2])
      {
         uint64_t key_before_hash = sparse_tensor1[tmp1].common_index;
         uint64_t uncommon_index1 = sparse_tensor1[tmp1].uncommon_index;
         auto it2 = hash_map2.find(key_before_hash);
         if (it2 != hash_map2.end())
            for (const Entry &entry : it2->second)
            {
               uint64_t key = entry.uncommon_index | uncommon_index1;
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
            std::vector<Entry> item = {sparse_tensor1[tmp1]};
            hash_map1[key_before_hash] = item;
         }
         ++tmp1;
      }
      else
      {
         uint64_t key_before_hash = sparse_tensor2[tmp2].common_index;
         uint64_t uncommon_index2 = sparse_tensor2[tmp2].uncommon_index;
         auto it1 = hash_map1.find(key_before_hash);
         if (it1 != hash_map1.end())
            for (const Entry &entry : it1->second)
            {
               uint64_t key = entry.uncommon_index | uncommon_index2;
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
            std::vector<Entry> item = {sparse_tensor2[tmp2]};
            hash_map2[key_before_hash] = item;
         }
         ++tmp2;
      }
   }
   auto end2 = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double, std::milli> duration2 = end2 - start2;
   std::cout << "Double pointer chasing time cost: " << duration2.count() << " ms" << std::endl;


   if (true) printf("Number of flops = %ld\n", flops_cnt);

   return vector_memory_usage(common_modes) + vector_memory_usage(result_modes_)
         + unordered_map_memory_usage(hash_map1) + unordered_map_memory_usage(hash_map2);
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

int main(int argc, char* argv[])
{
   static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

   bool verbose = true;

   // Sphinx: #2
   /**********************
   * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
   **********************/
   // Check if the correct number of arguments are provided
   if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
      return 1;
   }
   std::string directory_path = argv[1];

   std::string descriptionfilename = directory_path + "/description.txt";
   std::string tensorfilename = directory_path + "/tensors.bin";
   // Output the file paths to verify correctness
   std::cout << "Description file: " << descriptionfilename << std::endl;
   std::cout << "Tensor file: " << tensorfilename << std::endl;

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
      modes_list.push_back(modes);
	}

   // Create a vector of extents for each tensor
   std::vector<std::vector<int64_t>> extent_list;
	for (auto modes : modes_list)
	{
		std::vector<int64_t> ext;
		for (auto mode : modes)
			ext.push_back(extent[mode]);
		extent_list.push_back(ext);
	}

	std::vector<int64_t> extentR = extent_list.back();

   if(verbose)
      printf("Defined tensor network, modes, and extents\n");

   // Sphinx: #3
   /**********************
   * Allocating data
   **********************/


   std::vector<size_t> elements_list;
   for (const auto& modes: modes_list)
   {
      size_t elements = 1;
      for (const auto& mode: modes)
         elements *= extent[mode];
      elements_list.push_back(elements);
   }
   size_t elementsR = elements_list.back();

   size_t size_total = 0;
   std::vector<size_t> size_list;
   for (const size_t& elements : elements_list)
   {
      size_list.push_back(sizeof(floatType) * elements);
      size_total += sizeof(floatType) * elements;
   }
   size_t sizeR = sizeof(floatType) * elements_list.back();


   std::vector<floatType *> cpu_data_list;
   for (const size_t& elements : elements_list)
   {
      floatType *cpu_data = (floatType*) malloc(sizeof(floatType) * elements);
      cpu_data_list.push_back(cpu_data);
   }

   for (const floatType * cpu_data: cpu_data_list)
      if (cpu_data == NULL)
      {
         printf("Error: Host memory allocation failed!\n");
         return -1;
      }

   /*******************
   * Initialize data
   *******************/
   // Open the binary file in binary read mode
   std::ifstream binfile(tensorfilename, std::ios::binary);
    
   if (!binfile.is_open())
	{
   	std::cerr << "Error: Failed to open file " << tensorfilename << std::endl;
      return 1;
   }

   std::vector<std::vector<Entry>> sparse_tensor_list;
   for (int i = 0; i < n - 1; ++i)
   {
      size_t elements = elements_list[i];
      floatType *cpu_data = cpu_data_list[i];
      for (uint64_t j = 0; j < elements; ++j)
			binfile.read(reinterpret_cast<char*>(&cpu_data[j]), sizeof(floatType));
      std::vector<Entry> sparse_tensor;
      for (uint64_t j = 0; j < elements; ++j)
         if (std::abs(cpu_data[j]) > 1e-7)
         {
            sparse_tensor.push_back(Entry(j, cpu_data[j]));
         }

      std::sort(sparse_tensor.begin(), sparse_tensor.end());
      sparse_tensor_list.push_back(sparse_tensor);
   }

   memset(cpu_data_list.back(), 0, sizeof(floatType) * elements_list.back());
   
   std::cout<<"=========Start contracting tensors in sparse from on cpu...========="<<std::endl;
   auto start = std::chrono::high_resolution_clock::now();

   std::unordered_map<uint64_t, floatType> result_map;
   size_t memory_usage = contract_on_cpu(sparse_tensor_list[0], sparse_tensor_list[1], modes_list[0], modes_list[1], log2_extent, result_map, &modes_list.back());
   // floatType *result_data = cpu_data_list.back();
   // std::cout<<"start"<<std::endl;

   // for (auto pair : result_map)
   // {
   //    // printf("result[%ld] = %.5lf\n", pair.first, pair.second);
   //    result_data[pair.first] = pair.second;
   // }
   auto end = std::chrono::high_resolution_clock::now();

   // Calculate the duration in milliseconds

   std::chrono::duration<double, std::milli> duration = end - start;
   // Print the duration
   std::cout << "Sparse tensor contraction time taken: " << duration.count() << " ms" << std::endl;

   memory_usage += vector_memory_usage(sparse_tensor_list[0]) + vector_memory_usage(sparse_tensor_list[1]) + unordered_map_memory_usage(result_map);
   std::cout<<"Memory usage on cpu: " << memory_usage << " bytes" << std::endl;
   std::cout<<"=========End contraction on cpu.===================================="<<std::endl;
   sparse_tensor_list.push_back({});
	for (const auto &pair : result_map)
	   if (std::abs(pair.second) > 1e-10)
	   {
	   	sparse_tensor_list.back().push_back(Entry(pair.first, pair.second));
	   }
   std::sort(sparse_tensor_list.back().begin(), sparse_tensor_list.back().end());
	for (auto entry : sparse_tensor_list.back())
		printf("a[%ld] = %.10f\n", entry.index, entry.value);

   // Free Host memory resources

   for (floatType *cpu_data: cpu_data_list)
      if (cpu_data)
         free(cpu_data);
   

   if(verbose)
      printf("Freed resources and exited\n");

   return 0;
}