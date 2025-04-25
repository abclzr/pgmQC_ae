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

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>

typedef float floatType;
cudaDataType_t typeData = CUDA_R_32F;
cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;
bool verbose = true;

#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
  if( err != CUTENSORNET_STATUS_SUCCESS )                         \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};


struct GPUTimer
{
    GPUTimer(cudaStream_t stream): stream_(stream)
    {
        HANDLE_CUDA_ERROR(cudaEventCreate(&start_));
        HANDLE_CUDA_ERROR(cudaEventCreate(&stop_));
    }

    ~GPUTimer()
    {
        HANDLE_CUDA_ERROR(cudaEventDestroy(start_));
        HANDLE_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void start()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(start_, stream_));
    }

    float seconds()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(stop_, stream_));
        HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_));
        float time;
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start_, stop_));
        return time * 1e-3;
    }

    private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
};


int contract_on_gpu(std::string descriptionfilename, std::string tensorfilename, std::ofstream &outFile)
{
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

	for (int i = 0; i < num_vars; ++i)
	{
		int ex;
		file >> ex;
		extent[i + 100] = ex;
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


   int32_t numInputs = n - 1;

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

   if(verbose)
      printf("Total GPU memory used for tensor storage: %.2f GiB\n",
             (size_total) / 1024. /1024. / 1024);

   void** rawDataIn_d = (void **) malloc(sizeof(void*) * numInputs);
   void* R_d;

   HANDLE_CUDA_ERROR( cudaMalloc((void**) &R_d, size_list.back()));
   for (int i = 0; i < numInputs; ++i)
   {
      HANDLE_CUDA_ERROR( cudaMalloc((void**) &rawDataIn_d[i], size_list[i]) );
   }
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

   for (int i = 0; i < n - 1; ++i)
   {
      size_t elements = elements_list[i];
      floatType *cpu_data = cpu_data_list[i];
      for (uint64_t j = 0; j < elements; ++j)
			binfile.read(reinterpret_cast<char*>(&cpu_data[j]), sizeof(floatType));
   }
   floatType *R = (floatType*) malloc(sizeof(floatType) * elements_list.back());
   memset(R, 0, sizeof(floatType) * elements_list.back());

   for (int i = 0; i < n - 1; ++i)
   {
      size_t size = size_list[i];
      floatType *cpu_data = cpu_data_list[i];
      HANDLE_CUDA_ERROR( cudaMemcpy(rawDataIn_d[i], cpu_data, size, cudaMemcpyHostToDevice) );
   }
   if(verbose)
      printf("Allocated GPU memory for data, and initialize data\n");

   // Sphinx: #4
   /*************************
   * cuTensorNet
   *************************/

   cudaStream_t stream;
   HANDLE_CUDA_ERROR( cudaStreamCreate(&stream) );

   cutensornetHandle_t handle;
   HANDLE_ERROR( cutensornetCreate(&handle) );


   const int32_t nmodeR = modes_list.back().size();

   /*******************************
   * Create Network Descriptor
   *******************************/

   const int32_t** modesIn = (const int32_t**) malloc(sizeof(const int32_t*) * (n - 1));
   for (int i = 0; i < n - 1; ++i)
      modesIn[i] = modes_list[i].data();
   int32_t * const numModesIn = (int32_t *) malloc(sizeof(int32_t) * (n - 1));
   for (int i = 0; i < n - 1; ++i)
      numModesIn[i] = modes_list[i].size();
   const int64_t ** extentsIn = (const int64_t **) malloc(sizeof(const int64_t*) * (n - 1));
   for (int i = 0; i < n - 1; ++i)
      extentsIn[i] = extent_list[i].data();
   const int64_t** stridesIn = (const int64_t**) malloc(sizeof(const int64_t*) * (n - 1));
   for (int i = 0; i < n - 1; ++i)
      stridesIn[i] = NULL;

   // Set up tensor network
   cutensornetNetworkDescriptor_t descNet;
   HANDLE_ERROR( cutensornetCreateNetworkDescriptor(handle,
                     numInputs, numModesIn, extentsIn, stridesIn, modesIn, NULL,
                     nmodeR, extentR.data(), /*stridesOut = */NULL, modes_list.back().data(),
                     typeData, typeCompute,
                     &descNet) );

   if(verbose)
      printf("Initialized the cuTensorNet library and created a tensor network descriptor\n");

   // Sphinx: #5
   /*******************************
   * Choose workspace limit based on available resources.
   *******************************/

   size_t freeMem, totalMem;
   HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem) );
   uint64_t workspaceLimit = (uint64_t)((double)freeMem * 0.9);
   if(verbose)
      printf("Workspace limit = %lu\n", workspaceLimit);

   /*******************************
   * Find "optimal" contraction order and slicing
   *******************************/

   cutensornetContractionOptimizerConfig_t optimizerConfig;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig) );

   // Set the desired number of hyper-samples (defaults to 0)
   int32_t num_hypersamples = 8;
   HANDLE_ERROR( cutensornetContractionOptimizerConfigSetAttribute(handle,
                     optimizerConfig,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_SAMPLES,
                     &num_hypersamples,
                     sizeof(num_hypersamples)) );

   // Create contraction optimizer info and find an optimized contraction path
   cutensornetContractionOptimizerInfo_t optimizerInfo;
   HANDLE_ERROR( cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo) );

   HANDLE_ERROR( cutensornetContractionOptimize(handle,
                                             descNet,
                                             optimizerConfig,
                                             workspaceLimit,
                                             optimizerInfo) );

   // Query the number of slices the tensor network execution will be split into
   int64_t numSlices = 0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                  handle,
                  optimizerInfo,
                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
                  &numSlices,
                  sizeof(numSlices)) );
   assert(numSlices > 0);

   if(verbose)
      printf("Found an optimized contraction path using cuTensorNet optimizer\n");

   // Sphinx: #6
   /*******************************
   * Create workspace descriptor, allocate workspace, and set it.
   *******************************/

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR( cutensornetCreateWorkspaceDescriptor(handle, &workDesc) );

   int64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR( cutensornetWorkspaceComputeContractionSizes(handle,
                                                         descNet,
                                                         optimizerInfo,
                                                         workDesc) );

   HANDLE_ERROR( cutensornetWorkspaceGetMemorySize(handle,
                                                   workDesc,
                                                   CUTENSORNET_WORKSIZE_PREF_MIN,
                                                   CUTENSORNET_MEMSPACE_DEVICE,
                                                   CUTENSORNET_WORKSPACE_SCRATCH,
                                                   &requiredWorkspaceSize) );

   void* work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, requiredWorkspaceSize) );

   HANDLE_ERROR( cutensornetWorkspaceSetMemory(handle,
                                               workDesc,
                                               CUTENSORNET_MEMSPACE_DEVICE,
                                               CUTENSORNET_WORKSPACE_SCRATCH,
                                               work,
                                               requiredWorkspaceSize) );

   if(verbose)
      printf("Allocated and set up the GPU workspace\n");

   // Sphinx: #7
   /*******************************
   * Initialize the pairwise contraction plan (for cuTENSOR).
   *******************************/

   cutensornetContractionPlan_t plan;
   HANDLE_ERROR( cutensornetCreateContractionPlan(handle,
                                                descNet,
                                                optimizerInfo,
                                                workDesc,
                                                &plan) );

   /*******************************
   * Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
   *           for each pairwise tensor contraction.
   *******************************/
   cutensornetContractionAutotunePreference_t autotunePref;
   HANDLE_ERROR( cutensornetCreateContractionAutotunePreference(handle,
                                                      &autotunePref) );

   const int numAutotuningIterations = 5; // may be 0
   HANDLE_ERROR( cutensornetContractionAutotunePreferenceSetAttribute(
                           handle,
                           autotunePref,
                           CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                           &numAutotuningIterations,
                           sizeof(numAutotuningIterations)) );

   // Modify the plan again to find the best pair-wise contractions
   HANDLE_ERROR( cutensornetContractionAutotune(handle,
                                                plan,
                                                rawDataIn_d,
                                                R_d,
                                                workDesc,
                                                autotunePref,
                                                stream) );

   HANDLE_ERROR( cutensornetDestroyContractionAutotunePreference(autotunePref) );

   if(verbose)
      printf("Created a contraction plan for cuTensorNet and optionally auto-tuned it\n");

   // Sphinx: #8
   /**********************
   * Execute the tensor network contraction
   **********************/

   // Create a cutensornetSliceGroup_t object from a range of slice IDs
   cutensornetSliceGroup_t sliceGroup{};
   HANDLE_ERROR( cutensornetCreateSliceGroupFromIDRange(handle, 0, numSlices, 1, &sliceGroup) );

   GPUTimer timer {stream};
   double minTimeCUTENSORNET = 1e100;
   const int numRuns = 1; // number of repeats to get stable performance results
   for (int i = 0; i < numRuns; ++i)
   {
      HANDLE_CUDA_ERROR( cudaMemcpy(R_d, R, sizeR, cudaMemcpyHostToDevice) ); // restore the output tensor on GPU
      HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

      /*
      * Contract all slices of the tensor network
      */
      timer.start();

      int32_t accumulateOutput = 0; // output tensor data will be overwritten
      HANDLE_ERROR( cutensornetContractSlices(handle,
                     plan,
                     rawDataIn_d,
                     R_d,
                     accumulateOutput,
                     workDesc,
                     sliceGroup, // alternatively, NULL can also be used to contract over all slices instead of specifying a sliceGroup object
                     stream) );

      // Synchronize and measure best timing
      auto time = timer.seconds();
      minTimeCUTENSORNET = (time > minTimeCUTENSORNET) ? minTimeCUTENSORNET : time;
   }

   if(verbose)
      printf("Contracted the tensor network, each slice used the same contraction plan\n");

   // Print the 1-norm of the output tensor (verification)
   HANDLE_CUDA_ERROR( cudaStreamSynchronize(stream) );
   HANDLE_CUDA_ERROR( cudaMemcpy(R, R_d, sizeR, cudaMemcpyDeviceToHost) ); // restore the output tensor on Host
	
   for (int i = 0; i < elements_list.back(); ++i)
      outFile.write(reinterpret_cast<const char*>(&R[i]), sizeof(floatType));
	// check_results(R, cpu_data_list.back(), elements_list.back());
	// double norm1 = 0.0;
   // for (int64_t i = 0; i < elementsR; ++i) {
   //    norm1 += std::abs(R[i]);
   // }
   // if(verbose)
   //    printf("Computed the 1-norm of the output tensor: %e\n", norm1);

   /*************************/

   // Query the total Flop count for the tensor network contraction
   double flops {0.0};
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                     handle,
                     optimizerInfo,
                     CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                     &flops,
                     sizeof(flops)) );

   if(verbose) {
      printf("Number of tensor network slices = %ld\n", numSlices);
      printf("Tensor network contraction time (ms) = %.3f\n", minTimeCUTENSORNET * 1000.f);
   }

   // Free cuTensorNet resources
   HANDLE_ERROR( cutensornetDestroySliceGroup(sliceGroup) );
   HANDLE_ERROR( cutensornetDestroyContractionPlan(plan) );
   HANDLE_ERROR( cutensornetDestroyWorkspaceDescriptor(workDesc) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerInfo(optimizerInfo) );
   HANDLE_ERROR( cutensornetDestroyContractionOptimizerConfig(optimizerConfig) );
   HANDLE_ERROR( cutensornetDestroyNetworkDescriptor(descNet) );
   HANDLE_ERROR( cutensornetDestroy(handle) );

   // Free Host memory resources
   if (R) free(R);
   // if (D) free(D);
   // if (C) free(C);
   // if (B) free(B);
   // if (A) free(A);

   // Free GPU memory resources
   if (work) cudaFree(work);
   if (R_d) cudaFree(R_d);
   // if (rawDataIn_d[0]) cudaFree(rawDataIn_d[0]);
   // if (rawDataIn_d[1]) cudaFree(rawDataIn_d[1]);
   // if (rawDataIn_d[2]) cudaFree(rawDataIn_d[2]);
   // if (rawDataIn_d[3]) cudaFree(rawDataIn_d[3]);
   if (rawDataIn_d)
   {
      for (int i = 0; i < numInputs; ++i)
         if (rawDataIn_d[i])
            cudaFree(rawDataIn_d[i]);
      free(rawDataIn_d);
   }
   
   for (floatType *cpu_data: cpu_data_list)
      if (cpu_data)
         free(cpu_data);
   
   if (modesIn) free(modesIn);
   if (numModesIn) free(numModesIn);
   if (extentsIn) free(extentsIn);
   if (stridesIn) free(stridesIn);

   if(verbose)
      printf("Freed resources and exited\n");
   return 0;
}

int main(int argc, char* argv[])
{
   static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");


   // Check cuTensorNet version
   const size_t cuTensornetVersion = cutensornetGetVersion();
   if(verbose)
      printf("cuTensorNet version: %ld\n", cuTensornetVersion);

   // Set GPU device
   int numDevices {0};
   HANDLE_CUDA_ERROR( cudaGetDeviceCount(&numDevices) );
   const int deviceId = 0;
   HANDLE_CUDA_ERROR( cudaSetDevice(deviceId) );
   cudaDeviceProp prop;
   HANDLE_CUDA_ERROR( cudaGetDeviceProperties(&prop, deviceId) );

   if(verbose) {
      printf("===== device info ======\n");
      printf("GPU-name:%s\n", prop.name);
      printf("GPU-clock:%d\n", prop.clockRate);
      printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
      printf("GPU-nSM:%d\n", prop.multiProcessorCount);
      printf("GPU-major:%d\n", prop.major);
      printf("GPU-minor:%d\n", prop.minor);
      printf("========================\n");
   }


   if(verbose)
      printf("Included headers and defined data types\n");

   // Sphinx: #2
   /**********************
   * Computing: R_{k,l} = A_{a,b,c,d,e,f} B_{b,g,h,e,i,j} C_{m,a,g,f,i,k} D_{l,c,h,d,j,m}
   **********************/
   std::string outTensorBinFilename = std::string(argv[1]) + "/tensors.bin";
   std::ofstream outFile(outTensorBinFilename, std::ios::binary);
   if (!outFile) {
      std::cerr << "Could not open the file for writing.\n";
      return;
   }

   for (int i = 2; i < argc; ++i)
   {
      std::string directory_path = argv[i];

      std::string descriptionfilename = directory_path + "/description.txt";
      std::string tensorfilename = directory_path + "/tensors.bin";
      contract_on_gpu(descriptionfilename, tensorfilename, outFile);
   }

   outFile.close();
   return 0;
}