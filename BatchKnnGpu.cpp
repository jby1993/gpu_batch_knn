#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#include "CudaKernels.h"
void batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs);

//batch_pointcloud(batchsize,ndim,maxPnum), not all pointcloud has same vertex number, but not greater than maxPnum
//batch_query(ndim,all_query_size), splice all querys into one
//query_to_batchIds(all_query_size), indicate each query's batch id
//query_to_pc_sizes(all_query_size), indicate each query corresponding pc vertex number
//knn_dists(all_query_size,k), each query knearst's distances
//knn_indexs(all_query_size,k), each query knearst's indexs relate to corresponding to pointcloud
std::vector<at::Tensor> batch_knn_gpu_pytorch(at::Tensor batch_pointcloud,
                     int           maxPnum,
                     at::Tensor batch_query,
                     int           all_query_size,
                     at::Tensor query_to_batchIds,
                     at::Tensor query_to_pc_sizes,
                     int           ndim,
                     int           k
                     )
{
	CHECK_INPUT(batch_pointcloud);
	CHECK_INPUT(batch_query);
	CHECK_INPUT(query_to_batchIds);
	CHECK_INPUT(query_to_pc_sizes);
	AT_ASSERTM(batch_pointcloud.type().scalarType()==at::ScalarType::Float, "batch_pointcloud must be a float tensor");
	AT_ASSERTM(batch_query.type().scalarType()==at::ScalarType::Float, "batch_query must be a float tensor");
	AT_ASSERTM(query_to_batchIds.type().scalarType()==at::ScalarType::Int, "query_to_batchIds must be a int tensor");
	AT_ASSERTM(query_to_pc_sizes.type().scalarType()==at::ScalarType::Int, "query_to_pc_sizes must be a int tensor");
	cudaSetDevice(batch_query.get_device());
	auto options_float_nograd = torch::TensorOptions()
                                    .dtype(batch_pointcloud.dtype())
                                    .layout(batch_pointcloud.layout())
                                    .device(batch_pointcloud.device())
                                    .requires_grad(false);
    auto options_long_nograd = torch::TensorOptions()
                                    .dtype(torch::kInt64)
                                    .layout(query_to_batchIds.layout())
                                    .device(query_to_batchIds.device())
                                    .requires_grad(false);
    auto knn_dists=-torch::ones({all_query_size,k},options_float_nograd);
    auto knn_indexs=-torch::ones({all_query_size,k},options_long_nograd);	
	// //singleton mode implement, may have bugs for different threads use, and is a little faster than func implement
	// BKNN::Get(maxPnum,all_query_size,k).batchKnn(batch_pointcloud.data<float>(),batch_query.data<float>(),query_to_batchIds.data<int>(),query_to_pc_sizes.data<int>(),
	// 		ndim,maxPnum,all_query_size,k,knn_dists.data<float>(),knn_indexs.data<long int>());
	// func implement, a little slower, but can handle different threads easier
	batchKnn(batch_pointcloud.data<float>(),batch_query.data<float>(),query_to_batchIds.data<int>(),query_to_pc_sizes.data<int>(),
			ndim,maxPnum,all_query_size,k,knn_dists.data<float>(),knn_indexs.data<long int>());
	return {knn_dists,knn_indexs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_knn_gpu", &batch_knn_gpu_pytorch, "knn cuda global (CUDA)");
}