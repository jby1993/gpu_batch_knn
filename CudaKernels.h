

#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

#include <cuda_runtime.h>
#include <cstdio>
// //refer to 'GPU-SME- k NN: Scalable and memory efficient k NN and lazy learning using GPUs'
// //I change the algorithm to compute batch data
// /* Computes a chunk of the distance matrix */
// __global__ void distanceKernel(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, float* matrix, int chunkTrain, int chunkTest, int selections);
// /* Performs the quicksort-based selection on a chunk of the distance matrix */
// __global__ void quickSelection(float* globalRows, int* testTrainSizes, int testSize, int trainChunk, int testChunk, long int* returnPossitions, float* returnValues, int selections, float* quickRows, long int* quickActualIndex, long int* quickNextIndex, float* quickRowsRight, long int* quickIndexRight);
// /* Performs the SQRT operation on the neighborhood distances and copy the index values to their final structures */
// __global__ void cuSqrtAndCopy(float* valuesSource, long int* indexSource, float* valuesDestinationAll, long int* indexDestinationAll, int testChunk, int testSize, int totalValues, int iterations);
// //training(batch*nVariables*trainSize),test(nVariables*testSize),testSize is all test points number sum
// //testTrainBatchIds(testSize),testTrainSizes(testSize)
// void batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs);
class BKNN{
public:
	~BKNN();
	static BKNN& Get(int trainSize, int testSize, int k);	
	void batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs);
private:
	BKNN(int trainSize, int testSize, int k);
	void realloc(int k);
	cudaEvent_t e;				//Event for kernel call synchronization
	cudaEvent_t eCopy;			//Event for memory copies synchronization
	cudaStream_t copyStream;	//Stream for copies
	cudaStream_t kernelStream;	//Stream for kernels
	float* chunkDistanceMatrix;
	float* quickRows;
	float* quickRowsRight;
	float* returnValues;
	long int* quickActualIndex;
	long int* quickNextIndex;
	long int* quickIndexRight;
	long int* returnPositions;
	dim3 blockSize;
	int trainChunks;
	int testChunks;
	int k_;
};
#endif /* CUDAKERNELS_H_ */
