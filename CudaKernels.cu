// #include "CudaKernels.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "float.h"


/* GPU-SME-kNN parameters */
// Test chunk size
const int TESTCHUNK = 8*1024;
// Number of threads of the distance kenrel
const int THREADSMULT = 256;
// Number of distances computed by each thread of the distance kernel
const int NREP = 8;

/* GPU-Komvarov-kNN parameters */
/*// Test chunk size
const int TESTCHUNK = 2*1024;
// Number of threads of the distance kenrel
const int THREADSMULT = 256;
// Number of distances computed by each thread of the distance kernel
const int NREP = 256;*/

/* Training chunk size */
 const int TRAININGCHUNK = THREADSMULT*NREP;

/* Maximum value of k*/
 const int MAXK = 1024;

/* Maxium number of clases for center kNN */
 const int MAXCLASSES = 10;
 /* Maxium number of variables for center kNN */
 const int MAXVARIABLES = 50;
 /* Size of centers array of center kNN */
 const int MAXSIZECENTERS = sizeof(float)*MAXVARIABLES*MAXCLASSES;


#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error!=cudaSuccess) \
    { \
    	std::cerr<<"Error: "<<cudaGetErrorString(error)<<std::endl; \
    	exit(1); \
    } \
  } while (0)

#define CUDA_POST_KERNEL_CHECK cudaDeviceSynchronize();CUDA_CHECK(cudaPeekAtLastError())


template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};
//training(batch*nVariables*trainSize),test(nVariables*testSize),testSize is all test points number sum
//testTrainBatchIds(testSize),testTrainSizes(testSize)
__global__ void distanceKernel(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, float* matrix, int chunkTrain, int chunkTest, int selections)
{
	int trainId = threadIdx.x;
	int testId = blockIdx.x;
	int preTrain=chunkTrain*TRAININGCHUNK;
	int preTest=chunkTest*TESTCHUNK;
	if(testId+preTest>=testSize)
		return;
	float trainVal, testVal;
	float tmpResult;

	/* Setting pointers to their  positions in memory*/
	float* m = &matrix[testId*(TRAININGCHUNK+selections)];

	//Copy the test values to shared memory
	float* shared = SharedMemory<float>();
	int* shared_id_size = (int*)&shared[nVariables];
	for(int i = threadIdx.x; i < nVariables; i+=blockDim.x)
		shared[i] = test[i*testSize+preTest+testId];	
	if(threadIdx.x==0)
	{
		shared_id_size[0] = testTrainBatchIds[preTest+testId];
		shared_id_size[1] = testTrainSizes[preTest+testId];
	}
	__syncthreads();

	//Compute NREP distances
	for(int r = 0; r < NREP;r++)
	{
		float result = 0;
		int actual_trainId=preTrain+trainId;
		if(actual_trainId>=shared_id_size[1])
			return;
		//Euclidean distance
		for(int i = 0; i < nVariables; i++){
			trainVal = training[shared_id_size[0]*nVariables*trainSize+i*trainSize+actual_trainId];
			testVal = shared[i];
			tmpResult = (trainVal - testVal);
			result += tmpResult * tmpResult;
		}		
		m[trainId] = result; //Coping the result to global memory
		trainId+=blockDim.x;
	}
}

/* Functions to swap arrays */
__device__ void swap(long int* &p1, long int* &p2)
{
	long int* tmp = p1;
	p1 = p2;
	p2 = tmp;
}
__device__ void swap(float* &p1, float* &p2)
{
	float* tmp = p1;
	p1 = p2;
	p2 = tmp;
}

/* Function to compute the median
 * Returns the position of the median
 */
__device__ long int median(long int a, long int b, long int c, float va, float vb, float vc)
{
	long int ret;
	if(va == vb)
		ret = va < vc ? a : c;
	else if(va == vc)
		ret = va < vb ? a : b;
	else if(vb == vc)
		ret = vb < va ? b : a;
	else if((va > vb && va < vc) ||
			(va > vc && va < vb))
		ret = a;
	else if((vb > va && vb < vc) ||
			(vb > vc && vb < va))
		ret = b;
	else if((vc > vb && vc < va) ||
			(vc > va && vc < vb))
		ret = c;
	return ret;
}


__global__ void quickSelection(float* globalRows, int* testTrainSizes, int testSize, int trainChunk, int testChunk, long int* returnPossitions, float* returnValues, int selections, float* quickRows, long int* quickActualIndex, long int* quickNextIndex, float* quickRowsRight, long int* quickIndexRight)
{
	int tid = threadIdx.x;
	int row = blockIdx.x*blockDim.y+threadIdx.y;
	int rowOut = testChunk*TESTCHUNK+row;

	int prevTrain = trainChunk*TRAININGCHUNK;

	if(rowOut >= testSize)
		return;

	/* Setting pointers to their  positions in memory*/
	float* rows = &globalRows[(row)*(TRAININGCHUNK+selections)];
	long int* index = &quickActualIndex[(row)*(TRAININGCHUNK+selections)];
	float* left = &quickRows[(row)*(TRAININGCHUNK+selections)];
	long int* leftIndex = &quickNextIndex[(row)*(TRAININGCHUNK+selections)];
	float* right = &quickRowsRight[(row)*(TRAININGCHUNK+selections)];
	long int* rightIndex = &quickIndexRight[(row)*(TRAININGCHUNK+selections)];

	/* Setting shared memory arrays for each block*/
	float* smValues = SharedMemory<float>();
	long int* smIndex = (long int*)&smValues[blockDim.x*blockDim.y];
	int* smTrainSizes = (int*)&smIndex[blockDim.x*blockDim.y];
	smIndex = &smIndex[blockDim.x*threadIdx.y];
	smValues = &smValues[blockDim.x*threadIdx.y];
	if(threadIdx.x==0)
		smTrainSizes[threadIdx.y]=testTrainSizes[rowOut];
	__syncthreads();
	int trainingSize=smTrainSizes[threadIdx.y];
	// int trainingSize=testTrainSizes[rowOut];

	/* On first iteration the values of the previous neighborhood are not properly set
	 * High values are inserted to avoid erroneous selections
	 */
	if(prevTrain == 0){
		for(int ind = tid; ind < selections; ind += blockDim.x){
			rows[ind+TRAININGCHUNK] = FLT_MAX;
		}
	}

	int k = selections;
	int leftPos;
	int rightPos;
	int size = trainingSize - prevTrain < TRAININGCHUNK ? trainingSize - prevTrain: TRAININGCHUNK;
	if(size<=0)
		return;
	/* Before start the algorithm is needed to set the values for the indexes to which each distance corresponds*/
	for(int i = tid; i < size; i+=blockDim.x){
		index[i] = prevTrain + i;
	}

	/* Moving current best for last iteration */
	if(trainingSize - prevTrain < TRAININGCHUNK){
		for(int ind = tid; ind < selections; ind += blockDim.x){
			rows[ind + size] = rows[ind + TRAININGCHUNK];
			index[ind + size] = index[ind + TRAININGCHUNK];
		}
	}

	/* Updating size value */
	size += selections;


	float pivot;
	int localPos;
	/* The first pivot possition is set to the farthest neighborhood of the previous iteration
	 * If this is the first iteration the median heuristic is used
	 */
	int pivotPos = prevTrain != 0 ? size-1 : median(0,size-1,size/2,rows[0], rows[size-1],rows[size/2]);

	

	unsigned int greater;
	unsigned int lower;
	unsigned int greaterBallot;
	unsigned int lowerBallot;

	// Mask is used to compute the local offset. It is needed to know how many
	// threads with a lane ID less than the actual thread and will write to the same buffer
	unsigned mask;
	asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));

	// main quickSelection loop
	while(k > 0){
		leftPos = 0; rightPos = 0;
		pivot = rows[pivotPos];

		//The limit value is computed to ensure all threads enter the for loop, avoid thread divergence
		int limit = size+(blockDim.x-(size%blockDim.x));
		for(int i = tid; i < limit; i+= blockDim.x){
			greater = i<size ? rows[i] > pivot : 0;
			lower =  i<size ? rows[i] <= pivot : 0;

			//__ballot and __popc functions are used to compute the local position within the warp
			//unsigned int __ballot(int predicate); If predicate is nonzero, __ballot returns a value with the Nth bit set, where N is the thread index.
			//and will gather all results of a warp threads to one results
			lowerBallot = __ballot_sync(0xFFFFFFFF,lower);
			greaterBallot = __ballot_sync(0xFFFFFFFF,greater);
			localPos = greater ? __popc(greaterBallot & mask) : __popc(lowerBallot & mask);
			// Local memory non coalescent writings
			if(i < size && lower){
				smValues[localPos] = rows[i];
				smIndex[localPos] = index[i];
			}
			else if(i < size && leftPos < k){
				smValues[__popc(lowerBallot)+localPos] = rows[i];
				smIndex[__popc(lowerBallot)+localPos] = index[i];
			}

			// Global memory coalestcent writings
			if(i < size && tid < __popc(lowerBallot)){
				left[leftPos+tid] = smValues[tid];
				leftIndex[leftPos+tid] = smIndex[tid];
			}
			else if(i < size && leftPos < k){
				right[rightPos+tid-__popc(lowerBallot)] = smValues[tid];
				rightIndex[rightPos+tid-__popc(lowerBallot)] = smIndex[tid];
			}
			leftPos += __popc(lowerBallot);
			rightPos += __popc(greaterBallot);

		}

		//Checking left part size
		if(leftPos > k){
			size = leftPos;

			if(rightPos == 0){ // This happens when all values are equal
				// Selection of the k required values
				for(int i = tid; i < k; i+=blockDim.x){
					returnPossitions[row*selections+(selections-k)+i] = leftIndex[i];
					returnValues[row*selections+(selections-k)+i] = left[i];
				}
				k = 0;
			}
			//Swapping arrays
			swap(rows,left);
			swap(index,leftIndex);
			// Updating pivot
			if(k>0)pivotPos = median(0,size-1,size/2,rows[0], rows[size-1],rows[size/2]);

		}
		else{// Left part smaller than k
			// Saving left part elements (pivot is included)
			for(int i = tid; i < leftPos; i+=blockDim.x){
				returnPossitions[row*selections+(selections-k)+i] = leftIndex[i];
				returnValues[row*selections+(selections-k)+i] = left[i];
			}
			size = rightPos;
			k-=leftPos;	// Updating k value
			//Swapping arrays
			swap(rows,right);
			swap(index,rightIndex);
			// Updating pivot
			if(k>0)pivotPos = median(0,size-1,size/2,rows[0], rows[size-1],rows[size/2]);
		}
	}

	//Writing neighborhood into the distance matrix for next step computation
	for(int ind = tid; ind < selections; ind += blockDim.x)
	{
		globalRows[row * (TRAININGCHUNK+selections) + ind+TRAININGCHUNK] = returnValues[row*selections + ind];
		quickActualIndex[row * (TRAININGCHUNK+selections) + ind+TRAININGCHUNK] = returnPossitions[row*selections + ind];
	}
}

// __global__ void cuSqrtAndCopy(float* valuesSource, long int* indexSource, float* valuesDestinationAll, long int* indexDestinationAll, int testChunk, int testSize, int totalValues, int iterations)
// {
// 	int tid = threadIdx.x;
// 	int bid = blockIdx.x;

// 	int index = tid;
// 	int preTest=testChunk*TESTCHUNK+blockDim.x*bid;
// 	// Setting pointers to their positions in memory
// 	float* vs = &valuesSource[blockDim.x * iterations * bid];
// 	long int* is = &indexSource[blockDim.x * iterations * bid];
// 	for(int i = 0; i < iterations; i++)
// 	{
// 		if(index < totalValues&&tid+preTest < testSize)
// 		{
// 			valuesDestinationAll[preTest*iterations+index]=sqrt(vs[index]);
// 			indexDestinationAll[preTest*iterations+index]=is[index];
// 		}
// 		index += blockDim.x;
// 	}
// }
__global__ void cuSqrtAndCopy(float* valuesSource, long int* indexSource, float* valuesDestinationAll, long int* indexDestinationAll, int testChunk, int testSize, int iterations)
{
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int index=bid*blockDim.x+tid;
    int dindex=index+testChunk*TESTCHUNK*iterations;
    if(dindex<testSize*iterations)
    {
        valuesDestinationAll[dindex]=sqrt(valuesSource[index]);
        indexDestinationAll[dindex]=indexSource[index];
    }
}
// __global__ void cuSet(float* target, float value, int size)
// {
// 	int ind=blockIdx.x*blockDim.x+threadIdx.x;
// 	for(int i=ind;i<size;i+=gridDim.x*blockDim.x)
// 		target[i]=value;
// }



#include <memory>
#include "CudaKernels.h"
static std::shared_ptr<BKNN> thread_instance_;
BKNN& BKNN::Get(int trainSize, int testSize, int k) {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new BKNN(trainSize,testSize,k));
  }
  else{
  	thread_instance_.get()->trainChunks=trainSize/TRAININGCHUNK+int(trainSize%TRAININGCHUNK!=0);
  	thread_instance_.get()->testChunks=testSize/TESTCHUNK+int(testSize%TESTCHUNK!=0);
  	if(thread_instance_.get()->k_!=k)
	{
		thread_instance_.get()->realloc(k);
		thread_instance_.get()->k_=k;
	}
  }
  return *(thread_instance_.get());
}

BKNN::BKNN(int trainSize, int testSize, int k){
	cudaEventCreate(&e);
	cudaEventCreate(&eCopy);
	cudaStreamCreate(&copyStream);
	cudaStreamCreate(&kernelStream);
	trainChunks=trainSize/TRAININGCHUNK+int(trainSize%TRAININGCHUNK!=0);
	testChunks=testSize/TESTCHUNK+int(testSize%TESTCHUNK!=0);
	cudaMalloc((void**)&chunkDistanceMatrix,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRows,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRowsRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickActualIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickNextIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickIndexRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&returnValues,sizeof(float)*k*TESTCHUNK);
	cudaMalloc((void**)&returnPositions,sizeof(long int)*k*TESTCHUNK);
	blockSize.x=32;	//must be 32, which is the thread number of a warp
	blockSize.y=4;
	k_=k;
}

BKNN::~BKNN(){
	cudaFree(chunkDistanceMatrix);
	cudaFree(quickRows);
	cudaFree(quickRowsRight);
	cudaFree(quickActualIndex);
	cudaFree(quickNextIndex);
	cudaFree(quickIndexRight);
	cudaFree(returnValues);
	cudaFree(returnPositions);

	cudaEventDestroy(e);
	cudaEventDestroy(eCopy);
	cudaStreamDestroy(copyStream);
	cudaStreamDestroy(kernelStream);
}
void BKNN::realloc(int k)
{
	cudaFree(chunkDistanceMatrix);
	cudaFree(quickRows);
	cudaFree(quickRowsRight);
	cudaFree(quickActualIndex);
	cudaFree(quickNextIndex);
	cudaFree(quickIndexRight);
	cudaFree(returnValues);
	cudaFree(returnPositions);

	cudaMalloc((void**)&chunkDistanceMatrix,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRows,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRowsRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickActualIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickNextIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickIndexRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&returnValues,sizeof(float)*k*TESTCHUNK);
	cudaMalloc((void**)&returnPositions,sizeof(long int)*k*TESTCHUNK);
}

void BKNN::batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs)
{
	for(int j=0; j<testChunks; j++)
	{
		for(int i=0; i<trainChunks; i++)
		{
			distanceKernel<<<TESTCHUNK,THREADSMULT,nVariables*sizeof(float)+2*sizeof(int),kernelStream>>>
				(training,test,testTrainBatchIds,testTrainSizes,nVariables,trainSize,testSize,chunkDistanceMatrix,i,j,k);			
			int nblocks=TESTCHUNK/blockSize.y+int(TESTCHUNK%blockSize.y!=0);
			int totalThreads=blockSize.x*blockSize.y;
			int shared=(sizeof(float)+sizeof(long int))*totalThreads+sizeof(int)*blockSize.y;
			cudaEventSynchronize(eCopy);
			quickSelection<<<nblocks,blockSize,shared,kernelStream>>>
				(chunkDistanceMatrix,testTrainSizes,testSize,i,j,returnPositions,returnValues,k,quickRows,quickActualIndex,quickNextIndex,quickRowsRight,quickIndexRight);
			cudaEventRecord(e,kernelStream);
		}
		int threads=256;
		int totalValues=TESTCHUNK*k;
		int blocks=totalValues/threads+int(totalValues%threads!=0);
		cudaEventSynchronize(e);
		cuSqrtAndCopy<<<blocks,threads,0,copyStream>>>(returnValues,returnPositions,testKnnDistances,testKnnIndexs,j,testSize,k);
		cudaEventRecord(eCopy,copyStream);
	}
}

void batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs)
{
	cudaEvent_t e;				//Event for kernel call synchronization
	cudaEvent_t eCopy;			//Event for memory copies synchronization
	cudaStream_t copyStream;	//Stream for copies
	cudaStream_t kernelStream;	//Stream for kernels
	cudaEventCreate(&e);
	cudaEventCreate(&eCopy);
	cudaStreamCreate(&copyStream);
	cudaStreamCreate(&kernelStream);



	int trainChunks=trainSize/TRAININGCHUNK+int(trainSize%TRAININGCHUNK!=0);
	int testChunks=testSize/TESTCHUNK+int(testSize%TESTCHUNK!=0);
	float* chunkDistanceMatrix;
	float* quickRows;
	float* quickRowsRight;
	cudaMalloc((void**)&chunkDistanceMatrix,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRows,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	cudaMalloc((void**)&quickRowsRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(float));
	long int* quickActualIndex;
	long int* quickNextIndex;
	long int* quickIndexRight;
	cudaMalloc((void**)&quickActualIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickNextIndex,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	cudaMalloc((void**)&quickIndexRight,(TRAININGCHUNK+k)*TESTCHUNK*sizeof(long int));
	float* returnValues;
	cudaMalloc((void**)&returnValues,sizeof(float)*k*TESTCHUNK);
	long int* returnPositions;
	cudaMalloc((void**)&returnPositions,sizeof(long int)*k*TESTCHUNK);
	dim3 blockSize;
	blockSize.x=32;	//must be 32, which is the thread number of a warp
	blockSize.y=4;
	for(int j=0; j<testChunks; j++)
	{
		for(int i=0; i<trainChunks; i++)
		{
			distanceKernel<<<TESTCHUNK,THREADSMULT,nVariables*sizeof(float)+2*sizeof(int),kernelStream>>>
				(training,test,testTrainBatchIds,testTrainSizes,nVariables,trainSize,testSize,chunkDistanceMatrix,i,j,k);
			// CUDA_POST_KERNEL_CHECK;
			int nblocks=TESTCHUNK/blockSize.y+int(TESTCHUNK%blockSize.y!=0);
			int totalThreads=blockSize.x*blockSize.y;
			int shared=(sizeof(float)+sizeof(long int))*totalThreads+sizeof(int)*blockSize.y;
			cudaEventSynchronize(eCopy);
			quickSelection<<<nblocks,blockSize,shared,kernelStream>>>
				(chunkDistanceMatrix,testTrainSizes,testSize,i,j,returnPositions,returnValues,k,quickRows,quickActualIndex,quickNextIndex,quickRowsRight,quickIndexRight);
			// CUDA_POST_KERNEL_CHECK;
			cudaEventRecord(e,kernelStream);
		}
		int threads=256;
		int totalValues=TESTCHUNK*k;
		int blocks=totalValues/threads+int(totalValues%threads!=0);
		cudaEventSynchronize(e);
		cuSqrtAndCopy<<<blocks,threads,0,copyStream>>>(returnValues,returnPositions,testKnnDistances,testKnnIndexs,j,testSize,k);
		// CUDA_POST_KERNEL_CHECK;
		cudaEventRecord(eCopy,copyStream);
	}
	cudaFree(chunkDistanceMatrix);
	cudaFree(quickRows);
	cudaFree(quickRowsRight);
	cudaFree(quickActualIndex);
	cudaFree(quickNextIndex);
	cudaFree(quickIndexRight);
	cudaFree(returnValues);
	cudaFree(returnPositions);

	cudaEventDestroy(e);
	cudaEventDestroy(eCopy);
	cudaStreamDestroy(copyStream);
	cudaStreamDestroy(kernelStream);
}
// int main(int argc, char* argv[] )
// {
// 	return 0;
// }