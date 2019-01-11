#include "NNLayer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void NNLayerFeed(float* input, int nInputs, float* output, float* weights) {

	float val = 0;
	for (size_t i = 0; i < nInputs; i++)
	{
		val += input[i] * weights[blockDim.x * threadIdx.x + i];
	}

	val = tanhf(val);

	output[threadIdx.x] = val;
}

NNLayer::NNLayer(Layer * prevLayer, int nNeurons) : Layer(prevLayer)
{
	m_OW = nNeurons;
	m_OH = 1;
	m_numOutputs = 1;
	m_Stride = 1;

	//weights
	m_KernalSize = 1;
	m_numKernals = prevLayer->GetNumOfOutputElements() * nNeurons;
	m_nKernalElements = m_numKernals * m_KernalSize * m_KernalSize;

	//Calculate number of threads needed. (One thread per output neuron)
	m_nThreadsNeeded = m_nOutput…lements = m_OW * m_OH * m_numOutputs;
}

NNLayer::~NNLayer()
{
}

int NNLayer::FeedForward(float * input_ptr, float * output_ptr, float * kernel_ptr)
{
	dim3 nBlocks(m_numOutputs);
	dim3 nThreads(m_OW);

	NNLayerFeed << <nBlocks, nThreads >> > (input_ptr, m_prevLayer->GetNumOfOutputElements(), output_ptr, kernel_ptr);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		return -1;
	}

	return 1;
}

int NNLayer::Backpropagate()
{
	return 0;
}
