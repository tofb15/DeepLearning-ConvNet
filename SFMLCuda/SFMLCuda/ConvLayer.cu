#include "ConvLayer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void ConvLayerFeed(float* input, int I_W, int I_H, int nInputs, float* output, float* kernals, int K_WH) {

	int I_Size = I_W * I_H;
	int K_WH_2 = K_WH * K_WH;

	//int nKernals = nInputs * nOutputs;
	//int kernalSize = K_WH * K_WH * nKernals;

	int kernalOffset = blockIdx.x * K_WH_2 * nInputs;

	float val = 0;
	int currKernal = 0;
	for (size_t i = 0; i < nInputs; i++)
	{
		for (size_t ky = 0; ky < K_WH; ky++)
		{
			for (size_t kx = 0; kx < K_WH; kx++)
			{
				val += input[(threadIdx.x + kx) + (threadIdx.y + ky) * I_W + I_Size * i] * kernals[kernalOffset + currKernal + ky * K_WH + kx];
			}
		}
		currKernal += K_WH_2;
	}

	if (val > 0)
		output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = val;
	else
		output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = 0;
}

ConvLayer::ConvLayer(Layer * prevLayer, int nOutputs, int kernalSize, int stride) : Layer(prevLayer)
{
	m_numOutputs = nOutputs;
	m_KernalSize = kernalSize;
	m_Stride = stride;

	m_numKernals = prevLayer->GetNumOfOutputs() * m_numOutputs;
	m_OW = (prevLayer->GetOutputWidth() - kernalSize) / stride + 1;
	m_OH = (prevLayer->GetOutputHeight() - kernalSize) / stride + 1;

	m_nThreadsNeeded = m_nOutput…lements = m_OW * m_OH * m_numOutputs;
	m_nKernalElements = m_numKernals * m_KernalSize * m_KernalSize;
}

ConvLayer::~ConvLayer()
{
}

int ConvLayer::FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr)
{
	dim3 nBlocks(m_numOutputs);
	dim3 nThreads(m_OW, m_OH);

	ConvLayerFeed << <nBlocks, nThreads >> > (input_ptr, m_prevLayer->GetOutputWidth(), m_prevLayer->GetOutputHeight(), m_prevLayer->GetNumOfOutputs(), output_ptr, kernel_ptr, m_KernalSize);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		return -1;
	}

	return 1;
}

int ConvLayer::Backpropagate()
{
	return 1;
}
