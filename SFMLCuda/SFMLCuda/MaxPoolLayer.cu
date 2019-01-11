#include "MaxPoolLayer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MaxPoolLayerFeed(float* input, int I_W, int I_H, float* output, int K_WH, int stride) {

	float max = 0;
	int in_x = threadIdx.x * stride;
	int in_y = threadIdx.y * stride;

	for (size_t ky = 0; ky < K_WH; ky++)
	{
		for (size_t kx = 0; kx < K_WH; kx++)
		{
			float val = input[(in_x + kx) + (in_y + ky) * I_W + blockIdx.x * I_W * I_H];
			if (val > max)
				max = val;
		}
	}

	output[threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y] = max;
}

MaxPoolLayer::MaxPoolLayer(Layer* prevLayer, int kernalSize, int stride) : Layer(prevLayer)
{
	m_KernalSize = kernalSize;
	m_Stride = (stride == -1) ? m_KernalSize : stride;

	m_numOutputs = prevLayer->GetNumOfOutputs();
	m_OW = (prevLayer->GetOutputWidth() - kernalSize) / m_Stride + 1;
	m_OH = (prevLayer->GetOutputHeight() - kernalSize) / m_Stride + 1;

	m_nThreadsNeeded = m_nOutputÉlements = m_OW * m_OH * m_numOutputs;
	m_numKernals = 0;// prevLayer->GetNumOfOutputs();
	m_nKernalElements = 0;
}

MaxPoolLayer::~MaxPoolLayer()
{
}

int MaxPoolLayer::FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr)
{
	dim3 nBlocks(m_numOutputs);
	dim3 nThreads(m_OW, m_OH);

	MaxPoolLayerFeed << <nBlocks, nThreads >> > (input_ptr, m_prevLayer->GetOutputWidth(), m_prevLayer->GetOutputHeight(), output_ptr, m_KernalSize, m_Stride);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		return -1;
	}

	return 1;
}

int MaxPoolLayer::Backpropagate()
{
	return 1;
}
