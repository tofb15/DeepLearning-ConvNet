#include "ConvNet.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void ConvLayerFeed(unsigned char* input, int I_W, int I_H, int nInputs, unsigned char* output, float* kernals, int K_WH) {

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

	output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = val;
}

ConvNet::ConvNet(std::vector<LayerData> layers)
{
	m_layers = layers;
}

ConvNet::~ConvNet()
{
	Destroy();
}

void ConvNet::Init(int I_W, int I_H, int numInputs)
{
	Destroy();
	init = true;

	m_I_W = I_W;
	m_I_H = I_H;
	m_numInputs = numInputs;
	m_I_Size = I_W * I_H;

	//h_input = new unsigned char[m_I_Size];
	perLayerData = new PerLayerData[m_layers.size()];

	m_layers[0].O_W = m_I_W - m_layers[0].kernalSize + 1;
	m_layers[0].O_H = m_I_H - m_layers[0].kernalSize + 1;

	perLayerData[0].numKernals = m_numInputs * m_layers[0].numOutputs;
	perLayerData[0].numThreads = m_layers[0].O_W * m_layers[0].O_H * m_layers[0].numOutputs;
	//perLayerData[0].kernalOffset = 0;
	//perLayerData[0].dataInputOffset = 0;

	m_kernalArraySize = perLayerData[0].numKernals * m_layers[0].kernalSize * m_layers[0].kernalSize * sizeof(float);
	m_dataArraySize = m_I_Size * m_numInputs;										//Input
	m_dataArraySize += m_layers[0].O_W * m_layers[0].O_H * m_layers[0].numOutputs;	//FirstOutput

	for (size_t i = 1; i < m_layers.size(); i++)
	{
		//perLayerData[i].kernalOffset = m_kernalArraySize;
		//perLayerData[i].dataInputOffset = m_dataArraySize;

		m_layers[i].O_W = m_layers[i - 1].O_W - m_layers[i].kernalSize + 1;
		m_layers[i].O_H = m_layers[i - 1].O_H - m_layers[i].kernalSize + 1;

		perLayerData[i].numKernals = m_layers[i - 1].numOutputs * m_layers[i].numOutputs;
		perLayerData[i].numThreads = m_layers[i].O_W * m_layers[i].O_H * m_layers[i].numOutputs;

		m_kernalArraySize += perLayerData[i].numKernals * m_layers[i].kernalSize * m_layers[i].kernalSize * sizeof(float);
		m_dataArraySize += m_layers[i].O_W * m_layers[i].O_H * m_layers[i].numOutputs;
	}

	//Cuda Stuff
	cudaError_t error;

	error = cudaMalloc((void**)&d_dataArray, m_dataArraySize);
	if (error != cudaSuccess) {
		std::cout << "Malloc d_dataArray error: " << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMalloc((void**)&d_kernalArray, m_kernalArraySize);
	if (error != cudaSuccess) {
		std::cout << "Malloc d_kernalArray error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::Feed(unsigned char * inputData)
{
	if (!init)
		return;

	//Cuda Stuff
	cudaError_t error;

	error = cudaMemcpy(d_dataArray, inputData, m_numInputs * m_I_Size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "cudaMemcpy d_dataArray error: " << cudaGetErrorString(error) << std::endl;
	}

	int input_offset = 0;
	int output_offset = m_I_Size * m_numInputs;
	int kernal_offset = 0;

	dim3 nBlocks(m_layers[0].numOutputs);
	dim3 nThreads(m_layers[0].O_W, m_layers[0].O_H);

	ConvLayerFeed<<<nBlocks, nThreads>>>(d_dataArray, m_I_W, m_I_H, m_numInputs, d_dataArray + output_offset, d_kernalArray, 3);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cout << "ConvLayerFeed error: " << cudaGetErrorString(error) << std::endl;
	}

	//for (size_t i = 1; i < m_layers.size(); i++)
	//{
	//	nBlocks = dim3(m_layers[i].numOutputs);
	//	nThreads = dim3(m_layers[i].O_W, m_layers[i].O_H, 0);

	//	ConvLayerFeed<<<nBlocks, nThreads>>>(d_dataArray  + input_offset, m_layers[i].O_W, m_layers[i].O_H, m_layers[i - 1].numOutputs, d_dataArray + output_offset, m_kernalData + kernal_offset, m_layers[i].kernalSize);
	//	input_offset = output_offset;
	//	kernal_offset += m_layers[i].kernalSize * m_layers[i].kernalSize * perLayerData[i].numKernals * sizeof(float);
	//	output_offset += m_layers[i].numOutputs * m_layers[i].O_W * m_layers[i].O_H;
	//}

}

void ConvNet::SetKernalData(const void * kernalData, int bytes, int DeviceOffset)
{
	cudaError_t error;

	error = cudaMemcpy(d_kernalArray + DeviceOffset, kernalData, bytes, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "SetKernalData error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::GetKernalData(void * kernalData, int bytes, int DeviceOffset)
{
	cudaError_t error;

	error = cudaMemcpy(kernalData, d_kernalArray + DeviceOffset, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetKernalData error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::GetData(unsigned char * arrayData, int bytes, int DeviceOffset)
{
	cudaError_t error;

	error = cudaMemcpy(arrayData, d_dataArray + DeviceOffset, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetData error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::Destroy()
{
	if (!init)
		return;

	//Free Host
	delete[] m_kernalData;
	delete[] perLayerData;

	//Free Device
	cudaFree(d_dataArray);
	cudaFree(d_kernalArray);
}
