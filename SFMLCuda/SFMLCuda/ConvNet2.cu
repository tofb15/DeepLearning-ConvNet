#include "ConvNet.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdarg>

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

	if(val > 0)
		if(val < 255)
			output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = val;
		else
			output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = 255;
	else
		output[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x] = 0;
}

__global__ void MaxPoolLayer(unsigned char* input, int I_W, int I_H, unsigned char* output, int K_WH, int stride) {
	//Stride locked to K_WH at the moment

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

ConvNet::ConvNet(int I_W, int I_H, int nInputs)
{
	LayerData input;
	input.numOutputs = nInputs;
	input.O_W = I_W;
	input.O_H = I_H;

	m_layers.push_back(input);

	m_dataArraySize = input.O_W * input.O_H * input.numOutputs;
}

ConvNet::~ConvNet()
{
	Destroy();
}
/**
	Adds a new layer to the network. This function should not be called after Initialize().
	This is a variatic function and can take any number of parameters of type int. The number of parameters that will give effect depends of the layertype passed is.
	In case no or to few parameters was passed to the specific layertype the new layer will be givven default values to all missing parameters.
	In case to many parameters was passed to the specific layertype all extra parameters vill be ignored.

	@param type the layer type that should be added.
	@param args the number of extra parameters sent to this function, followd by "args" number of parameters. Exemple AddLayer(LAYER_TYPE::ConvLayer, 2, 1, 1) or AddLayer(LAYER_TYPE::ConvLayer, 0)

	@return
*/
void ConvNet::AddLayer(LAYER_TYPE type, int args ...)
{
	va_list args__;
	va_start(args__, args);
	LayerData layerData;

	layerData.type = type;

	int maxInputs;
	if (type == LAYER_TYPE::ConvLayer) {
		maxInputs = 3;

		//Set user values
		for (size_t i = 0; i < args && i < maxInputs; i++)
		{
			int val = va_arg(args__, int);

			switch (i)
			{
			case 0:
				layerData.kernalSize = val;
				break;
			case 1:
				layerData.numOutputs = val;
				break;
			case 2:
				layerData.stride = val;
				break;
			default:
				break;
			}
		}

		//Set default values
		for (size_t i = args; i < maxInputs; i++)
		{
			switch (i)
			{
			case 0:
				layerData.kernalSize = 3;
				break;
			case 1:
				layerData.numOutputs = 1;
				break;
			case 2:
				layerData.numOutputs = 1;
				break;
			default:
				break;
			}
		}


		//Calculate number of kernals needed
		layerData.numKernals = m_layers[m_layers.size() - 1].numOutputs * layerData.numOutputs;
	}
	else if (type == LAYER_TYPE::PoolLayer) {
		maxInputs = 2;

		//Set user values
		for (size_t i = 0; i < args && i < maxInputs; i++)
		{
			int val = va_arg(args__, int);

			switch (i)
			{
			case 0:
				layerData.kernalSize = val;
				break;
			case 1:
				layerData.stride = val;
				break;
			default:
				break;
			}
		}

		//Set default values
		for (size_t i = args; i < maxInputs; i++)
		{
			switch (i)
			{
			case 0:
				layerData.kernalSize = 2;
				break;
			case 1:
				layerData.stride = layerData.kernalSize;
				break;
			default:
				break;
			}
		}

		//Maxpool have same number of outputs as the layer before it
		layerData.numOutputs = m_layers[m_layers.size() - 1].numOutputs;
	}

	//Calculate Layer Output Dimensions
	layerData.O_W = (m_layers[m_layers.size() - 1].O_W - layerData.kernalSize) / layerData.stride + 1;
	layerData.O_H = (m_layers[m_layers.size() - 1].O_H - layerData.kernalSize) / layerData.stride + 1;
	//Calculate number of threads needed. (One thread per output pixel)
	layerData.numThreads = layerData.O_W * layerData.O_H * layerData.numOutputs;

	//Increese amount of datastorage needed on GPU
	m_kernalArraySize += layerData.numKernals * layerData.kernalSize * layerData.kernalSize * sizeof(float);
	m_dataArraySize += layerData.numThreads;

	//Add new layer to the layer vector
	m_layers.push_back(layerData);

	va_end(args__);
}

void ConvNet::Initialize()
{
	Destroy();
	init = true;

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

	//Kernal
	InitializeKernal();
}

void ConvNet::Feed(unsigned char * inputData)
{
	if (!init)
		return;

	//Cuda Stuff
	cudaError_t error;

	error = cudaMemcpy(d_dataArray, inputData, m_layers[0].O_W * m_layers[0].O_H * m_layers[0].numOutputs, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "cudaMemcpy d_dataArray error: " << cudaGetErrorString(error) << std::endl;
	}

	int input_offset = 0;
	int output_offset = 0;
	int kernal_offset = 0;

	dim3 nBlocks;// (m_layers[0].numOutputs);
	dim3 nThreads;// (m_layers[0].O_W, m_layers[0].O_H);

	//ConvLayerFeed<<<nBlocks, nThreads>>>(d_dataArray, m_I_W, m_I_H, m_numInputs, d_dataArray + output_offset, d_kernalArray, 3);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	std::cout << "ConvLayerFeed error: " << cudaGetErrorString(error) << std::endl;
	//}

	for (size_t i = 1; i < m_layers.size(); i++)
	{
		input_offset = output_offset;
		output_offset += m_layers[i - 1].numOutputs * m_layers[i - 1].O_W * m_layers[i - 1].O_H;
		kernal_offset += m_layers[i - 1].kernalSize * m_layers[i - 1].kernalSize * m_layers[i - 1].numKernals;

		nBlocks = dim3(m_layers[i].numOutputs);
		nThreads = dim3(m_layers[i].O_W, m_layers[i].O_H);

		if(m_layers[i].type == LAYER_TYPE::ConvLayer)
			ConvLayerFeed << <nBlocks, nThreads >> > (d_dataArray + input_offset, m_layers[i - 1].O_W, m_layers[i - 1].O_H, m_layers[i - 1].numOutputs, d_dataArray + output_offset, d_kernalArray, 3);
		else if (m_layers[i].type == LAYER_TYPE::PoolLayer) {
			MaxPoolLayer << <nBlocks, nThreads >> > (d_dataArray + input_offset, m_layers[i - 1].O_W, m_layers[i - 1].O_H, d_dataArray + output_offset, m_layers[i].kernalSize, m_layers[i].stride);
		}

		error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cout << "ConvLayerFeed error #" << i << ": " << cudaGetErrorString(error) << std::endl;
		}

	}

}

//void ConvNet::SetKernalData(const void * kernalData, int bytes, int DeviceOffset)
//{
//	cudaError_t error;
//
//	error = cudaMemcpy(d_kernalArray + DeviceOffset, kernalData, bytes, cudaMemcpyHostToDevice);
//	if (error != cudaSuccess) {
//		std::cout << "SetKernalData error: " << cudaGetErrorString(error) << std::endl;
//	}
//}
//
//void ConvNet::GetKernalData(void * kernalData, int bytes, int DeviceOffset)
//{
//	cudaError_t error;
//
//	error = cudaMemcpy(kernalData, d_kernalArray + DeviceOffset, bytes, cudaMemcpyDeviceToHost);
//	if (error != cudaSuccess) {
//		std::cout << "GetKernalData error: " << cudaGetErrorString(error) << std::endl;
//	}
//}

void ConvNet::GetData(unsigned char * arrayData, int bytes, int DeviceOffset)
{
	cudaError_t error;

	error = cudaMemcpy(arrayData, d_dataArray + DeviceOffset, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetData error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::GetData(unsigned char * arrayData, int & dataWidth, int & dataHeight, int maxBytes, int layerIndex, int outputIndex)
{
	cudaError_t error;

	dataWidth = m_layers[layerIndex].O_W;
	dataHeight = m_layers[layerIndex].O_H;

	int start = 0;
	int outputSize = m_layers[layerIndex].O_W * m_layers[layerIndex].O_H;
	int read = (outputSize < maxBytes) ? outputSize : maxBytes;

	for (size_t i = 0; i < layerIndex; i++)
	{
		start += m_layers[i].numOutputs * m_layers[i].O_W * m_layers[i].O_H;
	}
	start += outputIndex * m_layers[layerIndex].O_W * m_layers[layerIndex].O_H;

	error = cudaMemcpy(arrayData, d_dataArray + start, outputSize, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetData layer/output error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::Destroy()
{
	if (!init)
		return;

	//Free Host
	delete[] h_kernalArray;
	//delete[] perLayerData;

	//Free Device
	cudaFree(d_dataArray);
	cudaFree(d_kernalArray);
}

void ConvNet::InitializeKernal()
{
	int startInitFromIndex = 0;
	h_kernalArray = new float[m_kernalArraySize];

	int left = 0;
	int i = 0;
	int j = 0;
	do
	{
		if (i + 1 < m_layers.size())
			left = m_layers[i].numOutputs * m_layers[i + 1].numOutputs;

		while (left > 0)
		{
			h_kernalArray[j++] = 0;
			h_kernalArray[j++] = -1;
			h_kernalArray[j++] = 0;
			h_kernalArray[j++] = -1;
			h_kernalArray[j++] = 4;
			h_kernalArray[j++] = -1;
			h_kernalArray[j++] = 0;
			h_kernalArray[j++] = -1;
			h_kernalArray[j++] = 0;
			left--;
		}

		i++;
	} while (i < m_layers.size() - 1);

	cudaError_t error;

 	error = cudaMemcpy(d_kernalArray, h_kernalArray, m_kernalArraySize, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "SetKernalData error: " << cudaGetErrorString(error) << std::endl;
	}
}
