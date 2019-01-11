/**
	ConvNet2.cpp
	Purpose: ConvNet implementation

	@author Tobias Fast
	@version 1.0 9/12/18
*/

#include "ConvNet.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdarg>

#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "InputLayer.hpp"

ConvNet::ConvNet(int I_W, int I_H, int nInputs)
{

	InputLayer* input = new InputLayer(I_W, I_H, nInputs);
	m_layers.push_back(input);
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
//void ConvNet::AddLayer(LAYER_TYPE type, int args ...)
//{
//	va_list args__;
//	va_start(args__, args);
//	LayerData layerData;
//
//	layerData.type = type;
//
//	int maxInputs;
//	if (type == LAYER_TYPE::ConvLayer) {
//		maxInputs = 3;
//
//		//Set user values
//		for (size_t i = 0; i < args && i < maxInputs; i++)
//		{
//			int val = va_arg(args__, int);
//
//			switch (i)
//			{
//			case 0:
//				layerData.kernalSize = val;
//				break;
//			case 1:
//				layerData.numOutputs = val;
//				break;
//			case 2:
//				layerData.stride = val;
//				break;
//			default:
//				break;
//			}
//		}
//
//		//Set default values
//		for (size_t i = args; i < maxInputs; i++)
//		{
//			switch (i)
//			{
//			case 0:
//				layerData.kernalSize = 3;
//				break;
//			case 1:
//				layerData.numOutputs = 1;
//				break;
//			case 2:
//				layerData.stride = 1;
//				break;
//			default:
//				break;
//			}
//		}
//
//
//		//Calculate number of kernals needed
//		layerData.numKernals = m_layers[m_layers.size() - 1].numOutputs * layerData.numOutputs;
//
//		//Calculate Layer Output Dimensions
//		layerData.O_W = (m_layers[m_layers.size() - 1].O_W - layerData.kernalSize) / layerData.stride + 1;
//		layerData.O_H = (m_layers[m_layers.size() - 1].O_H - layerData.kernalSize) / layerData.stride + 1;
//	}
//	else if (type == LAYER_TYPE::PoolLayer) {
//		maxInputs = 2;
//
//		//Set user values
//		for (size_t i = 0; i < args && i < maxInputs; i++)
//		{
//			int val = va_arg(args__, int);
//
//			switch (i)
//			{
//			case 0:
//				layerData.kernalSize = val;
//				break;
//			case 1:
//				layerData.stride = val;
//				break;
//			default:
//				break;
//			}
//		}
//
//		//Set default values
//		for (size_t i = args; i < maxInputs; i++)
//		{
//			switch (i)
//			{
//			case 0:
//				layerData.kernalSize = 2;
//				break;
//			case 1:
//				layerData.stride = layerData.kernalSize;
//				break;
//			default:
//				break;
//			}
//		}
//
//		//Maxpool have same number of outputs as the layer before it
//		layerData.numOutputs = m_layers[m_layers.size() - 1].numOutputs;
//
//		//Calculate Layer Output Dimensions
//		layerData.O_W = (m_layers[m_layers.size() - 1].O_W - layerData.kernalSize) / layerData.stride + 1;
//		layerData.O_H = (m_layers[m_layers.size() - 1].O_H - layerData.kernalSize) / layerData.stride + 1;
//	}
//	else if (type == LAYER_TYPE::NN) {
//
//		maxInputs = 1;
//
//		//Set user values
//		for (size_t i = 0; i < args && i < maxInputs; i++)
//		{
//			int val = va_arg(args__, int);
//
//			switch (i)
//			{
//			case 0:
//				layerData.O_W = val;
//				break;
//			default:
//				break;
//			}
//		}
//
//		//Set default values
//		for (size_t i = args; i < maxInputs; i++)
//		{
//			switch (i)
//			{
//			case 0:
//				layerData.O_W = 10;
//				break;
//			default:
//				break;
//			}
//		}
//
//
//		layerData.numOutputs = 1;
//		layerData.O_H = 1;
//		layerData.kernalSize = 1;
//
//		//Calculate number of kernals needed
//		layerData.numKernals = m_layers[m_layers.size() - 1].numOutputs *  m_layers[m_layers.size() - 1].O_H * m_layers[m_layers.size() - 1].O_W * layerData.O_W;
//
//	}
//
//	//Calculate number of threads needed. (One thread per output pixel)
//	layerData.numThreads = layerData.O_W * layerData.O_H * layerData.numOutputs;
//
//	//Increese amount of datastorage needed on GPU
//	m_kernalArraySize += layerData.numKernals * layerData.kernalSize * layerData.kernalSize * sizeof(float);
//	m_dataArraySize += layerData.numThreads * sizeof(float);
//
//	//Add new layer to the layer vector
//	m_layers.push_back(layerData);
//
//	va_end(args__);
//}
//
//void ConvNet::Initialize()
//{
//	Destroy();
//	init = true;
//
//	//Cuda Stuff
//	cudaError_t error;
//
//	error = cudaMalloc((void**)&d_dataArray, m_dataArraySize);
//	if (error != cudaSuccess) {
//		std::cout << "Malloc d_dataArray error: " << cudaGetErrorString(error) << std::endl;
//	}
//
//	error = cudaMalloc((void**)&d_kernalArray, m_kernalArraySize);
//	if (error != cudaSuccess) {
//		std::cout << "Malloc d_kernalArray error: " << cudaGetErrorString(error) << std::endl;
//	}
//
//	//Kernal
//	InitializeKernal();
//}

int ConvNet::Feed(float* inputData)
{
	if (!init)
		return -2;

	int input_offset = 0;
	int output_offset = m_layers[0]->GetNumOfOutputElements();
	int kernal_offset = 0;

	//Cuda Stuff
	cudaError_t error;

	error = cudaMemcpy(d_dataArray, inputData, output_offset * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "cudaMemcpy d_dataArray error: " << cudaGetErrorString(error) << std::endl;
	}

	int nLayers = m_layers.size();

	for (size_t i = 1; i < nLayers; i++)
	{
		int result = m_layers[i]->FeedForward(d_dataArray + input_offset, d_dataArray + output_offset, d_kernalArray + kernal_offset);
		if (!result)
			return result;
		input_offset = output_offset;
		output_offset += m_layers[i]->GetNumOfOutputElements();
		kernal_offset += m_layers[i]->GetNumOfKernelElements();
	}

	return 1;

	//dim3 nBlocks;// (m_layers[0].numOutputs);
	//dim3 nThreads;// (m_layers[0].O_W, m_layers[0].O_H);

	//ConvLayerFeed<<<nBlocks, nThreads>>>(d_dataArray, m_I_W, m_I_H, m_numInputs, d_dataArray + output_offset, d_kernalArray, 3);
	//error = cudaGetLastError();
	//if (error != cudaSuccess) {
	//	std::cout << "ConvLayerFeed error: " << cudaGetErrorString(error) << std::endl;
	//}

	//for (size_t i = 1; i < m_layers.size(); i++)
	//{
	//	input_offset = output_offset;
	//	output_offset += m_layers[i - 1].numOutputs * m_layers[i - 1].O_W * m_layers[i - 1].O_H;
	//	kernal_offset += m_layers[i - 1].kernalSize * m_layers[i - 1].kernalSize * m_layers[i - 1].numKernals;

	//	nBlocks = dim3(m_layers[i].numOutputs);
	//	nThreads = dim3(m_layers[i].O_W, m_layers[i].O_H);

	//	if(m_layers[i].type == LAYER_TYPE::ConvLayer){
	//		ConvLayerFeed << <nBlocks, nThreads >> > (d_dataArray + input_offset, m_layers[i - 1].O_W, m_layers[i - 1].O_H, m_layers[i - 1].numOutputs, d_dataArray + output_offset, d_kernalArray + kernal_offset, 3);
	//	}
	//	else if (m_layers[i].type == LAYER_TYPE::PoolLayer) {
	//		MaxPoolLayer << <nBlocks, nThreads >> > (d_dataArray + input_offset, m_layers[i - 1].O_W, m_layers[i - 1].O_H, d_dataArray + output_offset, m_layers[i].kernalSize, m_layers[i].stride);
	//	}
	//	else if (m_layers[i].type == LAYER_TYPE::NN) {
	//		NNLayer << <nBlocks, nThreads >> > (d_dataArray + input_offset, m_layers[i - 1].O_W * m_layers[i - 1].O_H * m_layers[i - 1].numOutputs, d_dataArray + output_offset, d_kernalArray + kernal_offset);
	//	}

	//	error = cudaGetLastError();
	//	if (error != cudaSuccess) {
	//		std::cout << "ConvLayerFeed error #" << i << ": " << cudaGetErrorString(error) << std::endl;
	//	}

	//}

}

Layer * ConvNet::GetLastLayer()
{
	return m_layers[m_layers.size()-1];
}

Layer * ConvNet::operator[](int i)
{
	return m_layers[i];
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

void ConvNet::GetData(float* arrayData, int bytes, int DeviceOffset)
{
	cudaError_t error;

	error = cudaMemcpy(arrayData, d_dataArray + DeviceOffset, bytes, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetData error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::GetData(float* arrayData, int & dataWidth, int & dataHeight, int maxBytes, int layerIndex, int outputIndex)
{
	cudaError_t error;

	dataWidth = m_layers[layerIndex]->GetOutputWidth();
	dataHeight = m_layers[layerIndex]->GetOutputHeight();

	int start = 0;
	int outputSize = dataWidth * dataHeight * sizeof(float);
	int read = (outputSize < maxBytes) ? outputSize : maxBytes;

	for (size_t i = 0; i < layerIndex; i++)
	{
		start += m_layers[i]->GetNumOfOutputElements();
	}
	start += outputIndex * dataWidth * dataHeight;

	error = cudaMemcpy(arrayData, d_dataArray + start, outputSize, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "GetData layer/output error: " << cudaGetErrorString(error) << std::endl;
	}
}

void ConvNet::AddLayer(Layer * layer)
{
	m_layers.push_back(layer);
}

void ConvNet::CreateNetwork()
{
	Destroy();
	init = true;

	int nLayers = m_layers.size();
	for (size_t i = 0; i < nLayers; i++)
	{
		m_dataArraySize		+= m_layers[i]->GetNumOfOutputElements() * sizeof(float);
		m_kernelArraySize	+= m_layers[i]->GetNumOfKernelElements() * sizeof(float);
	}

	//Allocate GPU memory needed with Cuda
	cudaError_t error;

	error = cudaMalloc((void**)&d_dataArray, m_dataArraySize);
	if (error != cudaSuccess) {
		std::cout << "Malloc d_dataArray error: " << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMalloc((void**)&d_kernalArray, m_kernelArraySize);
	if (error != cudaSuccess) {
		std::cout << "Malloc d_kernalArray error: " << cudaGetErrorString(error) << std::endl;
	}

	//Kernal
	InitializeKernal();
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
	h_kernalArray = new float[m_kernelArraySize];

	int left = 0;
	int i = 1;
	int j = 0;
	do
	{
		if (i < m_layers.size())
			if (m_layers[i]->GetKernalSize() == 3)/*Special case used for testing. This should be removed*/
			{
				for (size_t k = 0; k < m_layers[i]->GetNumOfKernals(); k++)
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
				}
			}
			else {/*General case*/
				for (size_t k = 0; k < m_layers[i]->GetNumOfKernals(); k++)
				{
					for (size_t k2 = 0; k2 < m_layers[i]->GetKernalSize() * m_layers[i]->GetKernalSize(); k2++)
					{
						float w = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
						h_kernalArray[j++] = w;
					}
				}
			}

		i++;
	} while (i < m_layers.size());

	cudaError_t error;

 	error = cudaMemcpy(d_kernalArray, h_kernalArray, m_kernelArraySize, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "SetKernalData error: " << cudaGetErrorString(error) << std::endl;
	}
}
