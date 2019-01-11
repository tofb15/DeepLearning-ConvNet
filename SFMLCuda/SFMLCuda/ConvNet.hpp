/**
	ConvNet.hpp
	Purpose: ConvNet implementation

	@author Tobias Fast
	@version 1.0 9/12/18
*/

#pragma once
#ifndef CONVNET_HPP
#define CONVNET_HPP

#include <vector>
#include "Layer.hpp"

//enum class LAYER_TYPE
//{
//	ConvLayer,
//	PoolLayer,
//	NN
//};

//struct PerLayerData
//{
//	int numKernals;
//	int numThreads;
//	//int kernalOffset;
//	//int dataInputOffset;
//};

class ConvNet
{
public:
	ConvNet(int nInputs, int I_W, int I_H);
	~ConvNet();

	//void AddLayer(LAYER_TYPE type, int args...);

	void AddLayer(Layer* layer);
	void CreateNetwork();
	int Feed(float* inputData);
	Layer* GetLastLayer();

	//void SetKernalData(const void* kernalData, int bytes, int DeviceOffset = 0);
	//void GetKernalData(void* kernalData, int bytes, int DeviceOffset = 0);
	void GetData(float* arrayData, int bytes, int DeviceOffset = 0);
	void GetData(float* arrayData, int& dataWidth, int& dataHeight, int maxBytes, int layerIndex = 0, int outputIndex = 0);


	Layer* operator[](int i);
private:
	std::vector<Layer*> m_layers;
	int m_dataArraySize = 0;
	int m_kernelArraySize = 0;
	bool init = true;
	
	float * d_dataArray;		// pointer to GPU layerdata storage
	float * d_kernalArray;		// pointer to GPU kernal storage
	float * h_kernalArray;		//Kernal data will be Initialize here before sent to the GPU during initialization.

	//struct LayerData {
	//	LAYER_TYPE type;
	//	int kernalSize = 0;
	//	int numOutputs = 0;
	//	int O_W = 0, O_H = 0;
	//	int numKernals = 0;
	//	int numThreads = 0;
	//	int stride = 1;
	//};

	//std::vector<LayerData> m_layers;


	//int m_kernalArraySize = 0;	//Size in bytes needed on the GPU to store all kernals.
	//int m_dataArraySize = 0;	//Size in bytes needed on the GPU to store the input and all layer outputs.

	void Destroy();
	void InitializeKernal();

};

#endif // !CONVNET?HPP
