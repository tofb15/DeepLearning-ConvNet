#pragma once
#ifndef CONVNET_HPP
#define CONVNET_HPP

#include <vector>

enum class LAYER_TYPE
{
	ConvLayer,
	PoolLayer,
	NN
};

struct LayerData {
	LAYER_TYPE type;
	int kernalSize = 0;
	int numOutputs = 0;
	int O_W = 0, O_H = 0;
	int numKernals = 0;
	int numThreads = 0;
};

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

	void AddLayer(LAYER_TYPE type, int args...);

	void Initialize();
	void Feed(unsigned char* inputData);
	//void SetKernalData(const void* kernalData, int bytes, int DeviceOffset = 0);
	//void GetKernalData(void* kernalData, int bytes, int DeviceOffset = 0);
	void GetData(unsigned char* arrayData, int bytes, int DeviceOffset = 0);

private:
	bool init = false;
	std::vector<LayerData> m_layers;

	float * h_kernalArray;  //Kernal Data will be Initialize here before sent to the GPU during initialization.

	int m_kernalArraySize;	//Size in bytes needed on the GPU to store all kernals.
	int m_dataArraySize;	//Size in bytes needed on the GPU to store the input and all layer outputs.

	unsigned char* d_dataArray; // pointer to GPU layerdata storage
	float* d_kernalArray;		// pointer to GPU kernal storage

	void Destroy();
	void InitializeKernal();

};

#endif // !CONVNET?HPP
