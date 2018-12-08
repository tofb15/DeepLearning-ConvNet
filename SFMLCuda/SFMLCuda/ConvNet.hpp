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
	//LAYER_TYPE type;
	int kernalSize;
	int numOutputs;
	int O_W, O_H;
};

struct PerLayerData
{
	int numKernals;
	int numThreads;
	//int kernalOffset;
	//int dataInputOffset;
};

class ConvNet
{
public:
	ConvNet(std::vector<LayerData> layers);
	~ConvNet();

	void Initialize(int I_W, int I_H, int numInputs);
	void Feed(unsigned char* inputData);
	//void SetKernalData(const void* kernalData, int bytes, int DeviceOffset = 0);
	//void GetKernalData(void* kernalData, int bytes, int DeviceOffset = 0);
	void GetData(unsigned char* arrayData, int bytes, int DeviceOffset = 0);

private:
	bool init = false;
	std::vector<LayerData> m_layers;

	int m_I_W, m_I_H, m_numInputs, m_I_Size;

	float * h_kernalArray;
	PerLayerData* perLayerData;

	int m_kernalArraySize;	//Size in bytes
	int m_dataArraySize;	//Size in bytes

	unsigned char* d_dataArray;
	float* d_kernalArray;

	void Destroy();
	void InitializeKernal();

};

#endif // !CONVNET?HPP



