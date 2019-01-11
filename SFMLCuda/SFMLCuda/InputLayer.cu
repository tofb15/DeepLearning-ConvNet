#include "InputLayer.hpp"

InputLayer::InputLayer(int width, int height, int nOutputs) : Layer(nullptr)
{
	m_OW = width;
	m_OH = height;
	m_numOutputs = nOutputs;
	m_nOutput…lements = m_OW * m_OH * m_numOutputs;

	m_KernalSize = m_nKernalElements = m_numKernals = 0;
	m_Stride = 0;
	m_nThreadsNeeded = 0;
}

InputLayer::~InputLayer()
{}

//Should not be called
int InputLayer::FeedForward(float * input_ptr, float * output_ptr, float* kernel_ptr)
{
	return -2;
}

//Should not be called
int InputLayer::Backpropagate()
{
	return -2;
}