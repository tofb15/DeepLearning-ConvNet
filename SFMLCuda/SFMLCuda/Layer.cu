#include "Layer.hpp"

Layer::Layer(Layer * prevLayer) : m_prevLayer(prevLayer)
{
}

Layer::~Layer()
{}

int Layer::GetNumOfOutputs()
{
	return m_numOutputs;
}

int Layer::GetNumOfKernals()
{
	return m_numKernals;
}

int Layer::GetOutputWidth()
{
	return m_OW;
}

int Layer::GetOutputHeight()
{
	return m_OH;
}

int Layer::GetKernalSize()
{
	return m_KernalSize;
}

int Layer::GetStride()
{
	return m_Stride;
}

int Layer::GetNumOfOutputElements()
{
	return m_nOutput…lements;
}

int Layer::GetNumOfKernelElements()
{
	return m_nKernalElements;
}
