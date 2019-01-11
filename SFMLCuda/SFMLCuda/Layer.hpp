#pragma once
#ifndef LAYER_HPP
#define LAYER_HPP

class Layer
{
public:
	Layer(Layer* prevLayer);
	~Layer();
	virtual int FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr) = 0;
	virtual int Backpropagate() = 0;

	virtual int GetNumOfOutputs();
	virtual int GetNumOfKernals();
	virtual int GetOutputWidth();
	virtual int GetOutputHeight();
	virtual int GetKernalSize();
	virtual int GetStride();

	virtual int GetNumOfOutputElements();
	virtual int GetNumOfKernelElements();
protected:
	Layer* m_prevLayer;

	int m_numKernals;
	int m_numOutputs;
	int m_OW;
	int m_OH;
	int m_KernalSize;
	int m_Stride;

	int m_nThreadsNeeded;
	int m_nOutput…lements;
	int m_nKernalElements;
private:
	void* m_input;
	void* m_output;
};



#endif // !LAYER_HPP