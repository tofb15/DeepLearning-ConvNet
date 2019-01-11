#pragma once
#ifndef NNLAYER_HPP
#define NNLAYER_HPP

#include "Layer.hpp"

class NNLayer : public Layer
{
public:
	NNLayer(Layer* prevLayer, int nNeurons);
	~NNLayer();

	virtual int FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr = nullptr);
	virtual int Backpropagate();
private:

};

#endif // !NNLAYER_HPP
