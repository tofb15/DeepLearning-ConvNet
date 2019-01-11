#pragma once
#ifndef CONVLAYER_HPP
#define CONVLAYER_HPP

#include "Layer.hpp"

class ConvLayer : public Layer
{
public:
	ConvLayer(Layer* prevLayer, int nOutputs = 1, int kernalSize = 3, int stride = 1);
	~ConvLayer();

	virtual int FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr);
	virtual int Backpropagate();
private:

};

#endif