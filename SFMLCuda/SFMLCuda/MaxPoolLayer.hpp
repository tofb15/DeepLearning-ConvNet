#pragma once
#ifndef MAXPOOLLAYER_HPP
#define MAXPOOLLAYER_HPP

#include "Layer.hpp"

class MaxPoolLayer : public Layer
{
public:
	MaxPoolLayer(Layer* prevLayer, int kernalSize = 2, int stride = -1);
	~MaxPoolLayer();

	virtual int FeedForward(float* input_ptr, float* output_ptr, float* kernel_ptr = nullptr);
	virtual int Backpropagate();
private:

};

#endif // !MAXPOOLLAYER_HPP
