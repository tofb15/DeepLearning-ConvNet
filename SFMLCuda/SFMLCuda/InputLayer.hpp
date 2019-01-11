#pragma once
#ifndef INPUTLAYER_HPP
#define INPUTLAYER_HPP

#include "Layer.hpp"

class InputLayer : public Layer
{
public:
	InputLayer(int width, int height, int nOutputs = 1);
	~InputLayer();

	virtual int FeedForward(float* input, float* output_ptr, float* kernel_ptr);
	virtual int Backpropagate();
private:

};

#endif // !INPUTLAYER_HPP
