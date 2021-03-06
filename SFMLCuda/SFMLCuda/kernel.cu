
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <SFML/Graphics.hpp>
#include <fstream>

#include "ConvNet.hpp"

#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

//Remove This
__global__ void applyKernal(float* inputMaps, int numInputs, int I_W, int I_H, float* outputMaps, int numOutputs, float* kernalData, int K_WH) {

	int id = threadIdx.x;

	//int numKernals = numInputs * numOutputs;
	int O_W = I_W - K_WH + 1;
	int O_H = I_H - K_WH + 1;

	int O_index = id / (O_W*O_H);
	int K_WH_2 = K_WH * K_WH;
	int firstKernal = O_index * 2 * K_WH_2;

	int O_X = id % O_W;
	int O_Y = id / O_W;

	float val = 0;

	for (size_t inp = 0; inp < numInputs; inp++)
	{
		int inpStart = inp * I_W * I_H;
		int currentKernal = firstKernal + inp * K_WH_2;

		for (size_t kx = 0; kx < K_WH; kx++)
		{
			int inpX = O_X + kx;
			for (size_t ky = 0; ky < K_WH; ky++)
			{
				val += inputMaps[inpX + (O_Y+ky)*I_W + inpStart] * kernalData[currentKernal + kx + ky * K_WH];
			}
		}
	}
	
	if(val > 0)
		outputMaps[id] = val;
	else
		outputMaps[id] = 0;

}

//Remove This
__global__ void applyKernal_Dummy(char* inputMaps, int numInputs, int I_W, int I_H, char* outputMaps, int numOutputs, float* kernalData, int K_WH) {

	int id = threadIdx.x;
	outputMaps[id] = 50;

}

//Remove This
void initKernals(int kernalSize, int numKernals, float* kernalData) {
	/*for (size_t kx = 0; kx < kernalSize; kx++)
	{
		for (size_t ky = 0; ky < kernalSize; ky++)
		{

		}
	}*/

	kernalData[0] = 0;
	kernalData[1] = -1;
	kernalData[2] = 0;

	kernalData[3] = -1;
	kernalData[4] = 4;
	kernalData[5] = -1;

	kernalData[6] = 0;
	kernalData[7] = -1;
	kernalData[8] = 0;

	//

	kernalData[9] = -1;
	kernalData[10] = 0;
	kernalData[11] = 0;

	kernalData[12] = 0;
	kernalData[13] = 2;
	kernalData[14] = 0;

	kernalData[15] = 0;
	kernalData[16] = 0;
	kernalData[17] = -1;

	//

	kernalData[18] = 0;
	kernalData[19] = 0;
	kernalData[20] = -1;

	kernalData[21] = 0;
	kernalData[22] = 2;
	kernalData[23] = 0;

	kernalData[24] = -1;
	kernalData[25] = 0;
	kernalData[26] = 0;

}

//Remove This
void Kernal(float * imgData, float * imgData_Out) {

	int kernalSize = 3;
	float * kernalData = new float[kernalSize*kernalSize*3];
	initKernals(kernalSize, 3, kernalData);

	int O_W = 32 - kernalSize + 1; 
	int O_H = 32 - kernalSize + 1; 
	int nThreads = O_W * O_H;

	float* d_in;
	float* d_out; 
	float* d_kernal;

	cudaError_t error;

	error = cudaMalloc((void**)&d_in, 1024 * 3 * sizeof(float));
	if (error != cudaSuccess) {
		std::cout << "Malloc in error" << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMalloc((void**)&d_out, 900 * sizeof(float));
	if (error != cudaSuccess) {
		std::cout << "Malloc out error" << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMalloc((void**)&d_kernal, 9 * sizeof(float));
	if (error != cudaSuccess) {
		std::cout << "Malloc kernal error" << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMemcpy(d_in, imgData, 1024 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "cpy in error" << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMemcpy(d_kernal, kernalData, 9 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		std::cout << "cpy out error" << cudaGetErrorString(error) << std::endl;
	}

	applyKernal << < 1, nThreads >> > (d_in, 1, 32, 32, d_out, 1, d_kernal, kernalSize);
	error = cudaGetLastError();

	if (error != cudaSuccess) {
		std::cout << "Launch error" << cudaGetErrorString(error) << std::endl;
	}

	error = cudaMemcpy(imgData_Out, d_out, 900 * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		std::cout << "cpy out error" << cudaGetErrorString(error) << std::endl;
	}

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_kernal);

	cudaDeviceSynchronize();
}

void charToFloat(float* dest, const unsigned char* src, int nElements) {
	for (size_t i = 0; i < nElements; i++)
	{
		dest[i] = (float)(src[i]) / 255.0f;
	}
}

void loadImg(float* data, int imgID) {

	unsigned char temp[3072];
	std::ifstream in("cifar-10-batches-bin/data_batch_1.bin", std::ios::binary);
	char class_id;
	in.seekg(imgID * 3073 + 1, in.beg);
	in.read((char *)(temp), 3072);
	in.close();

	charToFloat(data, temp, 3072);
}

void getLayerOutputAsImage(char* data, ConvNet* network, int layerIndex, int outputIndex) {

}

int main()
{
	std::srand(time(NULL));

	int curImg = 4;
	float imgData[3072];
	float imgData_out[3072];

	loadImg(imgData, curImg);
	float * kernalData = new float[3 * 3 * 3];
	initKernals(3, 3, kernalData);

	ConvNet net(32, 32, 1);
	net.AddLayer(LAYER_TYPE::ConvLayer, 2, 3, 1);
	//net.AddLayer(LAYER_TYPE::ConvLayer, 2, 3, 1);
	net.AddLayer(LAYER_TYPE::PoolLayer, 1, 2);
	net.AddLayer(LAYER_TYPE::NN, 1, 10);
	net.AddLayer(LAYER_TYPE::NN, 1, 10);


	net.Initialize();
	net.Feed(imgData);

	//int ow = 0;
	//int oh = 0;
	//net.GetData(imgData_out, ow, oh, 3072, 4, 0);

	///*New Program*/return 0;

	int ImagesX = 4;
	int ImagesY = 5;
	int ImagesCount = ImagesX * ImagesY;

	int ScaleX = 2;
	int ScaleY = 2;

	sf::Image img;
	sf::Texture texture;
	sf::Sprite sprite;

	//loadImg(imgData, curImg);
	//Kernal(imgData, imgData_out);

	img.create(32 * ImagesX * ScaleX, 32 * ImagesY * ScaleY);

	sf::RenderWindow window(sf::VideoMode(32 * ImagesX * ScaleX, 32 * ImagesY * ScaleY), "ConvNet");
	
	for (size_t i = 0; i < 3; i++)
	{
		int ow = 0;
		int oh = 0;
		net.GetData(imgData_out, ow, oh, 3072, i, 0);

		int imgPosOfset = - 3 + i;

		for (size_t x = 0; x < 32; x++)
		{
			for (size_t y = 0; y < 32; y++)
			{
				int posOrig = x + y * 32;
				int posNew = (x - 1) + (y - 1) * 30;

				for (size_t sx = 0; sx < ScaleX; sx++)
				{
					for (size_t sy = 0; sy < ScaleY; sy++)
					{
						int diffX = (32 - ow) / 2;
						int diffY = (32 - ow) / 2;

						if (x < diffX || x >= 32 - diffX || y < diffY || y >= 32 - diffY)
							img.setPixel(x * ScaleX + sx + 32 * ScaleX*((curImg * 2 + imgPosOfset) % ImagesX), y * ScaleY + sy + 32 * ScaleY * (((curImg * 2 + imgPosOfset) % ImagesCount) / ImagesX), sf::Color(255, 255, 255, 255));
						else
							img.setPixel(x * ScaleX + sx + 32 * ScaleX*((curImg * 2 + imgPosOfset) % ImagesX), y * ScaleY + sy + 32 * ScaleY * (((curImg * 2 + imgPosOfset) % ImagesCount) / ImagesX), sf::Color(imgData_out[(x - diffX) + (y - diffY) * ow] * 255, imgData_out[(x - diffX) + (y - diffY) * ow + 0] * 255, imgData_out[(x - diffX) + (y - diffY) * ow + 0] * 255, 255));
					}
				}
			}
		}
	}

	

	curImg++;

	texture.loadFromImage(img);
	sprite.setTexture(texture);

	sf::Clock clock;

	clock.restart();
	float dt = 0;
	while (window.isOpen())
	{
		// Process events
		sf::Event event;
		while (window.pollEvent(event))
		{
			// Close window: exit
			if (event.type == sf::Event::Closed)
				window.close();
		}

		dt += clock.restart().asSeconds();
		window.clear();

		if (dt >= 5) {
			dt = 0;
			loadImg(imgData, curImg-1);

			for (size_t x = 0; x < 32; x++)
			{
				for (size_t y = 0; y < 32; y++)
				{
					int pos = x + y * 32;

					for (size_t sx = 0; sx < 2; sx++)
					{
						for (size_t sy = 0; sy < 2; sy++)
						{
							img.setPixel(x * ScaleX + sx + 32 * ScaleX*((curImg*2) % ImagesX), y * ScaleY + sy + 32 * ScaleY * (((curImg*2) % ImagesCount) / ImagesX), sf::Color(imgData[x + y * 32] * 255, imgData[x + y * 32] * 255, imgData[x + y * 32] * 255, 255));
						}
					}
				}
			}

			Kernal(imgData, imgData_out);

			for (size_t x = 0; x < 32; x++)
			{
				for (size_t y = 0; y < 32; y++)
				{
					int pos = x + y * 32;

					for (size_t sx = 0; sx < ScaleX; sx++)
					{
						for (size_t sy = 0; sy < ScaleY; sy++)
						{
							if (x == 0 || x == 31 || y == 0 || y == 31)
								img.setPixel(x * ScaleX + sx + 32 * ScaleX*((curImg*2 + 1) % ImagesX), y * ScaleY + sy + 32 * ScaleY * (((curImg*2 + 1) % ImagesCount) / ImagesX), sf::Color(255, 255, 255, 255));
							else
								img.setPixel(x * ScaleX + sx + 32 * ScaleX*((curImg*2 + 1) % ImagesX), y * ScaleY + sy + 32 * ScaleY * (((curImg*2 + 1) % ImagesCount) / ImagesX), sf::Color(imgData_out[(x - 1) + (y - 1) * 30] * 255, imgData_out[(x - 1) + (y - 1) * 30 + 0] * 255, imgData_out[(x - 1) + (y - 1) * 30 + 0] * 255, 255));
						}
					}
				}
			}

			texture.loadFromImage(img);
			sprite.setTexture(texture);

			curImg++;
		}

		window.draw(sprite);



		window.display();
	}

	return 0;
}