/**
	kernel.cu
	Purpose: main test program for the ConvNet

	@author Tobias Fast
	@version 1.0 9/12/18
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <SFML/Graphics.hpp>
#include <fstream>

#include "ConvNet.hpp"

#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "ConvLayer.hpp"
#include "MaxPoolLayer.hpp"
#include "NNLayer.hpp"

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

	ConvNet net(32, 32, 1);
	//net.AddLayer(new ConvLayer(net.GetLastLayer(), 1, 3));
	//net.AddLayer(new ConvLayer(net.GetLastLayer(), 1, 3));
	//net.AddLayer(new MaxPoolLayer(net.GetLastLayer(), 2));
	net.AddLayer(new NNLayer(net.GetLastLayer(), 50));
	net.CreateNetwork();
	if (!net.Feed(imgData))
		return;


	return 0;

	int ImagesX = 4;
	int ImagesY = 5;
	int ImagesCount = ImagesX * ImagesY;

	int ScaleX = 2;
	int ScaleY = 2;

	sf::Image img;
	sf::Texture texture;
	sf::Sprite sprite;

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

		window.clear();
		window.draw(sprite);
		window.display();
	}

	return 0;
}