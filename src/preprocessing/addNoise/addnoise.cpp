#include <opencv2/opencv.hpp>
//#include <opencv/cv.hpp>
#include <iostream>
#include "addnoise.h"
#include "../utils/FileUtil.h"

using namespace std;
using namespace cv;


int main_addNoise() {
	RNG rng(0xFFFFFFFF);
	string srcDir = "input";
	string dstDir[] = {"background", "gaussian", "saltPepper"};
	//string backgroundNoiseDir = "backgroundNoise";

	vector<string> files = FileUtil::getAllFiles(srcDir);


	for(unsigned int i = 0; i < 3; i++)
	{
		FileUtil::rmFile(dstDir[i] + "/*");
	}

	for (unsigned int i = 0; i < files.size(); i++) {
		string filepath = srcDir + "/" + files[i];
		cout << files[i] << endl;
		Mat src = imread(filepath, IMREAD_COLOR);
		Mat dst[3];
		//AddNoise::addRandomBackground(src, dst[0], rng, 10, 0.3, 0.7);

		cvtColor(src, src, COLOR_BGR2GRAY);
		threshold(src, src, 128, 255, THRESH_BINARY);
		AddNoise::addGaussianNoise(src, dst[1], 0.59, 0.41, 128, 100);
//		Mat roi(dst[1](Rect(300, 300, 500, 500)));
//		cout<<roi<<endl;
		//AddNoise::addSaltPepperNoise(src, dst[2], 10, 245);

		for(int j = 0; j < 3; j++)
		{
			string outputPath = dstDir[j] + "/" + FileUtil::getFileNameNoSuffix(files[i]) + ".jpg";
			cout<<outputPath<<endl;
			imwrite(outputPath, dst[j]);
		}

	}

}

