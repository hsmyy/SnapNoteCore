#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "denoise.h"
#include "../utils/OCRUtil.h"

using namespace std;
using namespace cv;


int main_denoise(int argc, char *argv[]) {
	//Mat img(100, 100, CV_8UC1);
//	string gaussianOrig = "gaussian";
//	string saltPepperOrig = " saltPepper";
	string lang = "eng";
	string srcDir[] = { "gaussian", "saltPepper" };
	string dstDir[] = { "gaussianDenoise", "saltPepperDenoise" };
	string textDir[] = { "gaussianOrigText", "saltPepperOrigText", "gaussianDenoiseText", "saltPepperDenoiseText" };
	Denoise::GaussianDenoise(srcDir[0], dstDir[0]);
	Denoise::saltPepperDenoise(srcDir[1], dstDir[1]);
	OCRUtil::ocrDir(srcDir[0], textDir[0], lang);
	OCRUtil::ocrDir(srcDir[1], textDir[1], lang);
	OCRUtil::ocrDir(dstDir[0], textDir[2], lang);
	OCRUtil::ocrDir(dstDir[1], textDir[3], lang);

}
