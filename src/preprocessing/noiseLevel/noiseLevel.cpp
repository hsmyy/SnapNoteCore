//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//
//#include "../utils/OCRUtil.h"
//#include "../binarize/binarize.h"
////#include "noiseLevel.h"
//#include "../utils/FileUtil.h"
//
//using namespace std;
//using namespace cv;
//
//int main_noiseLevel(int argc, char *argv[]) {
//
////	Mat img = imread("images/saltPepper/test2.jpg");
////	cout<<"avg : " << getAvgNoiseLevel(img)<<endl;
////	cout<<"max : "<< getMaxNoiseLevel(img)<<endl;
//
//	string lang = "eng";
//	string srcDir = "images/saltPepper";
//
//	string binarizeDir = "images/binarize";
//
//	Binarize::binarizeDir(srcDir, binarizeDir);
//
////	string dstDir[] =
////			{ "images/denoise1", "images/denoise2", "images/denoise3" };
////	String textDir[] = { "text/denoise1", "text/denoise2", "text/denoise3"};
////	for (int i = 0; i < 3; i++) {
////		FileUtil::rmFile(dstDir[i] + "/*");
////		FileUtil::rmFile(textDir[i] + "/*");
////	}
////	Denoise::saltPepperDenoiseDir(binarizeDir, dstDir[0], 3);
////	Denoise::saltPepperDenoiseDir(binarizeDir, dstDir[1], 5);
////	Denoise::saltPepperDenoiseDir(binarizeDir, dstDir[2], 7);
////	OCRUtil::ocrDir(dstDir[0], textDir[0], lang);
////	OCRUtil::ocrDir(dstDir[1], textDir[1], lang);
////	OCRUtil::ocrDir(dstDir[2], textDir[2], lang);
//
//}
