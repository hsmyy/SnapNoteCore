///**************************************************************
// * Binarization with several methods
// * (0) Niblacks method
// * (1) Sauvola & Co.
// *     ICDAR 1997, pp 147-152
// * (2) by myself - Christian Wolf
// *     Research notebook 19.4.2001, page 129
// * (3) by myself - Christian Wolf
// *     20.4.2007
// *
// * See also:
// * Research notebook 24.4.2001, page 132 (Calculation of s)
// **************************************************************/
//
//#include <iostream>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <tesseract/baseapi.h>
//#include <tesseract/strngs.h>
//#include <dirent.h>
//#include "binarize.h"
////#include "../utils/FileUtil.h"
//#include "../utils/OCRUtil.h"
//
//using namespace std;
//using namespace cv;
//using namespace tesseract;
//
//
//int main_binarize() {
//	int winx = 19;
//	int winy = 19;
//	double optK = 0.5;
//	string lang = "eng";
//
//	ofstream out;
//	string text;
//
//	string imgDir[] = {"input", "niblack", "sauvola", "wolfjolion", "adaptive"};
//	string textDir[] = {"origOut", "niblackOut", "sauvolaOut", "wolfjolionOut", "adaptiveOut"};
//
//	for(int i = 0; i < 5; i++)
//	{
//		if(imgDir[i] != "input")
//			FileUtil::rmFile(imgDir[i] + "/*");
//		FileUtil::rmFile(textDir[i] + "/*");
//	}
//
//	vector<string> files = FileUtil::getAllFiles(imgDir[0]);
//	for (unsigned int i = 0; i < files.size(); i++) {
//		Mat src = imread(imgDir[0] + "/" + files[i], IMREAD_GRAYSCALE);
//		Mat dst[5];
//		for(int j = 1; j < 5; j++)
//		{
//			dst[j].create(src.size(), src.type());
//		}
//
//		Binarize::NiblackSauvolaWolfJolion(src, dst[1], NIBLACK, winx, winy, optK, 128);
//		Binarize::NiblackSauvolaWolfJolion(src, dst[2], SAUVOLA, winx, winy, optK, 128);
//		Binarize::NiblackSauvolaWolfJolion(src, dst[3], WOLFJOLION, winx, winy, optK,
//				128);
//		adaptiveThreshold(src, dst[4], 255, ADAPTIVE_THRESH_GAUSSIAN_C,
//				THRESH_BINARY, 19, 0);
//
//		dst[0] = src;
//		for(int j = 0; j < 5; j++)
//		{
//			cout<<"write File: " << imgDir[j] + "/" + files[i]<<endl;
//			if(imgDir[j] != "input")
//				imwrite(imgDir[j] + "/" + files[i], dst[j]);
//			cout<<"ocr File: " << textDir[j] +"/" + FileUtil::getFileNameNoSuffix(files[i]) + ".txt"<<endl;
//			FileUtil::writeToFile(OCRUtil::ocrFile(dst[j], lang), textDir[j] +"/" + FileUtil::getFileNameNoSuffix(files[i]) + ".txt");
//		}
//
//
//	}
//}
