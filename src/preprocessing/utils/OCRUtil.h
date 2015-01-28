/*
 * OCRUtil.h
 *
 *  Created on: Jan 26, 2015
 *      Author: xxy
 */

#ifndef GAUSSIAN_SP_DENOISE_SRC_OCRUTIL_H_
#define GAUSSIAN_SP_DENOISE_SRC_OCRUTIL_H_

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <iostream>
#include "FileUtil.h"

using namespace cv;
using namespace std;
using namespace tesseract;

class OCRUtil
{
public:
	static string ocrFile(Mat& src, const string lang) {

		TessBaseAPI tess;
		tess.Init(NULL, lang.c_str(), OEM_DEFAULT);
		tess.SetPageSegMode(PSM_SINGLE_BLOCK);

		tess.SetImage((uchar*) src.data, src.cols, src.rows, 1, src.cols);

		char* out = tess.GetUTF8Text();
		//cout << out << endl;
		return string(out);
	}
	static void ocrDir(string srcDir, string dstDir, string lang)
	{
		vector<string> files = FileUtil::getAllFiles(srcDir);
		for(unsigned int i = 0 ; i < files.size(); i++)
		{
			string inputPath = srcDir + "/" + files[i];
			cout<<inputPath<<endl;
			Mat src = imread(inputPath, IMREAD_GRAYSCALE);
			string text = ocrFile(src, lang);
			string outputPath = dstDir + "/" + FileUtil::getFileName(files[i]) + ".txt";
			cout<<outputPath<<endl;
			FileUtil::writeToFile(text, outputPath);
		}
	}
};


#endif /* GAUSSIAN_SP_DENOISE_SRC_OCRUTIL_H_ */
