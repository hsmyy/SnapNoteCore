/*
 * denoise.h
 *
 *  Created on: Jan 28, 2015
 *      Author: xxy
 */

#ifndef SRC_DENOISE_H_
#define SRC_DENOISE_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "../utils/FileUtil.h"
using namespace std;
using namespace cv;

class Denoise {
public:
	static Mat noiseReduction(Mat& img, Mat& dst, int block = 3) //block=5 is our threshold for blob size. Less than that is noise
			{
		CV_Assert(img.channels() == 1);
		dst.create(img.rows, img.cols, img.type());
		int i, j, m, n, r = img.rows, c = img.cols, white, black;

		for (i = 0; i < r - block; i += block) {
			for (j = 0; j < c - block; j += block) {

				white = black = 0;
				//to check for large enough blob.
				for (m = i; m < i + block; m++) {
					for (n = j; n < j + block; n++) {

						if (img.at<uchar>(m, n) == 0)
							black++;
						else
							white++;
					}
				}
				//to remove the identified noise from the image
				int val;
				if (black >= white)
					val = 0;
				else
					val = 255;

				for (m = i; m < i + block; m++) {
					for (n = j; n < j + block; n++) {
						dst.at<uchar>(m, n) = val;
					}
				}
			}
		}

		return dst;
	}

	static void GaussianDenoise(Mat& src, Mat& dst, int kernelSize = 3) {
		GaussianBlur(src, dst, Size(kernelSize, kernelSize), 0);
	}

	static void GaussianDenoiseDir(string srcDir, string dstDir, int kernelSize = 3) {
		vector<string> files = FileUtil::getAllFiles(srcDir);
		for (unsigned int j = 0; j < files.size(); j++) {
			cout << srcDir + "/" + files[j] << endl;
			Mat src = imread(srcDir + "/" + files[j], IMREAD_GRAYSCALE);

			//threshold(src, src, 128, 255, THRESH_BINARY);
			Mat dst;
			GaussianDenoise(src, dst, kernelSize);
			imwrite(dstDir + "/" + files[j], dst);
		}
	}

	static void saltPepperDenoise(Mat& src, Mat& dst, int kernelSize = 3) {
		//cvtColor(src, src, COLOR_BGR2GRAY);
		CV_Assert(src.channels() == 1);
		Mat bin, spclean;
		threshold(src, bin, 128, 255, THRESH_BINARY);
		noiseReduction(bin, spclean, kernelSize);
		GaussianBlur(spclean, dst, Size(kernelSize, kernelSize), 0);
	}

	static void saltPepperDenoiseDir(string srcDir, string dstDir, int kernelSize = 3) {
		vector<string> files = FileUtil::getAllFiles(srcDir);
		for (unsigned int j = 0; j < files.size(); j++) {
			//cout << srcDir + "/" + files[j] << endl;
			Mat src = imread(srcDir + "/" + files[j], IMREAD_GRAYSCALE);
			Mat dst;
			saltPepperDenoise(src, dst, kernelSize);
			imwrite(dstDir + "/" + files[j], dst);
		}
	}
	static void denoise(Mat& src, Mat& dst) {
		saltPepperDenoise(src, dst);
	}
};

#endif /* SRC_DENOISE_H_ */
