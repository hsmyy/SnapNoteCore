/*
 * deskew.h
 *
 *  Created on: Jan 13, 2015
 *      Author: xxy
 */

#ifndef PREPROCESSING_SRC_DESKEW_H_
#define PREPROCESSING_SRC_DESKEW_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "../utils/FileUtil.h"

using namespace std;
using namespace cv;

class Deskew {
public:
	static void deskew(Mat& src, Mat& dst) {
		CV_Assert(src.channels() == 1);
		Mat grad_x, abs_grad_x, grad_y, abs_grad_y, grad;

		int ddepth = 3;

		Sobel(src, grad_x, ddepth, 1, 0);
		convertScaleAbs(grad_x, abs_grad_x);

		Sobel(src, grad_y, ddepth, 0, 1);
		convertScaleAbs(grad_y, abs_grad_y);

		addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, grad);
		threshold(grad, grad, 40.0, 255, THRESH_TOZERO);

		vector<cv::Vec4i> lines;
		HoughLinesP(grad, lines, 1, CV_PI / 180, 100, 70, 20);

		int vsize = 0;
		vector<double> tpK(1000);
		vector<int> tpN(1000);
		vector<vector<int> > tpV(1000);

		int vertN = 0;

		for (unsigned int i = 0; i < lines.size(); i++) {
			Vec4i v = lines[i];
			if (v[0] >= v[2] - 2 && v[0] <= v[2] + 2)
				vertN++;
			else {
				double k = (0.0 + v[3] - v[1]) / (0.0 + v[2] - v[0]);
				bool found = false;
				for (int j = 0; j < vsize; j++) {
					if (fabs(k - tpK[j]) < 0.18) {
						found = true;
						tpK[j] = (tpK[j] * tpN[j] + k) / (tpN[j] + 1);
						tpN[j] = tpN[j] + 1;
						tpV[j].push_back(i);
					}
				}
				if (!found) {
					tpK[vsize] = k;
					tpN[vsize] = 1;
					vector<int> v(1000);
					v.push_back(i);
					tpV.push_back(v);
					vsize++;
				}
			}
		}
		int maxTp = -1;
		int maxCt = 0;
		for (int i = 0; i < vsize; i++) {
			if (tpN[i] > maxCt) {
				maxCt = tpN[i];
				maxTp = i;
			}
		}

		Point center = Point(src.cols / 2, src.rows / 2);
		double angle = 0;
		Mat rot_mat(2, 3, CV_32FC1);
		if (maxTp >= 0) {
			angle = atan(tpK[maxTp]);
			cout << angle << endl;
		}
		angle = angle * 180 / M_PI;

		rot_mat = getRotationMatrix2D(center, angle, 1.0);
		warpAffine(src, dst, rot_mat, dst.size());
	}

	static void deskewDir(const char* inputDir, const char* outputDir) {
		vector<string> files = FileUtil::getAllFiles(inputDir);
		for (unsigned int i = 0; i < files.size(); i++) {
			cout << "report file :" << files[i] << endl;
			string inputPath = string(inputDir) + "/" + files[i];
			Mat src = imread(inputPath, IMREAD_GRAYSCALE);
			Mat dst(src.rows, src.cols, CV_8U);
			deskew(src, dst);
			string outputPath = string(outputDir) + "/" + files[i];

			imwrite(outputPath, dst);
		}
	}
	static void deskewSet(vector<Mat>& srcs, vector<Mat>& dsts)
	{
		CV_Assert(srcs.size() == dsts.size());
		for(unsigned int i = 0; i < srcs.size(); i++)
		{
			deskew(srcs[i], dsts[i]);
		}
	}

};
#endif /* PREPROCESSING_SRC_DESKEW_H_ */
