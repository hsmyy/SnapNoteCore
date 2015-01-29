/*
 * AddNoise.h
 *
 *  Created on: Jan 26, 2015
 *      Author: xxy
 */

#ifndef SRC_ADDNOISE_H_
#define SRC_ADDNOISE_H_

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

class AddNoise {
public:
	static Scalar randomColor(RNG& rng) {
		int icolor = (unsigned) rng;
		return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
	}
	//alpha: original image weight
	//beta: background weight

	static void Drawing_Random_Filled_Polygons(Mat image, RNG& rng,
			const int img_width, const int img_height) {
		int lineType = 8;
		int NUMBER = 10;

		int x_1 = -img_width / 2;
		int x_2 = img_width * 3 / 2;
		int y_1 = -img_height / 2;
		int y_2 = img_height * 3 / 2;

		for (int i = 0; i < NUMBER; i++) {
			Point pt[2][3];
			pt[0][0].x = rng.uniform(x_1, x_2);
			pt[0][0].y = rng.uniform(y_1, y_2);
			pt[0][1].x = rng.uniform(x_1, x_2);
			pt[0][1].y = rng.uniform(y_1, y_2);
			pt[0][2].x = rng.uniform(x_1, x_2);
			pt[0][2].y = rng.uniform(y_1, y_2);
			pt[1][0].x = rng.uniform(x_1, x_2);
			pt[1][0].y = rng.uniform(y_1, y_2);
			pt[1][1].x = rng.uniform(x_1, x_2);
			pt[1][1].y = rng.uniform(y_1, y_2);
			pt[1][2].x = rng.uniform(x_1, x_2);
			pt[1][2].y = rng.uniform(y_1, y_2);

			const Point* ppt[2] = { pt[0], pt[1] };
			int npt[] = { 3, 3 };

			fillPoly(image, ppt, npt, 2, randomColor(rng), lineType);
		}
	}

	static Mat addRandomBackground(Mat& src, Mat& dst, RNG& rng, int count,
			double alpha, double beta) {
		Mat mask(src.size(), src.type());
		Drawing_Random_Filled_Polygons(mask, rng, src.cols, src.rows);
		addWeighted(src, alpha, mask, beta, 0, dst);
		return dst;
	}
	static Mat addGaussianNoise(Mat& src, Mat& dst, double alpha, double beta,
			int mean, int stddev) {
		CV_Assert(src.type() == CV_8UC1);
		Mat noise(src.size(), src.type());
		randn(noise, mean, stddev);
		addWeighted(src, alpha, noise, beta, 0, dst);
		return dst;
	}
	static Mat addSaltPepperNoise(Mat&src, Mat& dst, int low_threshold,
			int high_threshold) {
		CV_Assert(src.type() == CV_8UC1);
		Mat saltpepper_noise = Mat::zeros(src.rows, src.cols, src.type());

		randu(saltpepper_noise, 0, 255);

		Mat black = saltpepper_noise < 10;
		Mat white = saltpepper_noise > 245;

		dst = src.clone();
		dst.setTo(255, white);
		dst.setTo(0, black);
		return dst;
	}
	static Mat rotateRandomAngle(Mat& src, Mat& dst, RNG& rng) {
		//dst.create(src.size(), src.type());
		Point center = Point(src.cols / 2, src.rows / 2);
		double angle = rng.uniform(-45, 45);
		Mat rot_mat;
		//angle = angle * 180 / M_PI;

		rot_mat = getRotationMatrix2D(center, angle, 1.0);
		warpAffine(src, dst, rot_mat, dst.size());
		return dst;
	}
	static void rotateRangeAngle(Mat& src, string outputDir, string prefix, int lower_angle = -45,
			int upper_angle = 45, int step = 1) {
		CV_Assert(lower_angle <= upper_angle && step > 0);
		Point center = Point(src.cols / 2, src.rows / 2);
		for (int angle = lower_angle; angle <= upper_angle; angle += step) {
			Mat rot_mat, dst;
			rot_mat = getRotationMatrix2D(center, angle, 1.0);
			warpAffine(src, dst, rot_mat, src.size(), 1, 0, Scalar(255));
			char filename[100];
			sprintf(filename, "%d", angle);
			string outputPath = outputDir + "/" + prefix + filename + ".jpg";
			//cout<<outputPath<<endl;
			imwrite(outputPath, dst);
		}


	}
};

#endif /* SRC_ADDNOISE_H_ */
