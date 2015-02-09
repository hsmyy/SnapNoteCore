/*
 * quantize.h
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#ifndef QUANTIZE_H_
#define QUANTIZE_H_


#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "main.h"

using namespace cv;
using namespace std;

class Quantizer{
public:
	/**
	 * quantize the color of image.
	 * Mat &img3f:
	 */
	int Quantize(Mat& img3f, Mat &idx1i, Mat &colorInfos3f, Mat &colorCount1i, double ratio = 0.95);
private:
	int getMaxColorNum(vector<pair<int, int> > &num, int rows, int cols, double ratio);
private:
	int clrNums[3] = {12,12,12};
};

int Quantizer::getMaxColorNum(vector<pair<int, int> > &num, int rows, int cols, double ratio){
	int maxNum = (int)num.size();
	int maxDropNum = cvRound(rows * cols * (1 - ratio));
	//ignore the small part
	for(int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; --maxNum){
		crnt += num[maxNum - 2].first;
	}
	// ignore number should larger than 10 and smaller than 256
	maxNum = min(maxNum, 256);
	if(maxNum <= 10){
		maxNum = min(10, (int)num.size());
	}
	return maxNum;
}

int Quantizer::Quantize(Mat& img3f, Mat &idx1i, Mat &colorInfos3f, Mat &colorCount1i, double ratio){

	float clrTmp[3] = {clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f};
	int w[3] = {clrNums[1] * clrNums[2], clrNums[2], 1};

	CV_Assert(img3f.data != NULL);
	idx1i = Mat::zeros(img3f.size(), 4);
	int rows = img3f.rows, cols = img3f.cols;
	if(img3f.isContinuous() && idx1i.isContinuous()){
		cols *= rows;
		rows = 1;
	}

	//store color into map
	map<int, int> pallet;
	//idx1i is used to store the hash key of each pixel
	//pallet store the <hashkey, num> pair
	for(int y = 0; y < rows; ++y){
		//TODO change from float to uchar
		const float * imgData = img3f.ptr<float>(y);
		int *idx = idx1i.ptr<int>(y);
		for(int x = 0; x < cols; ++x, imgData += 3){
			idx[x] = (int)(imgData[0] * clrTmp[0]) * w[0] + (int)(imgData[1] * clrTmp[1]) * w[1] +
					(int)(imgData[2] * clrTmp[2]);
			++pallet[idx[x]];
		}
	}
	int maxNum = 0;
	{
		vector<pair<int, int> > num;
		num.reserve(pallet.size());
		//reverse the pair to <num, hashkey>
		for(map<int, int>::iterator it = pallet.begin(); it != pallet.end(); ++it){
			num.push_back(pair<int, int>(it->second, it->first));
		}
		sort(num.begin(), num.end(), std::greater<pair<int, int> >());
		maxNum = getMaxColorNum(num, rows, cols, ratio);
		pallet.clear();
		// pallet stores <hashkey, idx> and idx is in the desc order.
		// the less the idx, the higher the number.
		for(int i = 0; i < maxNum; ++i){
			pallet[num[i].second] = i;
		}

		// restore the color
		vector<Vec3i> color3i(num.size());
		for(unsigned int i = 0; i < num.size(); ++i){
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		//cal distance
		// pallet store <hashkey, hashkey> second is the smallest distance between first
		for(unsigned int i = maxNum; i < num.size(); ++i){
			int simIdx = 0, simVal = INT_MAX;
			//find min distance vec
			for(int j = 0; j < maxNum; ++j){
				int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
				if(d_ij < simVal){
					simVal = d_ij, simIdx = j;
				}
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	colorInfos3f = Mat::zeros(1, maxNum, CV_32FC3);
	colorCount1i = Mat::zeros(colorInfos3f.size(), 4);

	Vec3f *color = (Vec3f *)(colorInfos3f.data);
	int *colorNum = (int *)(colorCount1i.data);
	for(int y = 0; y < rows; ++y){
		//origin image data
		const Vec3f *imgData = img3f.ptr<Vec3f>(y);
		//hashkey data
		int *idx = idx1i.ptr<int>(y);
		for(int x = 0; x < cols; ++x){
			//set idx1i as the nearest hashkey
			idx[x] = pallet[idx[x]];
			//??
			color[idx[x]] += imgData[x];
			colorNum[idx[x]]++;
		}
	}
	for(int i = 0; i < colorInfos3f.cols; ++i){
		color[i] /= (float)colorNum[i];
	}
	return colorInfos3f.cols;
}


#endif /* QUANTIZE_H_ */
