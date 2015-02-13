/*
 * rc.h
 *
 *  Created on: Jan 21, 2015
 *      Author: fc
 */

#ifndef RC_H_
#define RC_H_

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <cmath>

#include "quantize.h"

class RegionContrastSalient{
public:
	RegionContrastSalient(double sigmaDist = 0.4, float regionWeight = 2, float distanceWeight = 0.01, bool debug = false);
	void BuildRegions(Mat& regionIdxImage, vector<Region> &regionInfos, Mat &colorIdxImage, int colorNum);
	void RegionContrast(const vector<Region> &regionInfos, Mat &colorInfos, Mat& regionSalientScore, int pixelNum, float theta, Mat_<float> &cDistCache1f);
	void RegionContrast2(const vector<Region> &regionInfos, Mat &colorInfos, Mat& regionSalientScore, int pixelNum, float theta, Mat_<float> &cDistCache1f);
	Mat GetBorderReg(Mat &regionIdxImage, int regNum, double ratio, double thr);
	void SmoothSaliency(Mat &colorCount1i, Mat &colorSalientScore1d, float delta, const vector<vector<CostfIdx> > &similiarMatrix);
	void SmoothByHist(Mat &originImage3f, Mat &salientScoreImage1f, float delta);
	void SmoothByRegion(Mat &sal1f, Mat &segIdx1i, int regNum, bool bNormalize = true);

	Mat getRC(Mat &img3f, Mat &regionIdxImage1i, int regNum, double sigmaDist, bool debug= false);
	Mat equalize(Mat &img);

	Mat centerRC(
			vector<Region> region,
			Mat &regionColor,
			Mat &regionScore,
			double sigmaDist,
			Mat &originImage,
			Mat regionIdxImage,
			Mat_<float> &cDistCache1f,
			bool debug= false
			);

	Mat originRC(
			vector<Region> regions,
			Mat &regionColor,
			Mat &regionScore,
			double sigmaDist,
			Mat &originImage3f,
			Mat regionIdxImage,
			Mat_<float> &cDistCache1f,
			bool debug= false
			);

	Mat hardHC(vector<Region> regions,
			Mat &regionColor,
			Mat &regionScore,
			double sigmaDist,
			Mat &originImage,
			Mat regionIdxImage, bool debug= false
			);


private:
	double _sigmaDist;

	float _regionWeight;
	float _distanceWeight;
	bool _debug;
	Quantizer quantizer;

public:
	float getRegionWeight(){return _regionWeight;}
	float getDistanceWeight(){return _distanceWeight;}
	double getSigmaDist(){return _sigmaDist;}
	void updateRegionWeight(float updated){
		_regionWeight += updated;
	}
	void updateDistanceWeight(float updated){
		_distanceWeight += updated;
	}
};

RegionContrastSalient::RegionContrastSalient(double sigmaDist, float regionWeight, float distanceWeight, bool debug)
: _sigmaDist(sigmaDist), _regionWeight(regionWeight), _distanceWeight(distanceWeight), _debug(debug){

}

void RegionContrastSalient::BuildRegions(Mat& regionIdxImage, vector<Region> &regionInfos, Mat &colorIdxImage, int colorNum)
{
	int rows = regionIdxImage.rows, cols = regionIdxImage.cols, regNum = (int)regionInfos.size();
	//center
	double cx = cols/2.0, cy = rows / 2.0;
	// build region color frequency
	Mat_<int> regColorFre1i = Mat_<int>::zeros(regNum, colorNum);
	// accumulate pixel number, centroid and distance from center
	for (int y = 0; y < rows; y++){
		const int *regIdx = regionIdxImage.ptr<int>(y);
		const int *colorIdx = colorIdxImage.ptr<int>(y);
		for (int x = 0; x < cols; x++, regIdx++, colorIdx++){
			Region &reg = regionInfos[*regIdx];
			reg.pixNum ++;
			reg.centroid.x += x;
			reg.centroid.y += y;
			regColorFre1i(*regIdx, *colorIdx)++;
			reg.ad2c += Point2d(abs(x - cx), abs(y - cy));
		}
	}
	// all the information has normalized to [0,1)
	for (int i = 0; i < regNum; i++){
		Region &reg = regionInfos[i];
		reg.centroid.x /= reg.pixNum * cols;
		reg.centroid.y /= reg.pixNum * rows;
		reg.ad2c.x /= reg.pixNum * cols;
		reg.ad2c.y /= reg.pixNum * rows;
		int *regColorFre = regColorFre1i.ptr<int>(i);
		for (int j = 0; j < colorNum; j++){
			float fre = (float)regColorFre[j]/(float)reg.pixNum;
			if (regColorFre[j] > EPS)
				reg.freIdx.push_back(make_pair(fre, j));
		}
	}
}

bool region_cmp(const pair<int, double> &firstp, const pair<int, double> &secondp){
	return firstp.second > secondp.second;
}

Mat_<float> pairwiseColorDist(Mat &colorInfos){
	Mat_<float> cDistCache1f = Mat::zeros(colorInfos.cols, colorInfos.cols, CV_32F);
	Vec3f* pColor = (Vec3f*)colorInfos.data;
	for(int i = 0; i < cDistCache1f.rows; i++)
		for(int j= i+1; j < cDistCache1f.cols; j++)
			cDistCache1f[i][j] = cDistCache1f[j][i] = vecDist<float, 3>(pColor[i], pColor[j]);
	return cDistCache1f;
}

void RegionContrastSalient::RegionContrast2(const vector<Region> &regionInfos, Mat &colorInfos, Mat& regionSalientScore, int pixelNum, float theta, Mat_<float> &cDistCache1f){
	int i;
	int regNum = (int)regionInfos.size();
//	Mat_<float> cDistCache1f = pairwiseColorDist(colorInfos);

	Mat_<double> rDistCache1d = Mat::zeros(regNum, regNum, CV_64F);
	regionSalientScore = Mat::zeros(1, regNum, CV_64F);
	double* regSal = (double*)regionSalientScore.data;
	for (i = 0; i < regNum; i++){
		const Point2d &rc = regionInfos[i].centroid;
		for (int j = 0; j < regNum; j++){
			if(i<j) {
				double dd = 0;
				const vector<CostfIdx> &c1 = regionInfos[i].freIdx, &c2 = regionInfos[j].freIdx;
				for (size_t m = 0; m < c1.size(); m++){
					for (size_t n = 0; n < c2.size(); n++){
						dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
					}
				}
				rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(rc, regionInfos[j].centroid)/_sigmaDist);
			}
			regSal[i] += regionInfos[j].pixNum * rDistCache1d[i][j];
		}
		regSal[i] *= exp(_regionWeight * sqrt(regionInfos[i].pixNum / (float)pixelNum) - _distanceWeight * (sqr(regionInfos[i].ad2c.x) + sqr(regionInfos[i].ad2c.y)));
	}
}

void RegionContrastSalient::RegionContrast(
		const vector<Region> &regionInfos, Mat &colorInfos, Mat& regionSalientScore, int pixelNum, float theta,
		Mat_<float> &cDistCache1f)
{
	int i,len;
	int regNum = (int)regionInfos.size();
	if(_debug){
		//cout << "when theta=" << theta << endl;
		int mu = 0,sigma = 0;
		for(i = 0; i < regNum; ++i){
			mu += regionInfos[i].pixNum;
		}
		mu /= regNum;
		for(i = 0; i < regNum; ++i){
			sigma = sqr(mu - regionInfos[i].pixNum);
		}
		//cout << (mu / (float)pixelNum) << "," << (sqrt(sigma) / (float)pixelNum) << "," << regNum << endl;
	}
	//calculate the distance between any pair of color set.


	Mat_<double> rDistCache1d = Mat::zeros(regNum, regNum, CV_64F);
	regionSalientScore = Mat::zeros(1, regNum, CV_64F);
	double* regSal = (double*)regionSalientScore.data;
	for (i = 0; i < regNum; i++){
		const Point2d &rc = regionInfos[i].centroid;
		for (int j = 0; j < regNum; j++){
			if(i<j) {
				double dd = 0;
				// probability of each kind of color pixel
				const vector<CostfIdx> &c1 = regionInfos[i].freIdx, &c2 = regionInfos[j].freIdx;
				for (size_t m = 0; m < c1.size(); m++){
					for (size_t n = 0; n < c2.size(); n++){
						// color distance * frequency
						dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
					}
				}
				// Dr(rk,ri) * exp(Ds(rk,ri)/-sigmas^2)
				rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(rc, regionInfos[j].centroid)/_sigmaDist);
			}
			//then * w(ri)
			regSal[i] += regionInfos[j].pixNum * rDistCache1d[i][j];
		}
		// then * exp(-9dk^2)
		if(theta > 0){
			regSal[i] *= exp(-15.0 * (sqr(regionInfos[i].ad2c.x) + sqr(regionInfos[i].ad2c.y)));
		}else{
			regSal[i] *= exp(-12.0 * (sqr(regionInfos[i].ad2c.x) + sqr(regionInfos[i].ad2c.y)));
		}
//		regSal[i] /= regs[i].pixNum * sqrt(sqr(0.5 - regs[i].centroid.y) + sqr(0.5 - regs[i].centroid.x)) / 150;
//		regSal[i] *= sqrt(exp(regs[i].pixNum / pixelNum - sqrt(sqr(0.5 - regs[i].centroid.y) + sqr(0.5 - regs[i].centroid.x))) / 100 );

		//should be reserved
		if(theta > 0){
			double prior = exp( sqrt(regionInfos[i].pixNum * 2 / (float)pixelNum) -
					sqrt(sqr(regionInfos[i].ad2c.x) + sqr(regionInfos[i].ad2c.y))
					/ theta );
			regSal[i] *= prior;
			if(_debug){
				cout << "region " << i << "[" << regionInfos[i].pixNum << "]:" << prior << endl;
			}
		}
	}

	if(_debug){
		vector<pair<int, double> > debugRegions;
		len = (int)regionInfos.size();
		for(i = 0; i < len; ++i){
			debugRegions.push_back(make_pair(i, regSal[i]));
		}
		sort(debugRegions.begin(), debugRegions.end(), region_cmp);
		for(i = 0, len = debugRegions.size(); i < len; ++i){
			int idx = debugRegions[i].first;
			cout << "region " << idx << "[" << regionInfos[idx].centroid <<"]["<< regionInfos[idx].pixNum << "]: " << regSal[idx] << endl;
		}
	}
}

Mat RegionContrastSalient::GetBorderReg(Mat &regionIdxImage, int regNum, double ratio, double thr)
{
	// Variance of x and y
	vecD vX(regNum), vY(regNum);
	int w = regionIdxImage.cols, h = regionIdxImage.rows;
	{
		vecD mX(regNum), mY(regNum), n(regNum); // Mean value of x and y, pixel number of region
		// collect the x,y coordinate and number of each region
		for (int y = 0; y < regionIdxImage.rows; y++){
			const int *idx = regionIdxImage.ptr<int>(y);
			for (int x = 0; x < regionIdxImage.cols; x++, idx++)
				mX[*idx] += x, mY[*idx] += y, n[*idx]++;
		}
		// calculate average x,y
		for (int i = 0; i < regNum; i++)
			mX[i] /= n[i], mY[i] /= n[i];
		// cal abs of variance from each axis
		for (int y = 0; y < regionIdxImage.rows; y++){
			const int *idx = regionIdxImage.ptr<int>(y);
			for (int x = 0; x < regionIdxImage.cols; x++, idx++)
				vX[*idx] += abs(x - mX[*idx]), vY[*idx] += abs(y - mY[*idx]);
		}

		for (int i = 0; i < regNum; i++){
			vX[i] = vX[i]/n[i] + EPS, vY[i] = vY[i]/n[i] + EPS;
		}
	}

	// Number of border pixels in x and y border region
	vecI xbNum(regNum), ybNum(regNum);
	int wGap = cvRound(w * ratio), hGap = cvRound(h * ratio);
	vector<Point> bPnts;
	{
		ForPoints2(pnt, 0, 0, w, hGap) // Top region
			ybNum[regionIdxImage.at<int>(pnt)]++, bPnts.push_back(pnt);
		ForPoints2(pnt, 0, h - hGap, w, h) // Bottom region
			ybNum[regionIdxImage.at<int>(pnt)]++, bPnts.push_back(pnt);
		ForPoints2(pnt, 0, 0, wGap, h) // Left region
			xbNum[regionIdxImage.at<int>(pnt)]++, bPnts.push_back(pnt);
		ForPoints2(pnt, w - wGap, 0, w, h)
			xbNum[regionIdxImage.at<int>(pnt)]++, bPnts.push_back(pnt);
	}

	Mat bReg1u(regionIdxImage.size(), CV_8U);{  // likelihood map of border region
		double xR = 1.0/(4*hGap), yR = 1.0/(4*wGap);
		vector<byte> regL(regNum); // likelihood of each region belongs to border background
		for (int i = 0; i < regNum; i++) {
			//the core formula of judging border
			double lk = xbNum[i] * xR / vY[i] + ybNum[i] * yR / vX[i];
			regL[i] = lk/thr > 1 ? 255 : 0; //saturate_cast<byte>(255 * lk / thr);
		}

		//put answer to the mask data.
		for (int r = 0; r < h; r++)	{
			const int *idx = regionIdxImage.ptr<int>(r);
			byte* maskData = bReg1u.ptr<byte>(r);
			for (int c = 0; c < w; c++, idx++)
				maskData[c] = regL[*idx];
		}
	}

	//still set the border as true
	for (size_t i = 0; i < bPnts.size(); i++)
		bReg1u.at<byte>(bPnts[i]) = 255;
	return bReg1u;
}

void RegionContrastSalient::SmoothSaliency(Mat &colorCount1i, Mat &colorSalientScore1d, float delta, const vector<vector<CostfIdx> > &similiarMatrix)
{
	if (colorSalientScore1d.cols < 2)
		return;
	CV_Assert(colorSalientScore1d.rows == 1 && colorSalientScore1d.type() == CV_32FC1);
	CV_Assert(colorCount1i.size() == colorSalientScore1d.size() && colorCount1i.type() == CV_32SC1);

	int binN = colorSalientScore1d.cols;
	Mat newColorSalientScore1d= Mat::zeros(1, binN, CV_64FC1);
	float *sal = (float*)(colorSalientScore1d.data);
	double *newSal = (double*)(newColorSalientScore1d.data);
	int *pW = (int*)(colorCount1i.data);

	// Distance based smooth
	int n = max(cvRound(binN * delta), 2);
	vecD dist(n, 0), val(n), w(n);
	for (int i = 0; i < binN; i++){
		const vector<CostfIdx> &similari = similiarMatrix[i];
		double totalDist = 0, totoalWeight = 0;
		for (int j = 0; j < n; j++){
			int ithIdx =similari[j].second;
			dist[j] = similari[j].first;
			val[j] = sal[ithIdx];
			w[j] = pW[ithIdx];
			totalDist += dist[j];
			totoalWeight += w[j];
		}
		double valCrnt = 0;
		for (int j = 0; j < n; j++)
			valCrnt += val[j] * (totalDist - dist[j]) * w[j];

		newSal[i] =  valCrnt / (totalDist * totoalWeight);
	}
	normalize(newColorSalientScore1d, colorSalientScore1d, 0, 1, NORM_MINMAX, CV_32FC1);
}

void RegionContrastSalient::SmoothByHist(Mat &originImage3f, Mat &salientScoreImage1f, float delta)
{
	// 1) Quantize colors
	CV_Assert(originImage3f.size() == salientScoreImage1f.size() &&
			originImage3f.type() == CV_32FC3 && salientScoreImage1f.type() == CV_32FC1);
	Mat colorIdxImage1i, colorInfos3f, colorCount1i;
	int binN = quantizer.Quantize(originImage3f, colorIdxImage1i, colorInfos3f, colorCount1i);

	// Get initial color saliency
	Mat colorSalientScore1d =  Mat::zeros(1, binN, CV_64FC1);
	int rows = originImage3f.rows, cols = originImage3f.cols;
	// 2) get the average of salient score by color
	{
		double* colorSal = (double*)colorSalientScore1d.data;
		if (originImage3f.isContinuous() && salientScoreImage1f.isContinuous())
			cols *= originImage3f.rows, rows = 1;

		for (int y = 0; y < rows; y++){
			const int* idx = colorIdxImage1i.ptr<int>(y);
			const float* initialS = salientScoreImage1f.ptr<float>(y);
			for (int x = 0; x < cols; x++)
				colorSal[idx[x]] += initialS[x];
		}
		const int *colorNum = (int*)(colorCount1i.data);
		for (int i = 0; i < binN; i++)
			colorSal[i] /= colorNum[i];
		normalize(colorSalientScore1d, colorSalientScore1d, 0, 1, NORM_MINMAX, CV_32F);
	}
	// 3) Find similar colors & Smooth saliency value for color bins
	vector<vector<CostfIdx> > similiarMatrix(binN); // Similar color: how similar and their index
	Vec3f* color = (Vec3f*)(colorInfos3f.data);
	cvtColor(colorInfos3f, colorInfos3f, CV_BGR2Lab);
	for (int i = 0; i < binN; i++){
		vector<CostfIdx> &similari = similiarMatrix[i];
		similari.push_back(make_pair(0.f, i));
		for (int j = 0; j < binN; j++)
			if (i != j)
				similari.push_back(make_pair(vecDist<float, 3>(color[i], color[j]), j));
		sort(similari.begin(), similari.end());
	}
	cvtColor(colorInfos3f, colorInfos3f, CV_Lab2BGR);
	// 4) smooth saliency
	SmoothSaliency(colorCount1i, colorSalientScore1d, delta, similiarMatrix);

	// 5) Reassign pixel saliency values
	float* colorSal = (float*)(colorSalientScore1d.data);
	for (int y = 0; y < rows; y++){
		const int* idx = colorIdxImage1i.ptr<int>(y);
		float* resSal = salientScoreImage1f.ptr<float>(y);
		for (int x = 0; x < cols; x++)
			resSal[x] = colorSal[idx[x]];
	}
}

void RegionContrastSalient::SmoothByRegion(Mat &sal1f, Mat &segIdx1i, int regNum, bool bNormalize)
{
	vecD saliecy(regNum, 0);
	vecI counter(regNum, 0);
	for (int y = 0; y < sal1f.rows; y++){
		const int *idx = segIdx1i.ptr<int>(y);
		float *sal = sal1f.ptr<float>(y);
		for (int x = 0; x < sal1f.cols; x++){
			saliecy[idx[x]] += sal[x];
			counter[idx[x]] ++;
		}
	}

	for (size_t i = 0; i < counter.size(); i++)
		saliecy[i] /= counter[i];
	Mat rSal(1, regNum, CV_64FC1, &saliecy[0]);
	if (bNormalize)
		normalize(rSal, rSal, 0, 1, NORM_MINMAX);

	for (int y = 0; y < sal1f.rows; y++){
		const int *idx = segIdx1i.ptr<int>(y);
		float *sal = sal1f.ptr<float>(y);
		for (int x = 0; x < sal1f.cols; x++)
			sal[x] = (float)saliecy[idx[x]];
	}
}



Mat RegionContrastSalient::hardHC(vector<Region> regions,
		Mat &regionColor,
		Mat &regionScore,
		double sigmaDist,
		Mat &originImage,
		Mat regionIdxImage, bool debug
		){
	int regNum = (int)regions.size();
	int pixelNum = originImage.rows * originImage.cols;
	regionScore = Mat::zeros(1, regNum, CV_64F);
	double* regSal = (double*)regionScore.data;
	for(int i = 0; i < regNum; ++i){
		regSal[i] = exp( sqrt(regions[i].pixNum * 2 / (float)pixelNum) -
				sqrt(sqr(regions[i].ad2c.x) + sqr(regions[i].ad2c.y))
				/ 10 );
	}
	int maxIter = 0;
	double max = regSal[0];
	for(int i = 1; i < regNum; ++i){
		if(max < regSal[i]){
			maxIter = i;
			max = regSal[i];
		}
	}
	for(int i = 0; i < regNum; ++i){
		if(i == maxIter){
			regSal[i] = 1;
		}else{
			regSal[i] = 0;
		}
	}

	Mat sal1f = Mat::zeros(originImage.size(), CV_32F);
	for(int r = 0; r < originImage.rows; ++r){
		const int * regIdx = regionIdxImage.ptr<int>(r);
		float* sal = sal1f.ptr<float>(r);
		for(int c = 0; c < originImage.cols; ++c){
			sal[c] = saturate_cast<float>(regSal[regIdx[c]]);
		}
	}
	return sal1f;
}

Mat RegionContrastSalient::originRC(vector<Region> regions, Mat &regionColor, Mat &regionScore, double sigmaDist, Mat &originImage3f, Mat regionIdxImage, Mat_<float> &cDistCache1f, bool debug){
	RegionContrast(regions, regionColor, regionScore,
			originImage3f.rows * originImage3f.cols, 0, cDistCache1f);
	int regNum = (int)regions.size();
	Mat salientScoreImage1f = Mat::zeros(originImage3f.size(), CV_32F);
	if(debug){
		printMat<double>(regionScore);
	}
//	minmaxNorm<double>(regionScore);
	cv::normalize(regionScore, regionScore, 0, 1, NORM_MINMAX, CV_64F);
	if(debug){
		printMat<double>(regionScore);
	}
	double *regSal = (double *)regionScore.data;
	for(int r = 0; r < originImage3f.rows; ++r){
		const int * regIdx = regionIdxImage.ptr<int>(r);
		float* sal = salientScoreImage1f.ptr<float>(r);
		for(int c = 0; c < originImage3f.cols; ++c){
			sal[c] = saturate_cast<float>(regSal[regIdx[c]]);
		}
	}
	Mat borderRegion1u = GetBorderReg(regionIdxImage, regNum, 0.02, 0.4);
	salientScoreImage1f.setTo(0, borderRegion1u);
	SmoothByHist(originImage3f, salientScoreImage1f, 0.1f);
	SmoothByRegion(salientScoreImage1f, regionIdxImage, regNum);
	salientScoreImage1f.setTo(0, borderRegion1u);
//	GaussianBlur(salientScoreImage1f, salientScoreImage1f, Size(3,3), 0);
	normalize(salientScoreImage1f, salientScoreImage1f, 0, 1, NORM_MINMAX);
	return salientScoreImage1f;
}

Mat RegionContrastSalient::centerRC(vector<Region> regions, Mat &regionColor, Mat &regionScore, double sigmaDist, Mat &originImage, Mat regionIdxImage, Mat_<float> &cDistCache1f, bool debug){
	RegionContrast(regions, regionColor, regionScore,
			originImage.rows * originImage.cols, 15, cDistCache1f);
	int regNum = (int)regions.size();
	Mat sal1f = Mat::zeros(originImage.size(), CV_32F);
	if(debug){
		printMat<double>(regionScore);
	}
	cv::normalize(regionScore, regionScore, 0, 1, NORM_MINMAX, CV_64F);
	if(debug){
		printMat<double>(regionScore);
	}
	double *regSal = (double *)regionScore.data;
	for(int r = 0; r < originImage.rows; ++r){
		const int * regIdx = regionIdxImage.ptr<int>(r);
		float* sal = sal1f.ptr<float>(r);
		for(int c = 0; c < originImage.cols; ++c){
			sal[c] = saturate_cast<float>(regSal[regIdx[c]]);
		}
	}
	Mat bdReg1u = GetBorderReg(regionIdxImage, regNum, 0.02, 0.4);
	sal1f.setTo(0, bdReg1u);
	SmoothByHist(originImage, sal1f, 0.1f);
	SmoothByRegion(sal1f, regionIdxImage, regNum);
	sal1f.setTo(0, bdReg1u);
//	GaussianBlur(sal1f, sal1f, Size(3,3), 0);
	normalize(sal1f, sal1f, 0, 1, NORM_MINMAX);
	return sal1f;
}

Mat RegionContrastSalient::equalize(Mat &img){
	vector<Mat> planes;
	split(img, planes);

	for(int i = 0, len = (int)planes.size(); i < len; ++i){
		equalizeHist(planes[i], planes[i]);
	}
	merge(planes, img);
	return img;
}

Mat RegionContrastSalient::getRC(Mat &img3f, Mat &regionIdxImage1i, int regNum, double sigmaDist, bool debug){
	//quantize
	Mat colorIdx1i, regSal1v, tmp, color3fv;
	img3f.convertTo(img3f, CV_32FC3, 1.0/255);

	int QuantizeNum = quantizer.Quantize(img3f, colorIdx1i, color3fv, tmp);
	if(QuantizeNum == 2){
		Mat sal;
		compare(colorIdx1i, 1, sal, CMP_EQ);
		sal.convertTo(sal, CV_32F, 1.0/ 255);

		return sal;
	}else if(QuantizeNum == 0){
		return Mat::zeros(img3f.size(), CV_32F);
	}else if(QuantizeNum == 1){
		colorIdx1i.convertTo(colorIdx1i, CV_32F, 1.0/ 255);
		return colorIdx1i;
	}

	cvtColor(color3fv, color3fv, CV_BGR2Lab);
	vector<Region> regs(regNum);

	BuildRegions(regionIdxImage1i, regs, colorIdx1i, color3fv.cols);
	Mat_<float> cDistCache1f = pairwiseColorDist(color3fv);
	Mat centerRes = centerRC(regs, color3fv, regSal1v, sigmaDist, img3f, regionIdxImage1i, cDistCache1f, debug);
	Mat originRes = originRC(regs, color3fv, regSal1v, sigmaDist, img3f, regionIdxImage1i, cDistCache1f, debug);

	if(debug){
		namedWindow("RC-Center");
		imshow("RC-Center", centerRes);
		namedWindow("RC-Origin");
		imshow("RC-Origin", originRes);

	}
//	bool empty = true;

	for(int i = 0; i < centerRes.rows; ++i){
		float * centerRow = centerRes.ptr<float>(i);
		float * originRow = originRes.ptr<float>(i);
		for(int j = 0; j < centerRes.cols; ++j){
			if(isnan(originRow[j])){
				originRow[j] = 0.f;
			}
			if(isnan(centerRow[j])){
				centerRow[j] = 0.f;
			}
			if(originRow[j] > centerRow[j]){
				centerRow[j] = originRow[j];
//				empty = false;
			}
		}
	}
	if(debug){
		int originNum = 0;
		int centerNum = 0;
		for(int i = 0; i < centerRes.rows; ++i){
			float * centerRow = centerRes.ptr<float>(i);
			float * originRow = originRes.ptr<float>(i);
			for(int j = 0; j < centerRes.cols; ++j){
				if(originRow[j] > 0){
					originNum++;
				}
				if(centerRow[j] > 0){
					centerNum++;
				}
			}
		}
		cout << "origin:" << originNum << "," << "center:" << centerNum << endl;
	}

//	if(!empty){
//		Mat hardHCRes = hardHC(regs, color3fv, regSal1v, sigmaDist, img, regIdx1i, debug);
//		if(debug){
//			namedWindow("HC-Hard");
//			imshow("HC-Hard", hardHCRes);
//		}
//		for(int i = 0; i < centerRes.rows; ++i){
//			float * centerRow = centerRes.ptr<float>(i);
//			float * hardRow = hardHCRes.ptr<float>(i);
//			for(int j = 0; j < centerRes.cols; ++j){
//				if(hardRow[j] > 0){
//					centerRow[j] = centerRow[j] * 0.7 + hardRow[j] * 0.3;
//				}
//			}
//		}
//	}
	return centerRes;
}

#endif /* RC_H_ */
