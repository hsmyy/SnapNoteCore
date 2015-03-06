#ifndef SRC_NOISELEVEL_H_
#define SRC_NOISELEVEL_H_

#include <iostream>
#include <limits>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <gsl/gsl_cdf.h>
#include "../utils/NumUtil.h"

using namespace std;
using namespace cv;

int imfilter(Mat &src, Mat &ker, Mat &dest)
{
     Point anchor( -1,-1);
     double delta = 0.0;
     cv::filter2D(src, dest, -1, ker, anchor, delta, BORDER_REPLICATE);
     return 1;
}

void my_convmtx2(Mat& kernal, Mat& conv, int m, int n){
	int s1 = kernal.rows;
	int s2 = kernal.cols;

	conv = Mat::zeros((m-s1+1)*(n-s2+1),m*n,CV_64FC1);
	int k = 0;

	for(int i=0;i<m-s1+1;i++){
		for(int j=0;j<n-s2+1;j++){
			for(int p=0;p<s1;p++){
				for(int q=0;q<s2;q++)
					conv.at<double>(k,(i+p)*n+j+q)=kernal.at<double>(p,q);
			}
			k++;
		}
	}
}

void im2col(Mat& input, Mat& result, int rowBlock, int colBlock){
	int m = input.rows;
	int n = input.cols;

	// using right x = col; y = row
	int yB = m - rowBlock + 1;
	int xB = n - colBlock + 1;

	// you know the size of the result in the beginning, so allocate it all at once
	result = cv::Mat::zeros(xB*yB,rowBlock*colBlock,CV_64FC1);
	for(int i = 0; i< yB; i++)
	{
		for (int j = 0; j< xB; j++)
		{
			// here yours is in different order than I first thought:
			//int rowIdx = j + i*xB;    // my intuition how to index the result
			int rowIdx = i + j*yB;

			for(unsigned int yy =0; yy < rowBlock; ++yy)
				for(unsigned int xx=0; xx < colBlock; ++xx)
				{
					// here take care of the transpose in the original method
					//int colIdx = xx + yy*colBlock; // this would be not transposed
					int colIdx = xx*rowBlock + yy;
					result.at<double>(rowIdx,colIdx) = 0.0+ input.at<double>(i+yy, j+xx);
				}

		}
	}
}

Mat SortRows(Mat FeatureMatrix)
{
	// Bubble Sorting
	Mat SortedFeatureMatrix(FeatureMatrix.rows,FeatureMatrix.cols, FeatureMatrix.type());
	FeatureMatrix.row(0).copyTo(SortedFeatureMatrix.row(0));
	for (int i = 1; i < FeatureMatrix.rows; i++) {
		int index = i;
		//cout<<"sorting "<<i<<endl;
		while(index>0)
		{
			bool cmp = false;
			for (int j = 0; j < FeatureMatrix.cols; j++) {
				if(FeatureMatrix.at<double>(i, j)==SortedFeatureMatrix.at<double>(index-1, j))
				{
					continue;
				}
				else if(FeatureMatrix.at<double>(i, j)<SortedFeatureMatrix.at<double>(index-1, j))
				{
					cmp = true;
					SortedFeatureMatrix.row(index-1).copyTo(SortedFeatureMatrix.row(index));
					break;
				}
				else if(FeatureMatrix.at<double>(i, j)>SortedFeatureMatrix.at<double>(index-1, j))
				{
					cmp = false;
					FeatureMatrix.row(i).copyTo(SortedFeatureMatrix.row(index));
					break;
				}
			}
			if(cmp == true)
			{
				index-=1;
				if(index == 0)
				{
					FeatureMatrix.row(i).copyTo(SortedFeatureMatrix.row(index));
				}
				continue;
			}
			else
			{
				break;
			}
		}
	}
	return SortedFeatureMatrix;
}

double myeps(double d){
	return eps(d);
}

double getRankTol(Mat A, Mat s){

	double s1 = max(A.cols,A.rows);
	double s2 = numeric_limits<double>::min();
	for(int i=0;i<s.cols;i++)
	{
		if(s.at<double>(0,i)>s2)
			s2 = s.at<double>(0,i);
	}

	double s3 = myeps(s2);
	return s1*s3;
}

void myNormalSize2(Mat& src, Mat& tsrc, int type){

	double bili = src.cols>src.rows?(src.cols>500?500.0/src.cols:1):(src.rows>500?500.0/src.rows:1);
	Size sz = Size(src.cols*bili,src.rows*bili);

	tsrc = Mat(sz,type);
	cv::resize(src, tsrc, sz);
}

int noiseLevel(Mat& img, vector<double>& rst, int itr, double conf, int decim, int patchsize){

	Mat img2;
	myNormalSize2(img,img2,CV_64F);
	img = img2.clone();

	Mat kh(1,3,CV_64FC1);
	kh.at<double>(0,0) = -0.5;
	kh.at<double>(0,1) = 0.0;
	kh.at<double>(0,2) = 0.5;

	Mat kv = kh.t();

	Mat imgh, imgv;
	imfilter(img,kh,imgh);
	imfilter(img,kv,imgv);

	imgh = imgh.colRange(1,imgh.cols-1);
	imgv = imgv.rowRange(1,imgv.rows-1);

	imgh = imgh.mul(imgh);
	imgv = imgv.mul(imgv);

	Mat Dh, Dv, DD;
	my_convmtx2(kh,Dh,patchsize,patchsize);
	my_convmtx2(kv,Dv,patchsize,patchsize);

	DD = Dh.t()*Dh + Dv.t()*Dv;
	Mat matS;
	SVD::compute(DD,matS);

	matS = matS.t();
	//cout<<matS.rows<<" "<<matS.cols<<endl;
	double tol = getRankTol(DD,matS);
	int r = 0;
	for(int i=0;i<matS.cols;i++)
		if(matS.at<double>(0,i)>tol)
			r++;

	double dtr = 0;
	for(int i=0;i<DD.cols;i++)
		dtr += DD.at<double>(i,i);

	double a = r/2.0;
	double b = 2.0*dtr/r;
	double tao0 =  gsl_cdf_gamma_Pinv (conf, a, b);

	Mat channelsX[img.channels()],channelsXh[img.channels()],channelsXv[img.channels()];

	split(img,channelsX);
	split(imgh,channelsXh);
	split(imgv,channelsXv);

	for(int i=img.channels()-1;i>=0;i--){

		Mat X, Xh, Xv;
		im2col(channelsX[i],X,patchsize,patchsize);
		im2col(channelsXh[i],Xh,patchsize,patchsize-2);
		im2col(channelsXv[i],Xv,patchsize-2,patchsize);

		X = X.t();Xh = Xh.t();Xv = Xv.t();

		Mat Xcv, Xtr;
		vconcat(Xh,Xv,Xcv);

		Xtr = Mat::zeros(1,Xcv.cols,CV_64FC1);
		for(int c=0;c<Xcv.cols;c++){
			for(int r=0;r<Xcv.rows;r++)
				Xtr.at<double>(0,c)+=Xcv.at<double>(r,c);
		}

		if(decim > 0){
			Mat XtrX;
			vconcat(Xtr,X,XtrX);
			XtrX = SortRows(XtrX.t()).t();
			int p = XtrX.cols/(decim+1);

			Mat sample = XtrX.col(0);

			for(int i=1;i<p;i++){
				int idx =i*(decim+1);
				hconcat(sample,XtrX.col(idx),sample);
			}

			Xtr = sample.rowRange(0,1);
			X = sample.rowRange(1,sample.rows);
		}

		double tao = numeric_limits<double>::max();
		double sig2 = 0;
		Mat cov;

		if(X.cols>=X.rows){

			cov = X*X.t()/(X.cols-1);
			Mat eigv;
			eigen(cov,eigv);

			sig2 = eigv.at<double>(eigv.rows-1,0);
		}
		//cout<<"sig2 "<<sig2<<endl;

		for(int i2=1;i2<itr;i2++){

			tao = sig2 * tao0;

			vector<int> pv;
			pv.clear();
			for(int c=0;c<Xtr.cols;c++){
				if(Xtr.at<double>(0,c)<tao)
					pv.push_back(c);
			}

			if(pv.size()<X.rows)
				break;


			Mat Xt = X.t();
			Mat Xtrt = Xtr.t();

			Mat sample1 = Mat(0,Xtr.rows,CV_64F);
			Mat sample2 = Mat(0,X.rows,CV_64F);

			for(int pi=0;pi<pv.size();pi++){
				int idx = pv[pi];
				sample1.push_back(Xtrt.row(idx));
				sample2.push_back(Xt.row(idx));
			}

			Xtr = sample1.t();
			X = sample2.t();

			cov = X*X.t()/(X.cols-1);

			Mat eigv;
			eigen(cov,eigv);

			sig2 = eigv.at<double>(eigv.rows-1,0);
			//cout<<"sig2 "<<sig2<<endl;
		}
		if(sig2 < 0.0001)
			rst.push_back(0);
		else
			rst.push_back(sqrt(sig2));
	}
}

vector<double> getChannelsNoiseLevels(Mat& img, int itr=25, double confi=0.99, int decim=0, int patSize=7){

	if(img.channels()==3)
		img.convertTo(img, CV_64FC3);
	if(img.channels()==1)
		img.convertTo(img, CV_64FC1);

	vector<double> rst;
	noiseLevel(img, rst, itr, confi, decim, patSize);
	return rst;
}

double getAvgNoiseLevel(Mat& img, int itr=25, double confi=0.99, int decim=0, int patSize=7){
	vector<double> channelNoiseLevels = getChannelsNoiseLevels(img, itr, confi, decim, patSize);
	double sum = 0.0;
	for(int i=0;i<channelNoiseLevels.size();i++)
		sum+=channelNoiseLevels[i];

	return sum/channelNoiseLevels.size();
}

double getMaxNoiseLevel(Mat& img, int itr=25, double confi=0.99, int decim=0, int patSize=7){
	vector<double> channelNoiseLevels = getChannelsNoiseLevels(img, itr, confi, decim, patSize);
	double res = -1.0;
	for(int i=0;i<channelNoiseLevels.size();i++)
		if(channelNoiseLevels[i]>res)
			res = channelNoiseLevels[i];

	return res;
}
#endif
