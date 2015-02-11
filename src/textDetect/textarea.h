#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
//#include "opencv2/text.hpp"
#include "opencv2/core/utility.hpp"
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../borderPosition/border.h"

#include "../textExtraction/textExtraction.h"

using namespace std;
using namespace cv;
//using namespace cv::text;

size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}

bool isRepetitive(const string& s)
{
    int count = 0;
    for (int i=0; i<(int)s.size(); i++)
    {
        if ((s[i] == 'i') ||
                (s[i] == 'l') ||
                (s[i] == 'I'))
            count++;
    }
    if (count > ((int)s.size()+1)/2)
    {
        return true;
    }
    return false;
}


//void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
//{
//    for (int r=0; r<(int)group.size(); r++)
//    {
//        ERStat er = regions[group[r][0]][group[r][1]];
//        if (er.parent != NULL) // deprecate the root region
//        {
//            int newMaskVal = 255;
//            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
//            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
//                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
//        }
//    }
//}

bool   sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}

//square of line length
int changfang(cv::Vec4i line){
	return (line[0]-line[2])*(line[0]-line[2])+(line[1]-line[3])*(line[1]-line[3]);
}

bool isQuadr2(vector<cv::Vec4i> lines,int minB1,int maxB1,int i,int j,bool smallAng, int maxGap){

	if(i==41&&j==48){
		cout<<fabs(fabs(lines[i][1]-lines[i][3])-maxGap)<<endl;
		cout<<fabs(fabs(lines[j][1]-lines[j][3])-maxGap)<<endl;
		cout<<min(fabs(lines[minB1][0]-lines[minB1][2]),fabs(lines[maxB1][0]-lines[maxB1][2]))<<endl;
	}
	if(smallAng){
		if(fabs(fabs(lines[i][1]-lines[i][3])-maxGap)<20&&
				fabs(fabs(lines[j][1]-lines[j][3])-maxGap)<20&&
				fabs(lines[i][0]-lines[j][0])>=(min(fabs(lines[minB1][0]-lines[minB1][2]),fabs(lines[maxB1][0]-lines[maxB1][2]))-20)&&
				(fabs(lines[i][0]-lines[minB1][0])<20||fabs(lines[i][0]-lines[minB1][2])<20||fabs(lines[i][0]-lines[maxB1][0])<20||fabs(lines[i][0]-lines[maxB1][2])<20)&&
				(fabs(lines[j][0]-lines[minB1][0])<20||fabs(lines[j][0]-lines[minB1][2])<20||fabs(lines[j][0]-lines[maxB1][0])<20||fabs(lines[j][0]-lines[maxB1][2])<20)){
			return true;
		}
	}
	else{

	}
	return false;
}

vector<cv::Vec4i> gLines;
int compareLineVert(const void * a, const void * b){
	int ai = *(int*)a;
	int bi = *(int*)b;

	float heightA = (gLines[ai][1]+gLines[ai][3])/2.0;
	float heightB = (gLines[bi][1]+gLines[bi][3])/2.0;

	if(heightA-heightB<0)
		return -1;
	if(heightA-heightB>0)
		return 1;
	return 0;
}

vector<Vec4i> block;
int* gSortedLines;

float xielv(map<int, int>& dict, int type){
	if(dict.size()==1)
		return 200;
	int vertN = 0;
	int ptonN = 0;
	float sum = 0.0;

	int vsize = 0;
	vector<double> tpK(2000);
	vector<int> tpN(2000);

	int maxTp = -1;
	int maxCt = 0;

	vector<int> xs, ys;
	for(map<int,int>::iterator it = dict.begin();it!=dict.end();it++){
		ys.push_back(it->first);
		xs.push_back(it->second);
	}

	/**/
	if(type==0){
		for(int i0=0;i0<ys.size()-1;i0++){

			int y1 = ys[i0];
			int x1 = xs[i0];
			for(int i1=i0+1;i1<ys.size();i1++){
				int y2 = ys[i1];
				int x2 = xs[i1];

				if(x1==x2){
					vertN++;
					continue;
				}

				float curK = 20.0*(0.0+y1-y2)/(0.0+x1-x2);
				cout<<"("<<x1<<","<<y1*20<<") ("<<x2<<","<<y2*20<<") "<<curK<<endl;
				for(int i=0;i<tpK.size();i++){
					if(fabs(tpK[i]-curK)<2){
						tpK[i]=(tpK[i]*tpN[i]+curK)/(tpN[i]+1);
						tpN[i]++;
						break;
					}
					else{
						tpK.push_back(curK);
						tpN.push_back(1);
						break;
					}
				}
			}
		}

		for(int i=0;i<tpK.size();i++){
			if(tpN[i]>maxCt){
				maxTp = i;
				maxCt = tpN[i];
			}
		}
		cout<<"xielv max: "<<maxCt<<endl;

		if(vertN>maxCt)
			return 999999.9;
		else
			return tpK[maxTp];
	}
	else{
		int len = ys.size()-1;
		int x1 = xs[0];
		int y1 = ys[0];
		int x2 = xs[1];
		int y2 = ys[1];
		int x3 = xs[len-1];
		int y3 = ys[len-1];
		int x4 = xs[len-2];
		int y4 = ys[len-2];

		if(x1==x3||x2==x4)
		{
			cout<<"vertical line"<<endl;
			return 999999.9;
		}
		else{
			return ((0.0+y1-y3)/(0.0+x1-x3)+(0.0+y2-y4)/(0.0+x2-x4))*10;
		}
	}
}

int chonghe(int line0, int line2, int l2, int r2){
	int l1 = min(line0,line2);
	int r1 = max(line0,line2);

	int ret = 0;
	if(l1>=l2&&r1<=r2) ret = r1-l1;
	if(l2>=l1&&r2<=r1) ret = r2-l2;
	if(l1>=l2&&r2<=r1) ret = r2-l1;
	if(l2>=l1&&r2>=r1) ret = r1-l2;
	//cout<<"chonghe: "<<ret<<" "<<l1<<" "<<r1<<" "<<l2<<" "<<r2<<endl;
	return ret;
}
int sampleLeft(Mat& src, vector<cv::Vec4i>& lines, float& k){

	int vsize = 0;
	vector<double> tpK(1000);
	vector<int> tpN(1000);

	int vertN = 0;

	int sortedLines[lines.size()];
	for(int i=0;i<lines.size();i++)
		sortedLines[i]=i;

	gLines = lines;
	qsort(sortedLines, lines.size(), sizeof(int), compareLineVert);
	gSortedLines = sortedLines;
	block.clear();
	vector<vector<Vec4i> > blocks;
	vector<float> lastY, firstY;
	vector<int> left, right, lefty;
	vector<float> ks;
	//Mat src2 = src.clone();

	//merge lines
	for(int i=0;i<lines.size();i++){
		Vec4i line = lines[sortedLines[i]];
		float xm = (line[0]+line[2])/2.0;
		float ym = (line[1]+line[3])/2.0;
		float myleft = min(line[0],line[2]);
		float mylefty = myleft==line[0]?line[1]:line[3];

		bool found = false;
		for(int j=0;j<blocks.size();j++){

			//cout<<"yy: "<<lastY[j]<<" "<<ym<<" ";
			if(chonghe(line[0],line[2],left[j],right[j])>20&&fabs(ym-lastY[j])<=50){
				//cout<<"found! "<<j<<" "<<left[j]<<" "<<right[j]<<" "<<lastY[j]<<" "<<xm<<" "<<ym<<" "<<endl;
				found = true;

//				float curK = (mylefty-lefty[j])/(myleft-left[j]);
//				ks[j] = (ks[j]*blocks[j].size()+curK)/(1+blocks[j].size());
				blocks[j].push_back(line);
				left[j]=min(line[0],line[2]);
				lefty[j]=mylefty;
				right[j]=max(line[0],line[2]);
				lastY[j]=ym;

				break;
			}
		}
		if(!found){
			//cout<<"new! "<<blocks.size()<<endl;
			vector<Vec4i> newBlock;
			newBlock.push_back(line);
			blocks.push_back(newBlock);
			int cur = blocks.size()-1;
			left.push_back(min(line[0],line[2]));
			right.push_back(max(line[0],line[2]));
			lefty.push_back(mylefty);
			lastY.push_back(ym);
			firstY.push_back(ym);
//			ks[cur]=0;
		}
//		cv::line(src2,Point(line[0],line[1]),Point(line[2],line[3]), CV_RGB(0,0,255),1);
//		imshow("line",src2);
//		waitKey();
	}

	//merge blocks
	set<int> subs;

	for(int i=0;i<blocks.size();i++){
		//cout<<"block size: "<<blocks[i].size()<<endl;
		if(subs.find(i)!=subs.end())
			continue;
		vector<Vec4i> block1 = blocks[i];
		float l1 = left[i];
		float r1 = right[i];
		float y1 = lastY[i];
//		float k1 = ks[i];

		for(int j=i+1;j<blocks.size();j++){
			vector<Vec4i> block2 = blocks[j];
			float l2 = left[j];
			float r2 = right[j];
			float y2 = firstY[j];
//			float k2 = ks[j];
			float x2 = (l2+r2)/2;

			//merge the blocks
			if(chonghe(l1,r1,l2,r2)>20&&y2-y1<50){
				subs.insert(j);

				l1 = l2;
				r1 = r2;
				y1 = lastY[j];
//				k1 = (k1*block1.size()+k2*block2.size())/(block1.size()+block2.size());

				left[i] = l1;
				right[i] = r1;
				lastY[i] = y1;
//				ks[i] = k1;
				for(int k=0;k<blocks[j].size();k++)
					blocks[i].push_back(blocks[j][k]);
			}
		}
	}

	//find max block
	int maxTp = -1;
	int maxCt = 0;

	for(int i=0;i<blocks.size();i++){
		if(subs.find(i)==subs.end()&&(int)blocks[i].size()>maxCt){

			maxCt = blocks[i].size();
			maxTp = i;
		}
	}
	block = blocks[maxTp];
	//cout<<"final block size: "<<block.size()<<endl;
//	k = ks[maxTp];

	//find K value of lines
	//k = xielv();
	//cout<<"xielv left: "<<k<<endl;
	if(k>99999)
		return 2;
	return 1;
	/*
	for(int i1=0;i1<10;i1++){

		int ri1 = rand()%lines.size();
		int x1 = lines[ri1][0];
		int y1 = lines[ri1][1];
		if(x1>lines[ri1][2]){
			x1 = lines[ri1][2];
			y1 = lines[ri1][3];
		}
		int count = 0;
		while(count<10){

			int ri2 = rand()%lines.size();
			if(ri1!=ri2){

				int x2 = lines[ri2][0];
				int y2 = lines[ri2][1];
				if(x2>lines[ri2][2]){
					x2 = lines[ri2][2];
					y2 = lines[ri2][3];
				}

				if(x1>=x2-2&&x1<=x2+2)
					vertN ++;
				else{
					double k = (0.0+y2-y1)/(0.0+x2-x1);
					bool found = false;
					for(int j=0;j<vsize;j++){
						if(fabs(k-tpK[j])<0.18){
							found = true;
							tpK[j] = (tpK[j]*tpN[j]+k)/(tpN[j]+1);
							tpN[j] = tpN[j]+1;

						}
					}
					if(!found){
						tpK[vsize]=k;
						tpN[vsize]=1;
						vector<int> v(1000);

						vsize++;
					}
				}
				count++;
			}
		}

	}
	int maxTp = -1;
	int maxCt = 0;

	for(int i=0;i<vsize;i++){
		if(tpN[i]>maxCt){

			maxCt = tpN[i];
			maxTp = i;
		}
	}

	cout<<maxCt<<endl;
	if(vertN>70) return 2;
	if(maxCt>=70){
		k = tpK[maxTp];
		return 1;
	}
	return 0;
	*/
}

int sampleRight(vector<cv::Vec4i>& lines, float& k){
	int vsize = 0;
	vector<double> tpK(2000);
	vector<int> tpN(2000);

	int vertN = 0;

	for(int i1=0;i1<lines.size();i1++){

		int ri1 = i1;
		int x1 = lines[ri1][0];
		int y1 = lines[ri1][1];
		if(x1<lines[ri1][2]){
			x1 = lines[ri1][2];
			y1 = lines[ri1][3];
		}
		int count = 0;
		for(int i2=i1+1;i2<lines.size();i2++){

			int ri2 = i2;
			if(ri1!=ri2){

				int x2 = lines[ri2][0];
				int y2 = lines[ri2][1];
				if(x2<lines[ri2][2]){
					x2 = lines[ri2][2];
					y2 = lines[ri2][3];
				}

				if(x1>=x2-2&&x1<=x2+2)
					vertN ++;
				else{
					double k = (0.0+y2-y1)/(0.0+x2-x1);
					bool found = false;
					for(int j=0;j<vsize;j++){
						if(fabs(k-tpK[j])<0.18){
							found = true;
							tpK[j] = (tpK[j]*tpN[j]+k)/(tpN[j]+1);
							tpN[j] = tpN[j]+1;

						}
					}
					if(!found){
						tpK[vsize]=k;
						tpN[vsize]=1;
						vector<int> v(1000);

						vsize++;
					}
				}
				count++;
			}
		}


	}
	int maxTp = -1;
	int maxCt = 0;

	for(int i=0;i<vsize;i++){
		if(tpN[i]>maxCt){

			maxCt = tpN[i];
			maxTp = i;
		}
	}
	cout<<maxCt<<endl;
	if(vertN>70) {
		//cout<<"xielv right: inf"<<endl;
		k=99999.9;
		return 2;
	}
	if(maxCt>=0){
		//cout<<"xielv right: "<<tpK[maxTp]<<endl;
		k = tpK[maxTp];
		return 1;
	}
	return 0;
}

int sampleTop(vector<cv::Vec4i>& lines, float& k){
	int vsize = 0;
	vector<double> tpK(1000);
	vector<int> tpN(1000);

	int vertN = 0;

	for(int i1=0;i1<10;i1++){

		int ri1 = rand()%lines.size();
		int x1 = lines[ri1][0];
		int y1 = lines[ri1][1];
		if(y1>lines[ri1][3]){
			x1 = lines[ri1][2];
			y1 = lines[ri1][3];
		}
		int count = 0;
		while(count<10){

			int ri2 = rand()%lines.size();
			if(ri1!=ri2){

				int x2 = lines[ri2][0];
				int y2 = lines[ri2][1];
				if(y2>lines[ri2][3]){
					x2 = lines[ri2][2];
					y2 = lines[ri2][3];
				}

				if(x1>=x2-2&&x1<=x2+2)
					vertN ++;
				else{
					double k = (0.0+y2-y1)/(0.0+x2-x1);
					bool found = false;
					for(int j=0;j<vsize;j++){
						if(fabs(k-tpK[j])<0.18){
							found = true;
							tpK[j] = (tpK[j]*tpN[j]+k)/(tpN[j]+1);
							tpN[j] = tpN[j]+1;
						}
					}
					if(!found){
						tpK[vsize]=k;
						tpN[vsize]=1;
						vector<int> v(1000);

						vsize++;
					}
				}
				count++;
			}
		}
	}
	int maxTp = -1;
	int maxCt = 0;

	for(int i=0;i<vsize;i++){
		if(tpN[i]>maxCt){

			maxCt = tpN[i];
			maxTp = i;
		}
	}
	cout<<maxCt<<endl;
	if(vertN>70) return 2;
	if(maxCt>=70){
		k = tpK[maxTp];
		return 1;
	}
	return 0;
}

int sampleDirection(Mat src,vector<cv::Vec4i>& lines, float& k){

	float kl;
	int left = sampleLeft(src,lines, kl);

	float kr;
	int right = sampleRight(block, kr);

//	float kt;
//	int top = sampleTop(lines, kt);

	if(left>0){
		if(right==1&&kl*kr<0){
			k = kl;
			return 1;
		}
		if(left==2)
			return 2;

		k=kl;
		return 3;
	}

//	if(top>0){
//		k=kt;
//		return 4;
//	}
}

void myDrawLine(int idx, int x, int y, float k, int cols, int rows){
	if(x<0) x = 0;
	if(x>cols) x = cols;
	if(y<0) y = 0;
	if(y>rows) x = rows;

	if(k==0)
	{
		//line(img,Point(0,y),Point(img.cols,y),CV_RGB(255,0,0),2);
		finalines[idx][0] = 0;
		finalines[idx][1] = y;
		finalines[idx][2] = cols;
		finalines[idx][3] = y;
		return;
	}
	if(k<99999){
		float b = y-k*x;
		if(fabs(k)<1){
			int x1 = 0;
			int y1 = b;
			int x2 = cols;
			int y2 = k*x2+b;
			//line(img,Point(x1,y1),Point(x2,y2),CV_RGB(255,0,0),2);
			finalines[idx][0] = x1;
			finalines[idx][1] = y1;
			finalines[idx][2] = x2;
			finalines[idx][3] = y2;
		}
		else{
			int y1 = 0;
			int x1 = (y1-b)/k;
			int y2 = rows;
			int x2 = (y2-b)/k;
			//line(img,Point(x1,y1),Point(x2,y2),CV_RGB(255,0,0),2);
			finalines[idx][0] = x1;
			finalines[idx][1] = y1;
			finalines[idx][2] = x2;
			finalines[idx][3] = y2;
		}
	}
	if(k>=99999){
		//line(img,Point(x,0),Point(x,img.rows),CV_RGB(255,0,0),2);
		finalines[idx][0] = x;
		finalines[idx][1] = 0;
		finalines[idx][2] = x;
		finalines[idx][3] = rows;
	}
}

void textBorder(Mat& orig, Mat& src, vector<Mat>& rst){
	std::vector<cv::Point2f> corners;
	for (int i = 0; i < finalines.size(); i++)
	{
		for (int j = i+1; j < finalines.size(); j++)
		{
			cv::Point2f pt = computeIntersect(finalines[i], finalines[j]);

			if(pt.x<0&&pt.x>-10) pt.x = 0;
			if(pt.y<0&&pt.y>-10) pt.y = 0;
			if(pt.x>src.cols&&pt.x<src.cols+10) pt.x = src.cols;
			if(pt.y>src.rows&&pt.y<src.rows+10) pt.y = src.rows;
			if (pt.x >= 0 && pt.y >= 0&&pt.x<=src.cols&&pt.y<=src.rows){

				corners.push_back(pt);
			}

		}
	}

	std::vector<cv::Point2f> approx;
	cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true) * 0.02, true);

	if (approx.size() != 4)
	{
		std::cout << "The object is not quadrilateral!" << std::endl;
		rst.push_back(orig);
		return;
	}

	center.x = 0.0;
	center.y = 0.0;
	// Get mass center
	for (int i = 0; i < corners.size(); i++)
		center += corners[i];
	center *= (1. / corners.size());

	sortCorners(corners, center);
//	cout<<"center "<<center.x<<" "<<center.y<<endl;
//	for(int i=0;i<4;i++){
//		cout<<"corner "<<corners[i].x<<" "<<corners[i].y<<endl;
//	}
	if (corners.size() == 0){
		std::cout << "The corners were not sorted correctly!" << std::endl;
		rst.push_back(orig);
		return;
	}

	Mat turned;
	turnImage(orig, turned, corners, scale);
	rst.push_back(turned);
}

int detectText2(Mat& orig, Mat& src, vector<Mat>& rst, bool border){
	int result = -1;

	Mat image,grad_x,abs_grad_x,grad_y,abs_grad_y,grad;
	image = src.clone();
	Mat grey;
	cv::cvtColor(image, grey, CV_BGR2GRAY);

	int ddepth = 3;

	cv::Sobel(grey,grad_x,ddepth,1,0);
	cv::convertScaleAbs(grad_x,abs_grad_x);

	cv::Sobel(grey,grad_y,ddepth,0,1);
	cv::convertScaleAbs(grad_y,abs_grad_y);

	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);
	cv::threshold(grad,grad,40.0,255,CV_THRESH_TOZERO);

	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(grad, lines, 1, CV_PI/180, 100, 70, 20);

	//imshow("grad", grad);
	//waitKey();

	int vsize = 0;
	vector<double> tpK(1000);
	vector<int> tpN(1000);
	vector<vector<int> > tpV(1000);

	int vertN = 0;
	vector<int> verts;

	for(int i=0;i<lines.size();i++){
		cv::Vec4i v = lines[i];
		if(v[0]>=v[2]-2&&v[0]<=v[2]+2)
		{
			vertN ++;
			verts.push_back(i);
		}
		else{
			double k = (0.0+v[3]-v[1])/(0.0+v[2]-v[0]);
			bool found = false;
			for(int j=0;j<vsize;j++){
				if(fabs(k-tpK[j])<0.18){
					found = true;
					tpK[j] = (tpK[j]*tpN[j]+k)/(tpN[j]+1);
					tpN[j] = tpN[j]+1;
					tpV[j].push_back(i);
				}
			}
			if(!found){
				tpK[vsize]=k;
				tpN[vsize]=1;
				vector<int> v(1000);
				v.push_back(i);
				tpV.push_back(v);
				vsize++;
			}
		}

		//line( grad, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(255,255,255));
	}
	//cout<<lines.size()<<endl;
	//imshow("grad", grad);
	//waitKey();

	int maxTp = -1;
	int maxCt = 0;

	int secTp = -1;
	int secCt = 0;

	for(int i=0;i<vsize;i++){
		if(tpN[i]>maxCt){
			if(maxCt>secCt){
				secCt = maxCt;
				secTp = maxTp;
			}
			maxCt = tpN[i];
			maxTp = i;
		}
		else if(tpN[i]>secCt){
			secCt = tpN[i];
			secTp = i;
		}
	}

	if(vertN>maxCt){
		if(maxCt>secCt){
			secCt = maxCt;
			secTp = maxTp;
		}
		maxCt = vertN;
		maxTp = -1;
	}
	else{
		if(vertN>secCt){
			secCt = vertN;
			secTp = -1;
		}
	}
	//Test Paper or graph photo

	cout<<"is paper? "<<maxCt<<" "<<secCt<<" "<<(maxCt>=2*secCt)<<endl;
	if(maxTp!=-1&&fabs(tpK[maxTp])>0.5&&(secTp==-1||fabs(tpK[secTp])<0.2)&&secCt>200){
		maxTp = secTp;
		maxCt = secCt;
		secTp = 0;
		secCt = 0;
	}

	if(maxCt>600||maxCt<100||maxCt<2*secCt||!(maxTp==-1||fabs(tpK[maxTp])<0.5))
	{
		return -1;
	}

	//vertical direction
	if(maxTp==-1){
//		Mat src2 = src.clone();
//		for(int i=0;i<verts.size();i++){
//			int ln = verts[i];
//			line( src2, cv::Point(lines[ln][0], lines[ln][1]), cv::Point(lines[ln][2], lines[ln][3]), CV_RGB(0,255,0));
//		}
//
//		imshow("image1", src2);
//		waitKey();
		return 0;
	}

	//horizontal direction
	if(maxTp!=-1){
		float k = 0.0;
		float k0 = maxTp!=-1?tpK[maxTp]:99999.9;
		vector<Vec4i> tosample;

		for(int i=0;i<tpV[maxTp].size();i++)
		{
			Vec4i vec(lines[tpV[maxTp][i]]);
			tosample.push_back(vec);
		}

		int doctype = sampleDirection(src,tosample,k);

		//cout<<"doctype "<<doctype<<endl;

		int upM = 0;
		int downM = 0;
		int minB1 = -1;
		int maxB1 = -1;
		cout<<"max tp "<<maxTp<<" "<<secTp<<" "<<vertN<<" "<<verts.size()<<endl;
		Mat src3 = Mat::zeros(src.rows,src.cols,CV_8UC1);
//		Mat src2 = src.clone();

		for(int i=0;i</*tpV[maxTp].size()*/block.size();i++){
			//int ln = tpV[maxTp][i];

			Vec4i myline = block[i];
//			line( src2, cv::Point(myline[0], myline[1]), cv::Point(myline[2], myline[3]), CV_RGB(0,255,0));
			line( src3, cv::Point(myline[0], myline[1]), cv::Point(myline[2], myline[3]), CV_RGB(255,255,255));

			//line( src, cv::Point(lines[ln][0], lines[ln][1]), cv::Point(lines[ln][2], lines[ln][3]), CV_RGB(0,255,0));

		}

//		imshow("image1", src2);
		Mat eroElm = getStructuringElement(MORPH_RECT, Size(3,3),Point(1,1));
		Mat src4;
		erode(src3,src4,eroElm);
//		imshow("image2", src3);
//		waitKey();
		//Canny( src4, src4, 100, 200, 3 );
		Mat nonZeroCoordinates;
		findNonZero(src4, nonZeroCoordinates);
		map<int,int> dict;

		int up = 9999, bt = -1, left = 9999, right = -1;
		int upx, btx, lefty, righty;

		for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
			//cout << "Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y << endl;
			Point p = nonZeroCoordinates.at<Point>(i);
			int x = p.x;
			int y = p.y;
			if(dict.find(y/20)!=dict.end()){
				if(x<dict[y/20])
					dict[y/20]=x;
			}
			else{
				dict[y/20]=x;
			}

			if(x<left){left=x;lefty=y;}
			if(x>right){right=x;righty=y;}
			if(y<up){up=y;upx=x;}
			if(y>bt){bt=y;btx=x;}
		}

		int elimCount = 1;
		bool edgeFail = false;

		while(elimCount>0){
			elimCount = 0;
			vector<int> ys,xs;
			for(map<int,int>::iterator it=dict.begin();it!=dict.end();it++){
				ys.push_back(it->first);
				xs.push_back(it->second);
				//cout<<"POINT: "<<it->first<<" "<<it->second<<endl;
			}

			if(xs.size()<2){edgeFail = true; break;}

			if(xs[0]-xs[1]>20)
			{
				map<int,int>::iterator it = dict.find(ys[0]);
				dict.erase(it);
				elimCount++;
			}

			for(int i=0;i<ys.size()-1;i++){
				if(xs[i+1]-xs[i]>20)
				{
					map<int,int>::iterator it = dict.find(ys[i+1]);
					if(it!=dict.end())
					{
						dict.erase(it);
						//cout<<"erase: "<<i+1<<endl;
						elimCount++;
					}
				}

				if(xs[i]-xs[i+1]>100){
					map<int,int>::iterator it = dict.find(ys[i]);
					if(it!=dict.end())
					{
						dict.erase(it);
						//cout<<"erase: "<<i+1<<endl;
						elimCount++;
					}
				}
			}
		}

//		Mat src5 = Mat::zeros(src4.rows,src4.cols,CV_8UC1);
//		for(map<int,int>::iterator it=dict.begin();it!=dict.end();it++){
//			circle(src5, Point(it->second,20*it->first), 3, CV_RGB(255,255,255), 1);
//		}
//		imshow("image2", src5);
//		waitKey();

		float kl = 99999.9;
		if(!edgeFail)
			kl = xielv(dict,1);

		dict.clear();
		for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
			//cout << "Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y << endl;
			Point p = nonZeroCoordinates.at<Point>(i);
			int x = p.x;
			int y = p.y;
			if(dict.find(y/20)!=dict.end()){
				if(x>dict[y/20])
					dict[y/20]=x;
			}
			else{
				dict[y/20]=x;
			}

		}

		elimCount = 1;
		while(elimCount>0&&!edgeFail){
			elimCount = 0;
			vector<int> ys,xs;
			for(map<int,int>::iterator it=dict.begin();it!=dict.end();it++){
				ys.push_back(it->first);
				xs.push_back(it->second);
				//cout<<"POINT: "<<it->first<<" "<<it->second<<endl;
			}
			if(xs.size()<2){edgeFail = true; break;}
			if(xs[1]-xs[0]>20)
			{
				map<int,int>::iterator it = dict.find(ys[0]);
				dict.erase(it);
				elimCount++;
			}

			for(int i=0;i<ys.size()-1;i++){
				if(xs[i]-xs[i+1]>20)
				{
					map<int,int>::iterator it = dict.find(ys[i+1]);
					dict.erase(it);
					//cout<<"erase: "<<i+1<<endl;
					elimCount++;
				}
			}
		}
//		Mat src6 = Mat::zeros(src4.rows,src4.cols,CV_8UC1);
//		for(map<int,int>::iterator it=dict.begin();it!=dict.end();it++){
//			cout<<"right out: "<<it->first<<" "<<it->second<<endl;
//			circle(src6, Point(it->second,20*it->first), 3, CV_RGB(255,255,255), 1);
//		}
//		imshow("image3", src6);

		float kr = 99999.9;
		if(!edgeFail)
			xielv(dict,1);
//		cout<<"xielv: "<<kl<<" "<<kr<<endl;
//		waitKey();

		result = 1;
		if(fabs(kl)<3.5||fabs(kr)<3.5||kl>=99999||kr>=99999){
			kl = 999999;
			kr = 999999;
			k0 = 0;
			return 0;
		}

		//Mat src7 = src.clone();

		myDrawLine(0,upx,up-10,k0,src.cols,src.rows);
		myDrawLine(1,btx,bt+10,k0,src.cols,src.rows);
		myDrawLine(2,left-10,lefty,kl,src.cols,src.rows);
		myDrawLine(3,right+10,righty,kr,src.cols,src.rows);
		if(!border)
			textBorder(orig, src, rst);

//		imshow("imageF",src7);
//		waitKey();
	}
	return result;
}

int textDetect(Mat& src, vector<Mat>& textPieces, bool border){

	Mat tsrc;
	if (src.empty())
		return -1;
	scale = 1.0;
	myNormalSize(src,tsrc,CV_32S);
	vector<Mat> rst;
	int ret = detectText2(src, tsrc,rst,border);

	if(ret==1){
		if(border){
			textPieces.push_back(src);
		}
		else{
			textPieces = rst;
		}
	}else if(ret==0){
		textPieces.push_back(src);
	}else if(ret==-1){
		TextExtraction te;
		vector<Rect> regions = te.textExtract(src);
		textPieces = te.findRegions(src, regions);
	}
}
