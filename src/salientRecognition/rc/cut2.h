/*
 * cut2.h
 *
 *  Created on: Jan 27, 2015
 *      Author: fc
 */

#ifndef CUT2_H_
#define CUT2_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <list>

using namespace std;

class RegionCut{
private:
	typedef struct {
			float pixelNum;
			float centerX, centerY;
		} ConnectRegion;

	typedef struct { uchar r, g, b; } rgb;
public:
	RegionCut(float bgThreshold, float fgThreshold, bool debug = false);
	Mat cut(Mat img1f);
	void findConnectedRegion(Mat &img1f, Mat &flag, vector<ConnectRegion> &regionInfos);
	void modifyRegion(Mat &img1f, Mat &flag, vector<ConnectRegion> &regionInfos);

	Mat findMainBorder(Mat &img1f);
	void broadSearch(Mat &img1f, Mat &region, int y, int x);
private:
	float _bgThreshold;
	float _fgThreshold;
	bool _debug;
};

RegionCut::RegionCut(float bgThreshold, float fgThreshold, bool debug):
		_bgThreshold(bgThreshold), _fgThreshold(fgThreshold), _debug(debug){

}

Mat RegionCut::cut(Mat img1f){
	// 1) binarilization
	threshold(img1f, img1f, _bgThreshold, 1, THRESH_BINARY);
	// 2) get region connected map
	vector<ConnectRegion> cRegions;
	Mat cRegionIdx;
	findConnectedRegion(img1f, cRegionIdx, cRegions);
	// 3) judge big connection zone
	// 4) remove small connection zone if big zone exists
	// 5) link small connection zone if big zone not exists
	modifyRegion(img1f, cRegionIdx, cRegions);
	return img1f;
}

void RegionCut::modifyRegion(Mat &img1f, Mat &flag, vector<ConnectRegion> &regionInfos){
	float max = 0;
	int maxId = 0;
	for(int i = 0, len = (int)regionInfos.size(); i < len; ++i){
		if(regionInfos[i].pixelNum > max){
			max = regionInfos[i].pixelNum;
			maxId = i;
		}
	}
	//debug
	if(_debug){
		cout << "[regions]" << regionInfos.size() << "[max region]:" << max << ",";
		Mat highlight;
		img1f.copyTo(highlight);
		for(int y = 0; y < flag.rows; ++y){
			for(int x = 0; x < flag.cols; ++x){
				if(flag.at<int>(y,x) != maxId){
					highlight.at<float>(y,x) /= 2;
				}
			}
		}
//		namedWindow("main region");
//		imshow("main region", highlight);
	}
	// if the region is larger than 15%, remove other region, other remove all
	if(max > 0.15){
		for(int y = 0; y < flag.rows; ++y){
			for(int x = 0; x < flag.cols; ++x){
				if(flag.at<int>(y,x) != maxId){
					img1f.at<float>(y,x) = 0;
				}
			}
		}
		//draw centroid on this region
		// TODO filling some gap
		findMainBorder(img1f);
	}else{
		img1f = 0;
	}
}

Mat RegionCut::findMainBorder(Mat &img1f){
	Mat region(img1f.size(), 4);
	region = 0;

	// find border and outside empty zone
	for(int x = 0; x < img1f.cols; ++x){
		broadSearch(img1f, region, 0, x);
		broadSearch(img1f, region, img1f.rows - 1, x);
	}
	if(_debug){
		Mat showRegion(region.size(), CV_32F);
		for(int y = 0; y < region.rows; ++y){
			for(int x = 0; x < region.cols; ++x){
				if(region.at<int>(y,x) > 0){
					showRegion.at<float>(y,x) = 1;
				}
			}
		}
//		namedWindow("broadSearch");
//		imshow("broadSearch", showRegion);

	}
	for(int y = 1; y < img1f.rows; ++y){
		broadSearch(img1f, region, y, 0);
		broadSearch(img1f, region, y, img1f.cols - 1);
	}

	if(_debug){
		Mat showRegion(region.size(), CV_32F);
		for(int y = 0; y < region.rows; ++y){
			for(int x = 0; x < region.cols; ++x){
				if(region.at<int>(y,x) > 0){
					showRegion.at<float>(y,x) = 1;
				}
			}
		}
//		namedWindow("broadSearch2");
//		imshow("broadSearch2", showRegion);
	}

	//fill in color
	for(int y = 0; y < img1f.rows; ++y){
		float *imgRow = img1f.ptr<float>(y);
		int *regionRow = region.ptr<int>(y);
		for(int x = 0; x < img1f.cols; ++x){
			if(regionRow[x] == 0){
				imgRow[x] = 1;
			}
		}
	}

	return region;
}

void RegionCut::broadSearch(Mat &img1f, Mat &region, int y, int x){
	if(region.at<int>(y,x) != 0){
		return;
	}
	queue<Point, list<Point> > queue;
	queue.push(Point(x,y));
	int rowNum = img1f.rows, colNum = img1f.cols;
	while(queue.size()){
		Point pt = queue.front();
		queue.pop();
		region.at<int>(pt.y, pt.x) = 2;
		Point pointSet[] = {
			Point(pt.x - 1, pt.y - 1),
			Point(pt.x - 1, pt.y),
			Point(pt.x - 1, pt.y + 1),
			Point(pt.x, pt.y - 1),
			Point(pt.x, pt.y + 1),
			Point(pt.x + 1, pt.y - 1),
			Point(pt.x + 1, pt.y),
			Point(pt.x + 1, pt.y + 1),
		};
		for(int i = 0; i < 8; ++i){
			Point *curPt = &pointSet[i];
			if(curPt->x >= 0 && curPt->x < colNum &&
					curPt->y >= 0 && curPt->y < rowNum){
				if(img1f.at<float>(curPt->y, curPt->x) > 0){
					region.at<int>(curPt->y, curPt->x) = 3;
				}else{
					if(region.at<int>(curPt->y, curPt->x) == 0){
						region.at<int>(curPt->y, curPt->x) = 1;
						queue.push(*curPt);
					}
				}
			}
		}
	}
}

void RegionCut::findConnectedRegion(Mat &img1f, Mat &flag, vector<ConnectRegion> &regionInfos){
	flag.create(img1f.size(), 4);
	flag = -1;
	int regionIdx = 0;
	//find conn
	for(int y = 0; y < img1f.rows; ++y){
		int *flagRow = flag.ptr<int>(y);
		for(int x = 0; x < img1f.cols; ++x){
			if(img1f.at<float>(y,x) > 0 && flag.at<int>(y,x) == -1){
				int pixelNum = 0, centerX = 0, centerY = 0;
				Point pt(x,y);
				queue<Point, list<Point> > neighbors;
				flagRow[x] = regionIdx;
				neighbors.push(pt);

				while(neighbors.size() > 0){
					pt = neighbors.front();
					neighbors.pop();
					pixelNum += 1;
					centerX += pt.x;
					centerY += pt.y;

					//find four direction
					Point pointSet[] = {
							Point(pt.x + 1, pt.y),
							Point(pt.x, pt.y + 1),
							Point(pt.x + 1, pt.y + 1),
							Point(pt.x + 1, pt.y - 1),
							Point(pt.x - 1, pt.y + 1),
							Point(pt.x - 1, pt.y)
					};
					for(int i = 0; i < 6; ++i){
						Point *checkPt = &pointSet[i];
						if(checkPt->x < img1f.cols && checkPt->x >= 0 &&
								checkPt->y >= 0 && checkPt->y < img1f.rows &&
								img1f.at<float>(checkPt->y, checkPt->x) > 0 &&
								flag.at<int>(checkPt->y, checkPt->x) == -1){
							flag.at<int>(checkPt->y, checkPt->x) = regionIdx;
							neighbors.push(*checkPt);
						}
					}
				}

				regionIdx++;
				ConnectRegion newRegion;
				newRegion.centerX = pixelNum / (float)img1f.cols;
				newRegion.centerY = pixelNum / (float)img1f.rows;
				newRegion.pixelNum = pixelNum / (float)img1f.cols / img1f.rows;
				regionInfos.push_back(newRegion);
			}
		}
	}
	if(_debug){
		vector<rgb> colors(regionInfos.size());
		for(int i = 0; i < (int)regionInfos.size(); ++i){
			colors[i].b = rand() % 255;
			colors[i].g = rand() % 255;
			colors[i].r = rand() % 255;
		}
		Mat color(flag.size(), CV_8UC3);
		for(int y = 0; y < flag.rows; ++y){
			for(int x = 0; x < flag.cols; ++x){
				if(flag.at<int>(y,x) != -1){
					color.at<Vec3b>(y,x)[0] = colors[flag.at<int>(y,x)].b;
					color.at<Vec3b>(y,x)[1] = colors[flag.at<int>(y,x)].g;
					color.at<Vec3b>(y,x)[2] = colors[flag.at<int>(y,x)].r;
				}
			}
		}
//		namedWindow("ContinousRegion");
//		imshow("ContinousRegion", color);
	}
}

#endif /* CUT2_H_ */
