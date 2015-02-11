/*
 * lineFormation.h
 *
 *  Created on: Feb 9, 2015
 *      Author: fc
 */

#ifndef LINEFORMATION_H_
#define LINEFORMATION_H_

#include <opencv2/opencv.hpp>
#include "ConnectedComponent.h"
#include <list>
#include <iostream>

using namespace cv;
using namespace std;

inline bool isInRegion(int y, int x, Rect &region){
	return x > region.x && x < region.x + region.width && y > region.y && y < region.y + region.height;
}

inline bool isIntersect(Rect &rect1, Rect &rect2){
	return isInRegion(rect1.y, rect1.x, rect2) || isInRegion(rect1.y, rect1.x + rect1.width, rect2) ||
			isInRegion(rect1.y + rect1.height, rect1.x, rect2) || isInRegion(rect1.y + rect1.height, rect1.x + rect1.width, rect2);

}

Rect clamp(Rect &rect, Mat img1i){
	rect.x = rect.x >= 0 ? rect.x : 0;
	rect.y = rect.y >= 0 ? rect.y : 0;
	rect.width = rect.x + rect.width < img1i.cols ? rect.width : img1i.cols - rect.x;
	rect.height = rect.y + rect.height < img1i.rows ? rect.height : img1i.rows - rect.y;
	return rect;
}

typedef struct swipeline{
	int x;
	int ystart;
	int yend;
	int type;//start == 0, end == 1

	swipeline(int xx, int ys, int ye, int type){
		this->x = xx;
		this->ystart = ys;
		this->yend = ye;
		this->type = type;
	}
} swipeline;

bool compareSwipeline(const swipeline &line1 , const swipeline &line2){
	return (line1.x < line2.x) || (line1.x == line2.x && line1.ystart < line2.ystart);
}

double intersectRatio(Rect &rect1, Rect &rect2){
	vector<swipeline> lines;
	lines.push_back(swipeline(rect1.x, rect1.y, rect1.y + rect1.height, 0));
	lines.push_back(swipeline(rect1.x + rect1.width, rect1.y, rect1.y + rect1.height, 1));
	lines.push_back(swipeline(rect2.x, rect2.y, rect2.y + rect2.height, 0));
	lines.push_back(swipeline(rect2.x + rect2.width, rect2.y, rect2.y + rect2.height, 1));

	sort(lines.begin(), lines.end(), compareSwipeline);
	int curX = -1;
	int sum = 0;
	vector<pair<int, int> > yPairs(3);// yPairs(0) for the total length
	for(unsigned int i = 0; i < 3; ++i){
		yPairs[i] = make_pair(-1,-1);
	}
	for(unsigned int i = 0, len = lines.size(); i < len; ++i){
		//cal segment area
		if(curX >= 0){
			sum += (lines[i].x - curX) * (yPairs[0].second - yPairs[0].first);
		}
		curX = lines[i].x;
		if(lines[i].type){//end, remove one and update the total
			for(unsigned int j = 1, len = yPairs.size(); j < len; ++j){
				if(yPairs[j].first == lines[i].ystart && yPairs[j].second == lines[i].yend){
					yPairs[j].first = -1;
					yPairs[j].second = -1;
					yPairs[0].first = yPairs[3 - j].first;
					yPairs[0].second = yPairs[3 - j].second;
					break;
				}
			}
		}else{//start,
			if(yPairs[1].first != -1){
				yPairs[2].first = lines[i].ystart;
				yPairs[2].second = lines[i].yend;
				yPairs[0].first = min(yPairs[1].first, yPairs[2].first);
				yPairs[0].second = max(yPairs[1].second, yPairs[2].second);
			}else{
				yPairs[0].first = yPairs[1].first = lines[i].ystart;
				yPairs[0].second = yPairs[1].second = lines[i].yend;
			}
		}
	}
	int area1 = rect1.height * rect1.width;
	int area2 = rect2.height * rect2.width;
	int intersection = area1 + area2 - sum;
	return max(intersection / (double)area1, intersection / (double)area2);
}

class LineFormation{
public:
	vector<Rect> findLines(Mat &img1i);
private:
	double dotLineDistance(Point2f start, Point2f end, Point2f dot);
	double dotAngle(Point2f first, Point2f second);
	vector<ComponentProperty> pruneSoleRegion(vector<ComponentProperty> candidateRegions);
	vector<Rect> mergeRegions(vector<Rect> candidateRegions, Mat img1i);
};

double LineFormation::dotLineDistance(Point2f start, Point2f end, Point2f dot){
	end -= start;
	dot -= start;
	double area = end.y * dot.x + end.x * dot.y;
	double len = sqrt(end.y * end.y + end.x * end.x);
	return area / len;
}

double LineFormation::dotAngle(Point2f first, Point2f second){
	return abs((first.y - second.y) / (first.x - second.x));
}

vector<Rect> LineFormation::mergeRegions(vector<Rect> candidateRegions, Mat img1i){
	// the regions have already sorted by weight.
	vector<Rect> regions;
	if(candidateRegions.size() > 0){
		regions.push_back(candidateRegions[0]);
	}
	for(unsigned int i = 1; i < candidateRegions.size(); ++i){
		bool keepFlag = true;
		for(unsigned int j = 0; j < regions.size(); ++j){
			Rect &candidateRect = candidateRegions[i];
			Rect &choosedRect = regions[j];
			if(isIntersect(candidateRect, choosedRect) ||
					isIntersect(choosedRect, candidateRect)){
				if(intersectRatio(candidateRect, choosedRect) > 0.6){
					//merge, set flag and break
					int minX = min(candidateRect.x, choosedRect.x);
					int maxX = max(candidateRect.x + candidateRect.width, choosedRect.x + choosedRect.width);
					int minY = min(candidateRect.y, choosedRect.y);
					int maxY = max(candidateRect.y + candidateRect.height, choosedRect.y + choosedRect.width);
					regions[j].x = minX;
					regions[j].y = minY;
					regions[j].width = maxX - minX;
					regions[j].height = maxY - minY;
					regions[j] = clamp(regions[j], img1i);
					keepFlag = false;
					break;
				}
			}
		}
		if(keepFlag){
			regions.push_back(candidateRegions[i]);
		}
	}
	return regions;
}

vector<ComponentProperty> LineFormation::pruneSoleRegion(vector<ComponentProperty> candidateRegions){
	vector<pair<double, unsigned int> > score(candidateRegions.size());
	unsigned int len = candidateRegions.size();
	double max = -1;
	for(unsigned int i = 0; i < len; ++i ){
		double sum = 0;
		for(unsigned int j = 0; j < len; ++j){
			sum += sqrt(square(candidateRegions[i].centroid.x - candidateRegions[j].centroid.x) +
					square(candidateRegions[i].centroid.y - candidateRegions[j].centroid.y));
		}
		if(max < sum){
			max = sum;
		}
		score[i] = make_pair(sum, i);
	}
	vector<ComponentProperty> connected;
	cout << "sole check" << endl;
	//discard last 10%
	for(unsigned int i = 0; i < len; ++i){
		score[i].first /= max;
		if(score[i].first < 0.9){
			connected.push_back(candidateRegions[i]);
		}
	}

	return connected;
}

vector<Rect> LineFormation::findLines(Mat &img1i){
	ConnectedComponent conn_comp( 10000, 8);
	Mat labelImg = conn_comp.apply(img1i);
	vector<ComponentProperty> candidateProps = conn_comp.getComponentsProperties();
	vector<ComponentProperty> props = pruneSoleRegion(candidateProps);

	vector<Rect> textLines;

	vector<vector<int> > choosedIdx(2);
	int max = 1;
	while(props.size() > 3){
		unsigned int len = props.size();
		for(unsigned int i = 0; i < len; ++i){
			for(unsigned int j = i + 1; j < len; ++j){
				ComponentProperty outComponent = props[i];
				ComponentProperty innerComponent = props[j];
				//TODO only consider horizontal text
				if(abs(outComponent.centroid.y - innerComponent.centroid.y) < 15 &&
						dotAngle(outComponent.centroid, innerComponent.centroid) < 0.3 &&
						abs(outComponent.centroid.x - innerComponent.centroid.x) < 50){
					//calculate window size
					int space = outComponent.boundingBox.height > innerComponent.boundingBox.height ?
							outComponent.boundingBox.height / 2 : innerComponent.boundingBox.height / 2;
					// go through
					choosedIdx[1 - max].clear();
					for(unsigned int k = 0; k < len; ++k){
						if(k != i && k != j &&
								abs(dotLineDistance(outComponent.centroid, innerComponent.centroid, props[k].centroid)) < space){
							choosedIdx[1 - max].push_back(k);
						}
					}
					if(choosedIdx[1 - max].size() > 3 && choosedIdx[1 - max].size() + 2 > choosedIdx[max].size()){
						max = 1 - max;
						choosedIdx[max].push_back(i);
						choosedIdx[max].push_back(j);
					}
				}
			}
		}
		if(choosedIdx[max].size() == 0){
			break;
		}
		//draw rect and remove
		int minX = 10000000,minY = 100000000,maxX = -1,maxY = -1;
		for(unsigned int l = 0; l < choosedIdx[max].size(); ++l){
			int idx = choosedIdx[max][l];
			if(props[idx].boundingBox.x < minX){
				minX = props[idx].boundingBox.x;
			}
			if(props[idx].boundingBox.x + props[idx].boundingBox.width > maxX){
				maxX = props[idx].boundingBox.x + props[idx].boundingBox.width;
			}
			if(props[idx].boundingBox.y < minY){
				minY = props[idx].boundingBox.y;
			}else if(props[idx].boundingBox.y + props[idx].boundingBox.height > maxY){
				maxY = props[idx].boundingBox.y + props[idx].boundingBox.height;
			}
		}
		Rect rect(minX - 10, minY - 10, maxX - minX + 20, maxY - minY + 20);
//		rect.x = rect.x >= 0 ? rect.x : 0;
//		rect.y = rect.y >= 0 ? rect.y : 0;
//		rect.width = rect.x + rect.width < img1i.cols ? rect.width : img1i.cols - rect.x;
//		rect.height = rect.y + rect.height < img1i.rows ? rect.height : img1i.rows - rect.y;
		rect = clamp(rect, img1i);
		textLines.push_back(rect);
		//clean used Region
		vector<ComponentProperty> nextRound;
		for(unsigned int l = 0; l < len; ++l){
			if(find(choosedIdx[max].begin(), choosedIdx[max].end(), l) == choosedIdx[max].end()){
				nextRound.push_back(props[l]);
			}
		}
		line(img1i,Point(props[choosedIdx[max][ choosedIdx.size() - 1]].centroid),
				Point(props[choosedIdx[max][ choosedIdx.size() - 2]].centroid),
				Scalar(255), 2, 8);
		props = nextRound;
		choosedIdx[max].clear();
	}
	namedWindow("textline");
	imshow("textline", img1i);
	//merge region twice
	return mergeRegions(mergeRegions(textLines, img1i), img1i);
}


#endif /* LINEFORMATION_H_ */
