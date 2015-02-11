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

class LineFormation{
public:
	vector<Rect> findLines(Mat &img1i);
private:
	double dotLineDistance(Point2f start, Point2f end, Point2f dot);
	double dotAngle(Point2f first, Point2f second);
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

vector<Rect> LineFormation::findLines(Mat &img1i){
	ConnectedComponent conn_comp( 10000, 8);
	Mat labelImg = conn_comp.apply(img1i);
	vector<ComponentProperty> props = conn_comp.getComponentsProperties();

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
						dotAngle(outComponent.centroid, innerComponent.centroid) < 0.5 &&
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
		rect.x = rect.x >= 0 ? rect.x : 0;
		rect.y = rect.y >= 0 ? rect.y : 0;
		rect.width = rect.x + rect.width < img1i.cols ? rect.width : img1i.cols - rect.x;
		rect.height = rect.y + rect.height < img1i.rows ? rect.height : img1i.rows - rect.y;
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
		cout << nextRound.size() << endl;
	}
	namedWindow("textline");
	imshow("textline", img1i);
	return textLines;
}


#endif /* LINEFORMATION_H_ */
