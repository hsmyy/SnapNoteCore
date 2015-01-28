/*
 * rcLearning.h
 *
 *  Created on: Jan 28, 2015
 *      Author: fc
 */

#ifndef SALIENTRECOGNITION_RC_RCLEARNING_H_
#define SALIENTRECOGNITION_RC_RCLEARNING_H_

#include <iostream>
#include "main.h"
#include "rc.h"
#include "quantize.h"
#include "../segmentation/segment-image.h"

using namespace std;

class rcLearning{
public:

	/**
	 * we have an assumption that input and output file has the same name but different suffix.
	 */
	void learningToRank(string &inputFolder, string &outputFolder);
	void learningToRankInDetail(string &inputFile, string &outputFile);
	void splitSalientGroups(Mat &seg, Mat &output, int segNum, vector<int> &salientIdices, vector<int> &nonSalientIdices);
	void rcTraining(Mat &input, Mat &seg, int segNum, vector<int> salientIdices, vector<int> nonSalientIdices);
private:
	RegionContrastSalient rcs;
	int correctNum;
	int testNum;
	float lastPrecision;
	float curPrecision;
};

void rcLearning::rcTraining(Mat &input, Mat &seg, int segNum, vector<int> salientIdices, vector<int> nonSalientIdices){
	// 3) precompute something
	Mat colorIdx1i, regSal1v, tmp, color3fv;
	input.convertTo(input, CV_32FC3, 1.0/255);
	Quantizer quantizer;
	quantizer.Quantize(input, colorIdx1i, color3fv, tmp);

	cvtColor(color3fv, color3fv, CV_BGR2Lab);
	vector<Region> regs(segNum);

	rcs.BuildRegions(seg, regs, colorIdx1i, color3fv.cols);
	Mat_<float> cDistCache1f = pairwiseColorDist(color3fv);

	Mat_<double> rDistCache1d = Mat::zeros(segNum, segNum, CV_64F);
	Mat regionSalientScore = Mat::zeros(1, segNum, CV_64F);
	double* regSal = (double*)regionSalientScore.data;
	for (int i = 0; i < segNum; i++){
		const Point2d &rc = regs[i].centroid;
		for (int j = 0; j < segNum; j++){
			if(i<j) {
				double dd = 0;
				const vector<CostfIdx> &c1 = regs[i].freIdx, &c2 = regs[j].freIdx;
				for (size_t m = 0; m < c1.size(); m++){
					for (size_t n = 0; n < c2.size(); n++){
						dd += cDistCache1f[c1[m].second][c2[n].second] * c1[m].first * c2[n].first;
					}
				}
				rDistCache1d[j][i] = rDistCache1d[i][j] = dd * exp(-pntSqrDist(rc, regs[j].centroid)/rcs.getSigmaDist());
			}
			regSal[i] += regs[j].pixNum * rDistCache1d[i][j];
		}
//		regSal[i] *= exp(_regionWeight * sqrt(regs[i].pixNum / (float)pixelNum) - _distanceWeight * (sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y)));
	}
	// 4) for each pair, compare the score and update the parameter
	float pixelNum = input.rows * input.cols;
	for(unsigned i = 0, ilen = salientIdices.size(); i < ilen; ++i){
		for(unsigned j = 0, jlen = nonSalientIdices.size(); j < jlen; ++j){
			double regionScoreI = sqrt(regs[i].pixNum / pixelNum);
			double regionScoreJ = sqrt(regs[j].pixNum / pixelNum);
			double distScoreI = sqr(regs[i].ad2c.x) + sqr(regs[i].ad2c.y);
			double distScoreJ = sqr(regs[i].ad2c.x) + sqr(regs[j].ad2c.y);

			double scoreI = regSal[i] * exp(rcs.getRegionWeight() * regionScoreI - rcs.getDistanceWeight() * distScoreI);
			double scoreJ = regSal[j] * exp(rcs.getRegionWeight() * regionScoreJ - rcs.getDistanceWeight() * distScoreJ);
			double diff = scoreI - 10 * scoreJ;
			double updateRegionWeight = 0.005 * diff * (regionScoreI - regionScoreJ);
			double updateDistWeight = 0.005 * diff * (distScoreI - distScoreJ);

			if(updateRegionWeight > 0.05){
				updateRegionWeight = 0.05;
				if(updateRegionWeight < -0.05){
					updateRegionWeight = - 0.05;
				}
			}

			if(updateDistWeight > 0.05){
				updateDistWeight = 0.05;
				if(updateDistWeight < -0.05){
					updateDistWeight = -0.05;
				}
			}

			rcs.updateDistanceWeight(updateDistWeight);
			rcs.updateRegionWeight(updateRegionWeight);

			if(diff > 0){
				++correctNum;
			}
		}
	}
	testNum += (int)salientIdices.size() * (int)nonSalientIdices.size();
	cout << "Precision:" << (correctNum / (float)testNum) << endl;
}

void rcLearning::learningToRank(string &inputFolder, string &outputFolder){
	vector<string> inputFiles = listFiles(inputFolder.c_str());
//	vector<string> outputFiles = listFiles(outputFolder.c_str());
	//train times
	for(int times = 0; times < 50; ++times){
		correctNum = 0;
		testNum = 0;
		for(unsigned int i = 0, len = inputFiles.size(); i < len; ++i){
			size_t idx = inputFiles[i].find_last_of('.');
			if(idx != string::npos){
				string outputFilePath = outputFolder + inputFiles[i].substr(0, idx) + ".png";
				string inputFilePath(inputFolder + inputFiles[i]);
				learningToRankInDetail(inputFilePath, outputFilePath);
			}
		}
		lastPrecision = curPrecision;
		curPrecision = correctNum / (float)testNum;
		cout << "[Iteration " << times << "]" << curPrecision << (curPrecision - lastPrecision) << endl;
		cout << "Region:" << rcs.getRegionWeight() << ",Dist:" << rcs.getDistanceWeight() << endl;
	}
}

void rcLearning::learningToRankInDetail(string &inputFile, string &outputFile){
	cout << inputFile << "," << outputFile << endl;
	Mat input = imread(inputFile.c_str());
	Mat output = imread(outputFile.c_str());

	// 1) do segmentation
	GraphSegmentation segmentation;
	Mat seg;
	int segNum = segmentation.segment_image(input, seg);

	// 2) split the segment into 2 groups: salient or non-salient
	vector<int> salientIdices;
	vector<int> nonSalientIdices;

	splitSalientGroups(seg, output, segNum, salientIdices, nonSalientIdices);

	rcTraining(input, seg, segNum, salientIdices, nonSalientIdices);
}

void rcLearning::splitSalientGroups(Mat &seg, Mat &output, int segNum, vector<int> &salientIdices, vector<int> &nonSalientIdices){
	vector<pair<int, int> > segInfos(segNum);
	for(int y = 0; y < output.rows; ++y){
		int *segRow = seg.ptr<int>(y);
		Vec3b *outputRow = seg.ptr<Vec3b>(y);
		for(int x = 0; x < output.cols; ++x, ++segRow, ++outputRow){
			if(outputRow->val[0] > 0 || outputRow->val[1] > 0 || outputRow->val[2] > 0){
				segInfos[*segRow].first += 1;
			}else{
				segInfos[*segRow].second += 1;
			}
		}
	}
	for(unsigned i = 0, len = segInfos.size(); i < len; ++i){
		pair<int, int> *curPair = &segInfos[i];
		float sumPixel = curPair->first + curPair->second;
		if(curPair->first / sumPixel > 0.6){
			salientIdices.push_back(i);
		}else{
			nonSalientIdices.push_back(i);
		}
	}
}

void learn(){
	rcLearning rcl;
	string inputFolder("/home/fc/Downloads/MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/input/"), outputFolder("/home/fc/Downloads/MSRA10K_Imgs_GT/MSRA10K_Imgs_GT/output/");
	rcl.learningToRank( inputFolder, outputFolder);
}


#endif /* SALIENTRECOGNITION_RC_RCLEARNING_H_ */
