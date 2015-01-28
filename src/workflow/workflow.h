/*
 * workflow.h
 *
 *  Created on: Jan 28, 2015
 *      Author: fc
 */

#ifndef WORKFLOW_WORKFLOW_H_
#define WORKFLOW_WORKFLOW_H_

#include <iostream>
#include "../salientRecognition/execute.h"
#include "../borderPosition/border.h"
#include "../preprocessing/GaussianSPDenoise/denoise.h"
#include "../preprocessing/deskew/deskew.h"
#include "../preprocessing/binarize/binarize.h"
#include "../preprocessing/utils/OCRUtil.h"


using namespace std;

class Workflow{
public:
	void workflow(string &inputFile);
	void workflowDebug(string &inputFile);
	void workflowTrace(string &inputFolder,
			string &salientFolder,
			string &borderFolder);
private:
};

void Workflow::workflowTrace(string &inputFile,		string &salientFolder,		string &borderFolder){
	Mat input = imread(inputFile);
}

void Workflow::workflow(string &inputFile){
	Mat input = imread(inputFile);
	SalientRec src;
	Mat outputSRC, crossBD, outputBD;
	src.salient(input, outputSRC);
	int res;
	if(src.isResultUseful(outputSRC)){
		res = mainProc(outputSRC, 0, crossBD, outputBD);
	}else{
		res = mainProc(input, 0, crossBD, outputBD);
	}

	if(res == -1){
		Mat denoise, bin, deskew;
		Denoise::saltPepperDenoise(outputBD, denoise);
		Binarize::binarize(denoise, bin);
		Deskew::deskew(bin, deskew);
		Binarize::binarize(deskew, bin);
		string text = OCRUtil::ocrFile(bin, "eng");
		cout<<text<<endl;
	}



}

void Workflow::workflowDebug(string &inputFile){
	Mat input = imread(inputFile);
	SalientRec src;
	Mat outputSRC, crossBD, outputBD;
	src.salientDebug(input, outputSRC);
	int res;
	if(src.isResultUseful(outputSRC)){
		res = mainProc(outputSRC, 0, crossBD, outputBD);
	}else{
		res = mainProc(input, 0, crossBD, outputBD);
	}
	if(res != -1){
		//TODO preprocess
		namedWindow("crossBD");
		imshow("crossBD", crossBD);
		namedWindow("outputBD");
		imshow("outputBD", outputBD);
		waitKey(0);

		Mat denoise, bin, deskew;
		Denoise::saltPepperDenoise(outputBD, denoise);
		Binarize::binarize(denoise, bin);
		Deskew::deskew(bin, deskew);
		Binarize::binarize(deskew, bin);
		string text = OCRUtil::ocrFile(bin, "eng");
		cout<<text<<endl;
	}
	else{
		cout<<"border detection failed"<<endl;
	}
}


#endif /* WORKFLOW_WORKFLOW_H_ */
