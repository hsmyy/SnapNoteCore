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
#include "../util/util.h"

using namespace std;

class Workflow {
public:
	void workflow(string &inputFile);
	void workflowDebug(string &inputFile);
	void workflowTrace(string &inputFolder, string &salientFolder,
			string &borderFolder, string &textFolder);
private:
};

void Workflow::workflowTrace(string &inputFolder, string &salientFolder,
		string &borderFolder, string &textFolder) {
	vector<string> inputFiles = dir(inputFolder);
	for (unsigned i = 0, len = inputFiles.size(); i < len; ++i) {
		string inputFile(inputFolder + inputFiles[i]), salientFile(
				salientFolder + inputFiles[i]), borderFile(
				borderFolder + inputFiles[i]), textFile(
				textFolder + inputFiles[i]);
		Mat input = imread(inputFile);
		SalientRec src;
		Mat segSRC, outputSRC, crossBD, outputBD;
		src.salient(input, outputSRC, segSRC);

		imwrite(salientFile, outputSRC);
		int res = mainProc(input, outputSRC, 0, crossBD, outputBD);

		if (res != -1) {
			imwrite(borderFile, crossBD);
			Mat denoise, bin, deskew;
			Denoise::saltPepperDenoise(outputBD, denoise);
			Binarize::binarize(denoise, bin);
			Deskew::deskew(bin, deskew);
			Binarize::binarize(deskew, bin);
			string text = OCRUtil::ocrFile(bin, "eng");
			saveToFile(textFile, text);
		}
	}
}

void Workflow::workflow(string &inputFile) {
	Mat input = imread(inputFile);
	SalientRec src;
	Mat segSRC, outputSRC, crossBD, outputBD;
	src.salient(input, outputSRC, segSRC);

	int res = mainProc(input, outputSRC, 0, crossBD, outputBD);

	if (res != -1) {
		Mat denoise, bin, deskew;
		Denoise::saltPepperDenoise(outputBD, denoise);
		Binarize::binarize(denoise, bin);
		Deskew::deskew(bin, deskew);
		Binarize::binarize(deskew, bin);
		string text = OCRUtil::ocrFile(bin, "eng");
		cout << text << endl;
	}
}

void Workflow::workflowDebug(string &inputFile) {
	Mat input = imread(inputFile);
	SalientRec src(true);
	Mat segSRC, outputSRC, crossBD, outputBD;
	src.salient(input, outputSRC, segSRC);
	int res = mainProc(input, outputSRC, 0, crossBD, outputBD);

	if (res != -1) {
		//TODO preprocess
		namedWindow("crossBD");
		imshow("crossBD", crossBD);
		namedWindow("outputBD");
		imshow("outputBD", outputBD);

		Mat denoise, bin, deskew;
		Denoise::saltPepperDenoise(outputBD, denoise);
		Binarize::binarize(denoise, bin);
		Deskew::deskew(bin, deskew);
		Binarize::binarize(deskew, bin);
		string text = OCRUtil::ocrFile(bin, "eng");
		cout << text << endl;
	} else {
		cout << "border detection failed" << endl;
	}
}

#endif /* WORKFLOW_WORKFLOW_H_ */
