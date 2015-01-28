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

using namespace std;

class Workflow{
public:
	void workflow(string &inputFile);
	void workflowDebug(string &inputFile);
private:
};

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

	if(res != -1){
		//TODO preprocess
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
	}
	else{
		cout<<"border detection failed"<<endl;
	}
}


#endif /* WORKFLOW_WORKFLOW_H_ */
