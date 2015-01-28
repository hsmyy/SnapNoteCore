/*
 * main.cpp
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

//#include "salientRecognition/execute.h"
//#include "salientRecognition/rc/rcLearning.h"
#include "workflow/workflow.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
//	salientDebug("test/input/imaget3.png");
//	salient("test/input/book3.jpg","test/seg/book3.jpg","test/output/book3.jpg");
//	wholeTest();
//	waitKey(0);
//	learn();
	Workflow workflow;
	string input("test/workflow/input/"), border("test/workflow/border/"),
			salient("test/workflow/salient/"),text("test/workflow/text");
	workflow.workflowTrace(input,salient,border,text);
//	string input("test/SalientRec/input/image7.jpg");
//	workflow.workflowDebug(input);
//	waitKey();
	return 0;
}

