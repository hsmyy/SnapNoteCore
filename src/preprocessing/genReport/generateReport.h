/*
 * generateReport.h
 *
 *  Created on: Jan 13, 2015
 *      Author: xxy
 */

#ifndef PREPROCESSING_SRC_GENERATEREPORT_H_
#define PREPROCESSING_SRC_GENERATEREPORT_H_

#include <vector>
#include "../utils/FileUtil.h"

using namespace std;

class GenReport {
public:
	static const string UNLV_HOME =
			"/home/xxy/Desktop/tesseract-ocr/testing/unlv/";

	static void generateReport(const char* inputDir, const char* outputDir) {
		vector<string> files = FileUtil::getAllFiles(inputDir);
		for (int i = 0; i < files.size(); i++) {
			cout << "report file :" << files[i] << endl;
			string gtPath = string(inputDir) + "/"
					+ FileUtil::getFileNameNoSuffix(files[i]) + ".txt";
			string newPath = string(outputDir) + "/"
					+ FileUtil::getFileNameNoSuffix(files[i]) + ".txt";
			string chReport = string(outputDir) + "/"
					+ FileUtil::getFileNameNoSuffix(files[i]) + ".acc";
			string waReport = string(outputDir) + "/"
					+ FileUtil::getFileNameNoSuffix(files[i]) + ".wa";
			string chaccuCmd = UNLV_HOME + "accuracy " + gtPath + " " + newPath
					+ " " + chReport;
			cout << "cmd: " << chaccuCmd << endl;
			string waccuCmd = UNLV_HOME + "wordacc " + gtPath + " " + newPath
					+ " " + waReport;
			system(chaccuCmd.c_str());
			system(waccuCmd.c_str());
		}
	}
};

#endif /* PREPROCESSING_SRC_GENERATEREPORT_H_ */
