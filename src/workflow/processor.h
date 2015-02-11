/*
 * processor.h
 *
 *  Created on: Feb 3, 2015
 *      Author: xxy
 */

#ifndef IMAGE_PROCESS_SRC_WORKFLOW_PROCESSOR_H_
#define IMAGE_PROCESS_SRC_WORKFLOW_PROCESSOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "../util/configUtil.h"
#include "../salientRecognition/execute.h"
#include "../preprocessing/utils/FileUtil.h"
#include "../borderPosition/border.h"
#include "../textDetect/textarea.h"
#include "../preprocessing/binarize/binarize.h"
#include "../preprocessing/deskew/deskew.h"
#include "../preprocessing/GaussianSPDenoise/denoise.h"
#include "../preprocessing/utils/OCRUtil.h"
#include "../preprocessing/cca/CCA.h"

using namespace std;
using namespace cv;

/*
 * -s Single image mode.
 * -d Directory images mode.
 * -i Input file or input directory (depends on mode).
 * -o OCR output directory.
 * -c Configuration file path, (method = directory). see sn.conf as an example.
 * ex. ./image_process -s -i "test/workflow/input/ad1.jpg" -o "test/workflow/ocr" -c sn.conf
 */

class Processor {
public:
	const static string SEG;
	const static string SALIENT;
	const static string BORDER;
	const static string TURN;
	const static string TEXT;
	const static string BINARIZE;
	const static string DENOISE;
	const static string DESKEW;
	const static string CCA;

	static string lang;

	static void usage() {
		cout << "Please add parameters:" << endl;
		cout << " -s Single image mode." << endl;
		cout << " -d Directory images mode." << endl;
		cout << " -i Input file or input directory (depends on mode)." << endl;
		cout << " -o OCR output directory." << endl;
		cout
				<< " -c Configuration file path, (method = directory). see sn.conf as an example."
				<< endl;

	}
	static void process_main(int argc, char** argv) {
		int oc; /*选项字符 */
		char ec; /*无效的选项字符*/

		bool singleMode = true;
		string input;
		string ocrOutput;
		string configPath;

		cout << "Read parameters..." << endl;

		while ((oc = getopt(argc, argv, "sdi:o:c:")) != -1) {
			switch (oc) {
			case 's':
				printf("Single mode.\n");
				singleMode = true;
				break;
			case 'd':
				printf("Directory mode.\n");
				singleMode = false;
				break;
			case 'i':
				printf("Input path is %s\n", optarg);
				input = optarg;
				break;
			case 'o':
				printf("OCR output path is %s\n", optarg);
				ocrOutput = optarg;
				break;
			case 'c':
				printf("Config file path is %s\n", optarg);
				configPath = optarg;
				break;
			case '?':
				ec = (char) optopt;
				printf("Invalid option \' %c \'!\n", ec);
				break;
			case ':':
				printf("Lack option！\n");
				break;
			}
		}
		if (input.empty() && configPath.empty()) {
			usage();
			return;
		}

		Config config(configPath);

		char pattern[512] = "[^a-zA-Z0-9]+";
		if (singleMode) {
			vector<Mat> dsts;
			if (!input.empty())
				dsts = Processor::processFile(input, config);

			if (!ocrOutput.empty()) {
				string textPath = ocrOutput + "/"
						+ FileUtil::getFileNameNoSuffix(input) + ".txt";
				cout << "OCR to: " << textPath << endl;

				string text = ocrMats(dsts);

				FileUtil::writeToFile(text, textPath);
			}
		} else {
			if (!input.empty()) {
				vector<string> files = FileUtil::getAllFiles(input);
				for (int i = 0; i < files.size(); i++) {
					vector<Mat> mats = processFile(input + "/" + files[i],
							config);
					string text = ocrMats(mats);
					string textPath = ocrOutput + "/"
							+ FileUtil::getFileNameNoSuffix(files[i]) + ".txt";
					FileUtil::writeToFile(text, textPath);
				}
			}

		}
	}

	static string ocrMats(vector<Mat>& mats) {
		ostringstream os;
		for (unsigned i = 0; i < mats.size(); i++) {
			os << OCRUtil::ocrFile(mats[i], lang) << endl;
		}
		return os.str();
	}

	static vector<Mat> processFile(string input, const Config conf) {
		Config config = conf;
		Mat img = imread(input);
		cout << "Process " << input << endl;
		string segOut = config.getAndErase(SEG);
		string salientOut = config.getAndErase(SALIENT);
		string borderOut = config.getAndErase(BORDER);
		string turnOut = config.getAndErase(TURN);
		string textOut = config.getAndErase(TEXT);
		if (salientOut.empty() || borderOut.empty()) {
			cerr
					<< "salient output or border output is empty. (in config file)!"
					<< endl;
			return img;
		}

		SalientRec src;
		Mat outputSRC, seg, crossBD, outputBD;
		string salientOutPath = salientOut + "/" + FileUtil::getFileName(input);
		string segOutPath = segOut + "/" + FileUtil::getFileName(input);

		cout << "salient object..." << endl;
		src.salient(img, outputSRC, seg);
		Mat outputFileSRC = convertToVisibleMat<float>(outputSRC);
		imwrite(segOutPath, seg);
		imwrite(salientOutPath, outputFileSRC);
		//cout<<outputSRC(Rect(0, 0, 500, 500))<<endl;

		int res = mainProc(img, outputSRC, 0, crossBD, outputBD);
		if (res == -1)
			res = procBinary(img, outputSRC, 0, crossBD, outputBD);

		string borderOutPath = borderOut + "/" + FileUtil::getFileName(input);
		string turnOutPath = turnOut + "/" + FileUtil::getFileName(input);

//		imshow("cross",crossBD);
//		waitKey();
		imwrite(borderOutPath, crossBD);
		normalize(outputBD, outputBD, 0, 255, NORM_MINMAX);
		outputBD.convertTo(outputBD, CV_8UC1);
		imwrite(turnOutPath, outputBD);

		cout<<"text detection"<<endl;
		vector<Mat> textPieces;
		textDetect(outputBD, textPieces, res == -1 ? false : true);

		//TODO process all the text pieces!

		cout << "Preprocessing..." << endl;
		vector<Mat> pre = vector<Mat>(textPieces.size());
		for (unsigned int i = 0; i < pre.size(); i++) {
			cvtColor(textPieces[i], pre[i], COLOR_BGR2GRAY);
		}

		for (int i = 0; i < config.size(); i++) {
			vector<Mat> cur;
			pair<string, string> step = config.get(i);
			void (*process)(vector<Mat>&, vector<Mat>&) = getMethod(step.first);

			string outputPath = step.second + "/"
					+ FileUtil::getFileName(input);
			process(pre, cur);
			Mat all = merge(cur);
			imwrite(outputPath, all);
			pre = cur;
		}

		return pre;
	}
	static Mat merge(vector<Mat>& mats) {
		int width = maxWidth(mats);
		int height = totalHeight(mats);
		int index = 0;
		Mat dst(height, width, CV_8UC1);
		for (unsigned int i = 0; i < mats.size(); i++) {
			Mat roi = dst(Rect(0, index, mats[i].cols, mats[i].rows));
			mats[i].copyTo(roi, mats[i]);
			index += mats[i].rows;
		}
		return dst;
	}
	static int maxWidth(vector<Mat>& mats) {
		int width = 0;
		for (unsigned int i = 0; i < mats.size(); i++) {
			if (width < mats[i].cols)
				width = mats[i].cols;
		}
		return width;
	}
	static int totalHeight(vector<Mat>& mats)
	{
		int height = 0;
		for (unsigned int i = 0; i < mats.size(); i++) {
			height += mats[i].rows;
		}
		return height;
	}
	static void processDir(string input, const Config conf) {
		vector<string> files = FileUtil::getAllFiles(input);
		for (int i = 0; i < files.size(); i++) {
			vector<Mat> mats = processFile(input + "/" + files[i], conf);
		}
	}
	static void (*getMethod(string methodName))(vector<Mat>&, vector<Mat>&)
			{
				if (methodName == BINARIZE) {
					return Binarize::binarizeSet;
				} else if (methodName == DENOISE) {
					return Denoise::denoiseSet;
				} else if (methodName == DESKEW) {
					return Deskew::deskewSet;
				} else if (methodName == CCA) {
					return CCA::removeGarbageSet;
				} else
					return NULL;
			}

		}
		;

		const string Processor::SEG = "seg";
		const string Processor::SALIENT = "salient";
		const string Processor::BORDER = "border";
		const string Processor::TURN = "turn";
		const string Processor::TEXT = "text";
		const string Processor::BINARIZE = "binarize";
		const string Processor::DENOISE = "denoise";
		const string Processor::DESKEW = "deskew";
		const string Processor::CCA = "cca";

		string Processor::lang = "eng";

#endif /* IMAGE_PROCESS_SRC_WORKFLOW_PROCESSOR_H_ */
