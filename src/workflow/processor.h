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
#include "../preprocessing/binarize/binarize.h"
#include "../preprocessing/deskew/deskew.h"
#include "../preprocessing/GaussianSPDenoise/denoise.h"
#include "../preprocessing/utils/OCRUtil.h"

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
	const static string BINARIZE;
	const static string DENOISE;
	const static string DESKEW;


	static string lang ;

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
			Mat dst;
			if (!input.empty())
				dst = Processor::processFile(input, config);

			if (!ocrOutput.empty()) {
				string textPath = ocrOutput + "/"
						+ FileUtil::getFileNameNoSuffix(input) + ".txt";
				cout << "OCR to: " << textPath << endl;

				string text = OCRUtil::ocrFile(dst, lang);
				FileUtil::writeToFile(text, textPath);
			}
		} else {
			if (!input.empty())
				Processor::processDir(input, config);
			if (!ocrOutput.empty() && config.size() > 0) {

				OCRUtil::ocrDir(config.get(config.size() - 1).second, ocrOutput,lang);
			}
		}
	}

	static Mat processFile(string input, const Config conf) {
		Config config = conf;
		Mat img = imread(input);
		cout << "Process " << input << endl;
		string segOut = config.getAndErase(SEG);
		string salientOut = config.getAndErase(SALIENT);
		string borderOut = config.getAndErase(BORDER);
		string turnOut = config.getAndErase(TURN);
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

//		int res;
//		if (src.isResultUseful(outputSRC)) {
		int res = mainProc(img, outputSRC, 0, crossBD, outputBD);
//		} else {
//			res = mainProc(img, outputSRC, 0, crossBD, outputBD);
//		}

		string borderOutPath = borderOut + "/" + FileUtil::getFileName(input);
		string turnOutPath = turnOut + "/" + FileUtil::getFileName(input);

		imwrite(borderOutPath, crossBD);
		normalize(outputBD, outputBD, 0, 255, NORM_MINMAX);
		outputBD.convertTo(outputBD, CV_8UC1);
		imwrite(turnOutPath, outputBD);

		if (res == -1)
			outputBD = img;

		cout << "Preprocessing..." << endl;
		Mat pre = outputBD;
		cvtColor(outputBD, pre, COLOR_BGR2GRAY);
		//pre.convertTo(pre, CV_8UC1);

//		imshow("output", outputBD);
//		waitKey(0);
//		imwrite("test/img.jpg", outputBD);
//		cout<<outputBD<<endl;

		for (int i = 0; i < config.size(); i++) {
			Mat cur;
			pair<string, string> step = config.get(i);
			void (*process)(Mat& src, Mat& dst) = getMethod(step.first);
			string outputPath = step.second + "/"
					+ FileUtil::getFileName(input);
			process(pre, cur);
			imwrite(outputPath, cur);
			pre = cur;
		}

		return pre;
	}
	static void processDir(string input, const Config conf) {
		vector<string> files = FileUtil::getAllFiles(input);
		for (int i = 0; i < files.size(); i++) {
			processFile(input + "/" + files[i], conf);
		}
	}
	static void (*getMethod(string methodName))(Mat&, Mat&)
			{
				if (methodName == BINARIZE) {
					return Binarize::binarize;
				} else if (methodName == DENOISE) {
					return Denoise::denoise;
				} else if (methodName == DESKEW) {
					return Deskew::deskew;
				} else
					return NULL;
			}

		};

		const string Processor::SEG = "seg";
		const string Processor::SALIENT = "salient";
		const string Processor::BORDER = "border";
		const string Processor::TURN = "turn";
		const string Processor::BINARIZE = "binarize";
		const string Processor::DENOISE = "denoise";
		const string Processor::DESKEW = "deskew";
		string Processor::lang = "eng";


#endif /* IMAGE_PROCESS_SRC_WORKFLOW_PROCESSOR_H_ */
