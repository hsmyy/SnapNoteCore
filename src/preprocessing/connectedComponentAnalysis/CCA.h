/*
 * ccs.h
 *
 *  Created on: Jan 19, 2015
 *      Author: xxy
 */

#ifndef SRC_CCS_H_
#define SRC_CCS_H_

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;

class CCA
{
public:
	static Mat labelByTwoPass(const Mat& binImg, Mat& lableImg) {
		// connected component analysis (4-component)
		// use two-pass algorithm
		// 1. first pass: label each foreground pixel with a label
		// 2. second pass: visit each labeled pixel and merge neighbor labels
		//
		// foreground pixel: _binImg(x,y) = 1
		// background pixel: _binImg(x,y) = 0

		CV_Assert(binImg.type() == CV_8UC1);

		// 1. first pass

		binImg.convertTo(lableImg, CV_32SC1);

		lableImg = lableImg & 1;

		int label = 1;  // start by 256
		vector<int> labelSet;
		labelSet.push_back(0);   // background: 0
		labelSet.push_back(1);   // foreground: 1

		int rows = binImg.rows;
		int cols = binImg.cols;
		for (int i = 0; i < rows; i++) {
			int* data_preRow = (i >= 1 ? lableImg.ptr<int>(i - 1) : 0);
			int* data_curRow = lableImg.ptr<int>(i);
			for (int j = 0; j < cols; j++) {
				if (data_curRow[j] == 1) {
					vector<int> neighborLabels;
					neighborLabels.reserve(2);
					int leftPixel = (j >= 1 ? data_curRow[j - 1] : 0);
					int upPixel = (i >= 1 ? data_preRow[j] : 0);
					if (leftPixel > 1) {
						neighborLabels.push_back(leftPixel);
					}
					if (upPixel > 1) {
						neighborLabels.push_back(upPixel);
					}

					if (neighborLabels.empty()) {
						labelSet.push_back(++label);  // assign to a new label
						data_curRow[j] = label;
						labelSet[label] = label;
					} else {
						sort(neighborLabels.begin(), neighborLabels.end());
						int smallestLabel = neighborLabels[0];
						data_curRow[j] = smallestLabel;

						// save equivalence
						for (size_t k = 1; k < neighborLabels.size(); k++) {
							int tempLabel = neighborLabels[k];
							int& oldSmallestLabel = labelSet[tempLabel];
							if (oldSmallestLabel > smallestLabel) {
								labelSet[oldSmallestLabel] = smallestLabel;
								oldSmallestLabel = smallestLabel;
							} else {
								labelSet[smallestLabel] = oldSmallestLabel;
							}
						}
					}
				}
			}
		}

		// update equivalent labels
		// assigned with the smallest label in each equivalent label set
		for (size_t i = 2; i < labelSet.size(); i++) {
			int curLabel = labelSet[i];
			int preLabel = labelSet[curLabel];
			while (preLabel != curLabel) {
				curLabel = preLabel;
				preLabel = labelSet[preLabel];
			}
			labelSet[i] = curLabel;
		}

		// 2. second pass
		for (int i = 0; i < rows; i++) {
			int* data = lableImg.ptr<int>(i);
			for (int j = 0; j < cols; j++) {
				int& pixelLabel = data[j];
				pixelLabel = labelSet[pixelLabel];
			}
		}
		return lableImg;
	}
	static Scalar icvprGetRandomColor() {
		uchar r = 255 * (rand() / (1.0 + RAND_MAX));
		uchar g = 255 * (rand() / (1.0 + RAND_MAX));
		uchar b = 255 * (rand() / (1.0 + RAND_MAX));
		return Scalar(b, g, r);
	}

	static void labelColor(const Mat& _labelImg, Mat& _colorLabelImg) {
		if (_labelImg.empty() || _labelImg.type() != CV_32SC1) {
			return;
		}

		map<int, Scalar> colors;

		int rows = _labelImg.rows;
		int cols = _labelImg.cols;

		_colorLabelImg.release();
		_colorLabelImg.create(rows, cols, CV_8UC3);
		_colorLabelImg = Scalar::all(0);

		for (int i = 0; i < rows; i++) {
			const int* data_src = (int*) _labelImg.ptr<int>(i);
			uchar* data_dst = _colorLabelImg.ptr<uchar>(i);
			for (int j = 0; j < cols; j++) {
				int pixelValue = data_src[j];
				if (pixelValue > 1) {
					if (colors.count(pixelValue) <= 0) {
						colors[pixelValue] = icvprGetRandomColor();
					}
					Scalar color = colors[pixelValue];
					*data_dst++ = color[0];
					*data_dst++ = color[1];
					*data_dst++ = color[2];
				} else {
					data_dst++;
					data_dst++;
					data_dst++;
				}
			}
		}
		cout << "color size: " << colors.size() << endl;
	}

	static void findBlobs(const Mat &binary, vector<vector<Point> > &blobs) {
		map<int, vector<Point> > lable2blobs;
		for (int i = 0; i < binary.rows; i++) {
			const int* data_row = binary.ptr<int>(i);
			for (int j = 0; j < binary.cols; j++) {
				if (data_row[j] == 0)
					continue;
				if (lable2blobs.count(data_row[j]) <= 0) {
					lable2blobs[data_row[j]] = vector<Point>();
				}
				vector<Point>& cur = lable2blobs[data_row[j]];
				cur.push_back(Point(i, j));
			}
		}

		cout << "lable2blobs size:" << lable2blobs.size() << endl;
		blobs.clear();

		map<int, vector<Point> >::iterator it = lable2blobs.begin();
		for (; it != lable2blobs.end(); it++) {
			blobs.push_back(it->second);
			if (it->second.size() > 10000)
				cout << it->first << endl;
		}
		cout << "blobs size:" << blobs.size() << endl;

	}
};


#endif /* SRC_CCS_H_ */
