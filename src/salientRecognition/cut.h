/*
 * cut.h
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#ifndef CUT_H_
#define CUT_H_

#include <opencv2/opencv.hpp>
#include "main.h"
#include <queue>
#include <list>
#include "graph.h"
#include "gmm.h"

using namespace std;

class CmSalCut
{
public: // Functions for saliency cut
	// User supplied Trimap values
	enum TrimapValue {TrimapBackground = 0, TrimapUnknown = 128, TrimapForeground = 255};

	CmSalCut(CMat &img3f);
	~CmSalCut(void);

	// Refer initialize for parameters
//	static Mat CutObjs(CMat &img3f, CMat &sal1f, float t1 = 0.2f, float t2 = 0.9f,
//		CMat &borderMask = Mat(), int wkSize = 20);


public: // Functions for GrabCut

	// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper
	void initialize(const Rect &rect);

	// Initialize using saliency map. In the Trimap: background < t1, foreground > t2, others unknown.
	// Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight to train fore and back ground GMMs
	void initialize(CMat &sal1f, float t1, float t2);
	void initialize(CMat &sal1u); // Background = 0, unknown = 128, foreground = 255

	void fitGMMs();

	// Run Grabcut refinement on the hard segmentation
	void refine() {int changed = 1; while (changed) changed = refineOnce();}
	int refineOnce();

	// Edit Trimap, mask values should be 0 or 255
	void setTrimap(CMat &mask1u, const TrimapValue t) {_trimap1i.setTo(t, mask1u);}

	// Get Trimap for effective interaction. Format: CV_32SC1. Values should be TrimapValue
	Mat& getTrimap() {return _trimap1i; }

	// Draw result
	void drawResult(Mat& maskForeGround) {compare(_segVal1f, 0.5, maskForeGround, CMP_GT);}
	// Return number of difference and then expand fMask to get mask1u.
	static int ExpandMask(CMat &fMask, Mat &mask1u, CMat &bdReg1u, int expandRatio = 5);

	static Mat GetNZRegionsLS(CMat &mask1u, double ignoreRatio = 0.02);

	static int GetNZRegions(const Mat_<byte> &label1u, Mat_<int> &regIdx1i, vecI &idxSum);

	static Rect GetMaskRange(CMat &mask1u, int ext, int thresh = 10);
private:
	// Update hard segmentation after running GraphCut,
	// Returns the number of pixels that have changed from foreground to background or vice versa.
	int updateHardSegmentation();

	void initGraph();	// builds the graph for GraphCut



private:
	int _w, _h;		// Width and height of the source image
	Mat _imgBGR3f, _imgLab3f; // BGR images is used to find GMMs and Lab for pixel distance
	Mat _trimap1i;	// Trimap value
	Mat _segVal1f;	// Hard segmentation with type SegmentationValue

	// Variables used in formulas from the paper.
	float _lambda;		// lambda = 50. This value was suggested the GrabCut paper.
	float _beta;		// beta = 1 / ( 2 * average of the squared color distances between all pairs of neighboring pixels (8-neighborhood) )
	float _L;			// L = a large value to force a pixel to be foreground or background
	GraphF *_graph;

	// Storage for N-link weights, each pixel stores links to only four of its 8-neighborhood neighbors.
	// This avoids duplication of links, while still allowing for relatively easy lookup.
	// First 4 directions in DIRECTION8 are: right, rightBottom, bottom, leftBottom.
	Mat_<Vec4f> _NLinks;

	int _directions[4]; // From DIRECTION8 for easy location

	CmGMM _bGMM, _fGMM; // Background and foreground GMM
	Mat _bGMMidx1i, _fGMMidx1i;	// Background and foreground GMM components, supply memory for GMM, not used for Grabcut
	Mat _show3u; // Image for display medial results
};

CmSalCut::CmSalCut(CMat &img3f)
	:_fGMM(5), _bGMM(5), _w(img3f.cols), _h(img3f.rows), _lambda(50), _graph( NULL)
{
	CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
	_imgBGR3f = img3f;
	cvtColor(_imgBGR3f, _imgLab3f, CV_BGR2Lab);
	_trimap1i = Mat::zeros(_h, _w, CV_32S);
	_segVal1f = Mat::zeros(_h, _w, CV_32F);
//	_graph = NULL;

	_L = 8 * _lambda + 1;// Compute L
	_beta = 0;
	{// compute beta: 0.5 / Expectation(pair pixel distance)
		int edges = 0;
		double result = 0;
		for (int y = 0; y < _h; ++y) {
			const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
			for (int x = 0; x < _w; ++x){
				Point pnt(x, y);
				for (int i = 0; i < 4; i++)	{
					Point pntN = pnt + DIRECTION8[i];
					if (CHK_IND(pntN))
						result += vecSqrDist(_imgLab3f.at<Vec3f>(pntN), img[x]), edges++;
				}
			}
		}
		_beta = (float)(0.5 * edges/result);
	}
	_NLinks.create(_h, _w); {// compute prior V(alpha, z)
		//dw is the distance between cur point and neighbor point
		static const float dW[4] = {1, (float)(1/SQRT2), 1, (float)(1/SQRT2)};
		for (int y = 0; y < _h; y++) {
			Vec4f *nLink = _NLinks.ptr<Vec4f>(y);
			const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
			for (int x = 0; x < _w; x++, nLink++) {
				Point pnt(x, y);
				const Vec3f& pixelC1 = img[x];
				for (int i = 0; i < 4; i++)	{
					Point pntN = pnt + DIRECTION8[i];
					if (CHK_IND(pntN))
						(*nLink)[i] = _lambda * dW[i] * exp(-_beta * vecSqrDist(_imgLab3f.at<Vec3f>(pntN), pixelC1));
				}
			}
		}
	}

	// pre-compute direction index offset
	for (int i = 0; i < 4; i++)
		_directions[i] = DIRECTION8[i].x + DIRECTION8[i].y * _w;
}

CmSalCut::~CmSalCut(void)
{
//	if (_graph)
//		delete _graph;
}

int CmSalCut::GetNZRegions(const Mat_<byte> &label1u, Mat_<int> &regIdx1i, vecI &idxSum)
{
	vector<pair<int, int> > counterIdx;
	int _w = label1u.cols, _h = label1u.rows, maxIdx = -1;
	regIdx1i.create(label1u.size());
	regIdx1i = -1;

	//use a* algorithm to find each seperated region.
	for (int y = 0; y < _h; y++){
		int *regIdx = regIdx1i.ptr<int>(y);
		const byte *label = label1u.ptr<byte>(y);
		for (int x = 0; x < _w; x++) {
			if (regIdx[x] != -1 || label[x] == 0)
				continue;

			pair<int, int> counterReg(0, ++maxIdx); // Number of pixels in region with index maxIdx
			Point pt(x, y);
			queue<Point, list<Point> > neighbs;
			regIdx[x] = maxIdx;
			neighbs.push(pt);

			// Repeatably add pixels to the queue to construct neighbor regions
			while(neighbs.size()){
				// Mark current pixel
				pt = neighbs.front();
				neighbs.pop();
				counterReg.first += label1u(pt);

				// Mark its unmarked neighbor pixels if similar
				Point nPt(pt.x, pt.y - 1); //Upper
				if (nPt.y >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);
				}

				nPt.y = pt.y + 1; // lower
				if (nPt.y < _h && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);
				}

				nPt.y = pt.y, nPt.x = pt.x - 1; // Left
				if (nPt.x >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);
				}

				nPt.x = pt.x + 1;  // Right
				if (nPt.x < _w && regIdx1i(nPt) == -1 && label1u(nPt) > 0)	{
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);
				}
			}

			// Add current region to regions
			counterIdx.push_back(counterReg);
		}
	}
	//counterIdx means <colorNum, regionIdx>
	//regIdx1i maps the point to region
	sort(counterIdx.begin(), counterIdx.end(), greater<pair<int, int> >());
	int idxNum = (int)counterIdx.size();
	vector<int> newIdx(idxNum);
	idxSum.resize(idxNum);
	for (int i = 0; i < idxNum; i++){
		// i means ordered region idx
		// idSum: i -> color Sum
		idxSum[i] = counterIdx[i].first;
		// newIdx: regionIdx -> i
		newIdx[counterIdx[i].second] = i;
	}

	//set the ordered region idx
	for (int y = 0; y < _h; y++){
		int *regIdx = regIdx1i.ptr<int>(y);
		for (int x = 0; x < _w; x++)
			if (regIdx[x] >= 0){

				regIdx[x] = newIdx[regIdx[x]];
			}
	}
	return idxNum;
}

Mat CmSalCut::GetNZRegionsLS(CMat &mask1u, double ignoreRatio)
{
	CV_Assert(mask1u.type() == CV_8UC1 && mask1u.data != NULL);
	ignoreRatio *= mask1u.rows * mask1u.cols * 255;
	Mat_<int> regIdx1i;
	vecI idxSum;
	Mat resMask;
	int regionSize = GetNZRegions(mask1u, regIdx1i, idxSum);
	// if the selected region is larger than ratio, we believe it is useful
	if (regionSize >= 1 && idxSum[0] > ignoreRatio)
		compare(regIdx1i, 0, resMask, CMP_EQ);
	return resMask;
}

Rect CmSalCut::GetMaskRange(CMat &mask1u, int ext, int thresh)
{
	int maxX = INT_MIN, maxY = INT_MIN, minX = INT_MAX, minY = INT_MAX, rows = mask1u.rows, cols = mask1u.cols;
	// get the edge of the pixel which is higher than threshold
	for (int r = 0; r < rows; r++)	{
		const byte* data = mask1u.ptr<byte>(r);
		for (int c = 0; c < cols; c++)
			if (data[c] > thresh) {
				maxX = max(maxX, c);
				minX = min(minX, c);
				maxY = max(maxY, r);
				minY = min(minY, r);
			}
	}

	maxX = maxX + ext + 1 < cols ? maxX + ext + 1 : cols;
	maxY = maxY + ext + 1 < rows ? maxY + ext + 1 : rows;
	minX = minX - ext > 0 ? minX - ext : 0;
	minY = minY - ext > 0 ? minY - ext : 0;

	return Rect(minX, minY, maxX - minX, maxY - minY);
}



// Initialize using saliency map. In the Trimap: background < t1, foreground > t2, others unknown.
// Saliency values are in [0, 1], "sal1f" and "1-sal1f" are used as weight to train fore and back ground GMMs
void CmSalCut::initialize(CMat &sal1f, float t1, float t2)
{
	CV_Assert(sal1f.type() == CV_32F && sal1f.size == _imgBGR3f.size);
	sal1f.copyTo(_segVal1f);

	for (int y = 0; y < _h; y++) {
		int* triVal = _trimap1i.ptr<int>(y);
		const float *segVal = _segVal1f.ptr<float>(y);
		for (int x = 0; x < _w; x++) {
			triVal[x] = segVal[x] < t1 ? TrimapBackground : TrimapUnknown;
			triVal[x] = segVal[x] > t2 ? TrimapForeground : triVal[x];
		}
	}
}

void CmSalCut::initialize(CMat &sal1u) // Background = 0, unknown = 128, foreground = 255
{
	CV_Assert(sal1u.type() == CV_8UC1 && sal1u.size == _imgBGR3f.size);
	for (int y = 0; y < _h; y++) {
		int* triVal = _trimap1i.ptr<int>(y);
		const byte *salVal = sal1u.ptr<byte>(y);
		float *segVal = _segVal1f.ptr<float>(y);
		for (int x = 0; x < _w; x++) {
			triVal[x] = salVal[x] < 70 ? TrimapBackground : TrimapUnknown;
			triVal[x] = salVal[x] > 200 ? TrimapForeground : triVal[x];
			segVal[x] = salVal[x] < 70 ? 0 : 1.f;
		}
	}
}

// Initial rect region in between thr1 and thr2 and others below thr1 as the Grabcut paper
void CmSalCut::initialize(const Rect &rect)
{
	_trimap1i = TrimapBackground;
	_trimap1i(rect) = TrimapUnknown;
	_segVal1f = 0;
	_segVal1f(rect) = 1;
}

void CmSalCut::fitGMMs()
{
	_fGMM.BuildGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
	Mat complement(_segVal1f);
	for(int i = 0; i < complement.rows; ++i){
		float *row = complement.ptr<float>(i);
		for(int j = 0; j < complement.cols; ++j){
			row[j] = 1 - row[j];
		}
	}
	_bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, complement);
}

int CmSalCut::refineOnce()
{
	// Steps 4 and 5: Learn new GMMs from current segmentation
	if (_fGMM.GetSumWeight() < 50 || _bGMM.GetSumWeight() < 50)
		return 0;

	_fGMM.RefineGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
	Mat complement(_segVal1f);
	for(int i = 0; i < complement.rows; ++i){
		float *row = complement.ptr<float>(i);
		for(int j = 0; j < complement.cols; ++j){
			row[j] = 1 - row[j];
		}
	}
	_bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, complement);

	// Step 6: Run GraphCut and update segmentation
	initGraph();
	if (_graph)
		_graph->maxflow();

	return updateHardSegmentation();
}
//
int CmSalCut::updateHardSegmentation()
{
	int changed = 0;
	for (int y = 0, id = 0; y < _h; ++y) {
		float* segVal = _segVal1f.ptr<float>(y);
		int* triMapD = _trimap1i.ptr<int>(y);
		for (int x = 0; x < _w; ++x, id++) {
			float oldValue = segVal[x];
			if (triMapD[x] == TrimapBackground)
				segVal[x] = 0.f; // SegmentationBackground
			else if (triMapD[x] == TrimapForeground)
				segVal[x] = 1.f; // SegmentationForeground
			else
				segVal[x] = _graph->what_segment(id) == GraphF::SOURCE ? 1.f : 0.f;
			changed += abs(segVal[x] - oldValue) > 0.1 ? 1 : 0;
		}
	}
	return changed;
}

void CmSalCut::initGraph()
{
	// Set up the graph (it can only be used once, so we have to recreate it each time the graph is updated)
	if (_graph == NULL)
		_graph = new GraphF(_w * _h, 4 * _w * _h);
	else
		_graph->reset();
	_graph->add_node(_w * _h);

	for (int y = 0, id = 0; y < _h; ++y) {
		int* triMapD = _trimap1i.ptr<int>(y);
		const float* img = _imgBGR3f.ptr<float>(y);
		for(int x = 0; x < _w; x++, img += 3, id++) {
			float back, fore;
			if (triMapD[x] == TrimapUnknown ) {
				fore = -log(_bGMM.P(img));
				back = -log(_fGMM.P(img));
			}
			else if (triMapD[x] == TrimapBackground )
				fore = 0, back = _L;
			else		// TrimapForeground
				fore = _L,	back = 0;

			// Set T-Link weights
			_graph->add_tweights(id, fore, back); // _graph->set_tweights(_nodes(y, x), fore, back);

			// Set N-Link weights from precomputed values
			Point pnt(x, y);
			const Vec4f& nLink = _NLinks(pnt);
			for (int i = 0; i < 4; i++)	{
				Point nPnt = pnt + DIRECTION8[i];
				if (CHK_IND(nPnt))
					_graph->add_edge(id, id + _directions[i], nLink[i], nLink[i]);
			}
		}
	}
}



int CmSalCut::ExpandMask(CMat &fMask, Mat &mask1u, CMat &bdReg1u, int expandRatio)
{
	compare(fMask, mask1u, mask1u, CMP_NE);
	int changed = cvRound(sum(mask1u).val[0] / 255.0);

	Mat bigM, smalM;
	dilate(fMask, bigM, Mat(), Point(-1, -1), expandRatio);
	erode(fMask, smalM, Mat(), Point(-1, -1), expandRatio);
	static const double erodeSmall = 255 * 50;
	if (sum(smalM).val[0] < erodeSmall)
		smalM = fMask;
	mask1u = bigM * 0.5 + smalM * 0.5;
	mask1u.setTo(0, bdReg1u);
	return changed;
}


#endif /* CUT_H_ */
