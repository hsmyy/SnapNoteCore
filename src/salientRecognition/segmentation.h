/*
 * segmentation.h
 *
 *  Created on: Jan 19, 2015
 *      Author: fc
 */

#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

typedef struct {
	float w;
	int a, b;
} edge;

typedef struct {
	int rank;
	int p;
	int size;
} disjointElement;

class DisjointSet {
public:
	DisjointSet(int elements);
	~DisjointSet();
	int find(int x);
	void join(int x, int y);
	int size(int x) const { return elts[x].size; }
	int nu_sets() const { return num; }
	void debug();

private:
	disjointElement *elts;
	int num;
};

void DisjointSet::debug(){
	for(int i = 0; i < num; ++i){
		disjointElement ele = elts[i];
		int c = 0;
		if(ele.p != i){
			cout << "[" << ele.rank << "," << ele.size << "]" << endl;
			++c;
		}
		if(c > 100){
			break;
		}
	}
}

DisjointSet::DisjointSet(int elements) {
	elts = new disjointElement[elements];
	num = elements;
	for (int i = 0; i < elements; i++) {
		elts[i].rank = 0;
		elts[i].size = 1;
		elts[i].p = i;
	}
}

DisjointSet::~DisjointSet() {
	delete [] elts;
}

int DisjointSet::find(int x) {
	int y = x;
	while (y != elts[y].p)
		y = elts[y].p;
	elts[x].p = y;
	return y;
}

void DisjointSet::join(int x, int y) {
	if (elts[x].rank > elts[y].rank) {
		elts[y].p = x;
		elts[x].size += elts[y].size;
	} else {
		elts[x].p = y;
		elts[y].size += elts[x].size;
		if (elts[x].rank == elts[y].rank)
			elts[y].rank++;
	}
	num--;
}

int segmentImage(Mat &img, Mat &segImg, double sigma, double count, int min_size);
DisjointSet *segment_graph(int nu_vertices, int nu_edges, edge *edges, float c);

static inline float diff(Mat &img3f, int x1, int y1, int x2, int y2)
{
	const Vec3f &p1 = img3f.at<Vec3f>(y1, x1);
	const Vec3f &p2 = img3f.at<Vec3f>(y2, x2);
	return sqrt(sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2]));
}

int segmentImage(Mat &img, Mat &segImg, double sigma, double count, int min_size){
	CV_Assert(img.type() == CV_32FC3);
	int width = img.cols;
	int height = img.rows;
	Mat smImg3f;

	GaussianBlur(img, smImg3f, Size(), sigma, 0, BORDER_REPLICATE);
//	namedWindow("RC-SEG1");
//	imshow("RC-SEG1", smImg3f);

	edge *edges = new edge[width * height * 4];
	int num = 0;
	for(int y = 0; y < height; ++y){
		for(int x = 0; x < width; ++x){
			if(x < width - 1){
				edges[num].a = y * width + x;
				edges[num].b = y * width + x + 1;
				edges[num].w = diff(smImg3f, x, y, x + 1, y);
				++num;
			}

			if(y < height - 1){
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width +x;
				edges[num].w = diff(smImg3f, x, y, x, y + 1);
				++num;
			}

			if( (x < width - 1) && (y < height - 1)){
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width + x + 1;
				edges[num].w = diff(smImg3f, x, y, x + 1, y + 1);
				++num;
			}

			if( (x < width - 1) && (y > 0)){
				edges[num].a = y * width + x;
				edges[num].b = (y - 1) * width + (x + 1);
				edges[num].w = diff(smImg3f, x, y, x + 1, y - 1);
				++num;
			}
		}
	}
	cout << "edge num:" << num << endl;

	DisjointSet *u = segment_graph(width * height, num, edges, (float)count);


	for(int i = 0; i < num; ++i){
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if( (a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size))){
			u->join(a, b);
		}
	}
	u->debug();
	delete[] edges;

	map<int, int> marker;
	segImg.create(smImg3f.size(), 4);

	int idxNum = 0;
	for(int y = 0; y < height; ++y){
		int *imgIdx = segImg.ptr<int>(y);
		for(int x = 0; x < width; ++x){
			int comp = u->find(y * width + x);
			if(marker.find(comp) == marker.end()){
				marker[comp] = idxNum++;
				cout << "comp[" << comp << "],idxNum[" << idxNum <<"],size[" << u->size(comp) << "]" << endl;
			}
			int idx = marker[comp];
			imgIdx[x] = idx;
		}
	}
	delete u;

	return idxNum;
}

int comp(const void *a, const void *b){
	edge wa = *reinterpret_cast<const edge*>(a);
	edge wb = *reinterpret_cast<const edge*>(b);
	return wa.w < wb.w ? -1 : wa.w == wb.w ? 0 : 1;
}

DisjointSet *segment_graph(int nu_vertices, int nu_edges, edge edges[], float c) {
	// sort edges by weight
//	std::sort(edges, edges + nu_edges);
	std::qsort(edges, nu_edges, sizeof(edge), comp);

//	// make a disjoint-set forest
	DisjointSet *u = new DisjointSet(nu_vertices);

	// init thresholds
	float *threshold = new float[nu_vertices];
	for (int i = 0; i < nu_vertices; i++)
		threshold[i] = THRESHOLD(1,c);

	// for each edge, in non-decreasing weight order...
	for (int i = 0; i < nu_edges; i++) {
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b) {
			if ((pedge->w <= threshold[a]) &&
				(pedge->w <= threshold[b])) {
					u->join(a, b);
					a = u->find(a);
					threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
			}
		}
	}

	// free up
	delete threshold;
	return u;
}

#endif /* SEGMENTATION_H_ */
