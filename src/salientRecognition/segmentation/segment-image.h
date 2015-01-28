#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include <cmath>
#include "image.h"
#include "misc.h"
#include "filter.h"
#include "disjoint-set.h"

#define THRESHOLD(size, c) (c/size)

typedef struct {
	float w;
	int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
	return a.w < b.w;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
DisjointSet *segment_graph(int num_vertices, int num_edges, edge *edges,
		float c) {
	// sort edges by weight
	std::sort(edges, edges + num_edges);

	// make a disjoint-set forest
	DisjointSet *u = new DisjointSet(num_vertices);

	// init thresholds
	float *threshold = new float[num_vertices];
	int i;
	for (i = 0; i < num_vertices; i++)
		threshold[i] = THRESHOLD(1, c);

	// for each edge, in non-decreasing weight order...
	for (i = 0; i < num_edges; i++) {
		edge *pedge = &edges[i];

		// components conected by this edge
		int a = u->find(pedge->a);
		int b = u->find(pedge->b);
		if (a != b) {
			if ((pedge->w <= threshold[a]) && (pedge->w <= threshold[b])) {
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

class GraphSegmentation {
public:
	GraphSegmentation(float sigma = 1.2, float mergeThreshold = 200,
			int min_size = 1000, bool debug = false);
	int segment_image(Mat &input, Mat &realSeg);
	Mat getRealSeg();
private:
	rgb random_rgb();
	float diff(Vec3f &originPixel, Vec3f &comparedPixel);

	float _sigma; //variance of guassian filter
	float _mergeThreshold;
	int _minSize;
	bool _debug;
	Mat _realSeg;
};

GraphSegmentation::GraphSegmentation(float sigma, float mergeThreshold,
		int minSize, bool debug) :
		_sigma(sigma), _mergeThreshold(mergeThreshold), _minSize(minSize), _debug(
				debug) {

}

// random color
rgb GraphSegmentation::random_rgb() {
	rgb c;
	c.r = (uchar) rand();
	c.g = (uchar) rand();
	c.b = (uchar) rand();
	return c;
}

// dissimilarity measure between pixels
// sqrt((r1 - r2)^2 + (g1 - g2)^2 + (b1 - b2)^ 2)
inline float GraphSegmentation::diff(Vec3f &originPixel, Vec3f &comparedPixel) {
	float squared = square(originPixel[0] - comparedPixel[0])
			+ square(originPixel[1] - comparedPixel[1])
			+ square(originPixel[2] - comparedPixel[2]);
	return sqrt(squared);
}

Mat GraphSegmentation::getRealSeg() {
	return _realSeg;
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
int GraphSegmentation::segment_image(Mat &img3f, Mat &segments) {
	Mat input;
	cvtColor(img3f, input, CV_BGR2Lab);
	Mat smoothImage;
	GaussianBlur(input, smoothImage, Size(), _sigma, 0, BORDER_REPLICATE);
	smoothImage.convertTo(smoothImage, CV_32FC3);
	int width = input.cols, height = input.rows;
	int x, y, i;
	// build graph
	edge *edges = new edge[width * height * 4];
	int num = 0;
	for (y = 0; y < height; y++) {
		Vec3f *row = smoothImage.ptr < Vec3f > (y);
		for (x = 0; x < width; x++) {
			Vec3f currentPixel = row[x];
			if (x < width - 1) {
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x + 1);
				edges[num].w = diff(currentPixel,
						smoothImage.at < Vec3f > (y, x + 1));
				num++;
			}
			if (y < height - 1) {
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width + x;
				edges[num].w = diff(currentPixel,
						smoothImage.at < Vec3f > (y + 1, x));
				num++;
			}
			if ((x < width - 1) && (y < height - 1)) {
				edges[num].a = y * width + x;
				edges[num].b = (y + 1) * width + (x + 1);
				edges[num].w = diff(currentPixel,
						smoothImage.at < Vec3f > (y + 1, x + 1));
				num++;
			}
			if ((x < width - 1) && (y > 0)) {
				edges[num].a = y * width + x;
				edges[num].b = (y - 1) * width + (x + 1);
				edges[num].w = diff(currentPixel,
						smoothImage.at < Vec3f > (y - 1, x + 1));
				num++;
			}
		}
	}
	// segment
	DisjointSet *edgeSet = segment_graph(width * height, num, edges,
			_mergeThreshold);

	// post process small components
	for (i = 0; i < num; i++) {
		int a = edgeSet->find(edges[i].a);
		int b = edgeSet->find(edges[i].b);
		if ((a != b)
				&& ((edgeSet->size(a) < _minSize)
						|| (edgeSet->size(b) < _minSize)))
			edgeSet->join(a, b);
	}
	delete[] edges;
	int num_ccs = edgeSet->num_sets();
	// pick random colors for each component
	// for debug
	if (_debug) {
		image11<rgb> *output = new image11<rgb>(width, height);
		rgb *colors = new rgb[width * height];
		for (i = 0; i < width * height; i++)
			colors[i] = random_rgb();
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				int comp = edgeSet->find(y * width + x);
				imRef(output, x, y) = colors[comp];
			}
		}
		_realSeg.create(input.size(), CV_8UC3);
		for (int i = 0; i < output->h; ++i) {
			for (int j = 0; j < output->w; ++j) {
				_realSeg.at < cv::Vec3b > (i, j)[0] = output->data[i * output->w
						+ j].b;
				_realSeg.at < cv::Vec3b > (i, j)[1] = output->data[i * output->w
						+ j].g;
				_realSeg.at < cv::Vec3b > (i, j)[2] = output->data[i * output->w
						+ j].r;
			}
		}
		delete[] colors;
	}

	map<int, int> marker;
	segments.create(input.size(), 4);
	int idxNum = 0;
	for (int y = 0; y < height; ++y) {
		int *imgIdx = segments.ptr<int>(y);
		for (int x = 0; x < width; ++x) {
			int comp = edgeSet->find(y * width + x);
			if (marker.find(comp) == marker.end()) {
				marker[comp] = idxNum++;
			}
			int idx = marker[comp];
			imgIdx[x] = idx;
		}
	}
	delete edgeSet;
	return num_ccs;
}

#endif
