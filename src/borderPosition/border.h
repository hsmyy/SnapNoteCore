/*
 * DisplayImage.cpp
 *
 *  Created on: Jan 7, 2015
 *      Author: litton
 */


/**
 * Automatic perspective correction for quadrilateral objects. See the tutorial at
 * http://opencv-code.com/tutorials/automatic-perspective-correction-for-quadrilateral-objects/
 */
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <strstream>
#include <fstream>

using namespace cv;

typedef struct CvLinePolar2
{
	float x1, y1, x2, y2;
    float rho;
    float angle;
    float votes;
    float score;
}
CvLinePolar2;

typedef struct OppositeLines{
	int one;
	int two;
}OppositeLines;

typedef struct Quadrangle{
	cv::Point2f a,b,c,d;
}Quadrangle;

int THRESHOLD[2] = {2,7};
int SIZE[2] = {3,7};
int RUN[4] = {1,2,2,1};
int VOTERATE = 2;
double OPPOANG = 1.0/6;
int THRESHSCALE = 40;//210;
int MAXLINK = 20;

#define hough_cmp_gt(l1,l2) (aux[l1] > aux[l2])
#define CV_IMPLEMENT_QSORT_EX( func_name, T, LT, user_data_type )                           \
		void func_name( T *array, size_t total, user_data_type aux )                        \
		{                                                                                   \
		    int isort_thresh = 7;                                                           \
		    T t;                                                                            \
		    int sp = 0;                                                                     \
		                                                                                    \
		    struct                                                                          \
		    {                                                                               \
		        T *lb;                                                                      \
		        T *ub;                                                                      \
		    }                                                                               \
		    stack[48];                                                                      \
		                                                                                    \
		    aux = aux;                                                                      \
		                                                                                    \
		    if( total <= 1 )                                                                \
		        return;                                                                     \
		                                                                                    \
		    stack[0].lb = array;                                                            \
		    stack[0].ub = array + (total - 1);                                              \
		                                                                                    \
		    while( sp >= 0 )                                                                \
		    {                                                                               \
		        T* left = stack[sp].lb;                                                     \
		        T* right = stack[sp--].ub;                                                  \
		                                                                                    \
		        for(;;)                                                                     \
		        {                                                                           \
		            int i, n = (int)(right - left) + 1, m;                                  \
		            T* ptr;                                                                 \
		            T* ptr2;                                                                \
		                                                                                    \
		            if( n <= isort_thresh )                                                 \
		            {                                                                       \
		            insert_sort:                                                            \
		                for( ptr = left + 1; ptr <= right; ptr++ )                          \
		                {                                                                   \
		                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1]); ptr2--)   \
		                        CV_SWAP( ptr2[0], ptr2[-1], t );                            \
		                }                                                                   \
		                break;                                                              \
		            }                                                                       \
		            else                                                                    \
		            {                                                                       \
		                T* left0;                                                           \
		                T* left1;                                                           \
		                T* right0;                                                          \
		                T* right1;                                                          \
		                T* pivot;                                                           \
		                T* a;                                                               \
		                T* b;                                                               \
		                T* c;                                                               \
		                int swap_cnt = 0;                                                   \
		                                                                                    \
		                left0 = left;                                                       \
		                right0 = right;                                                     \
		                pivot = left + (n/2);                                               \
		                                                                                    \
		                if( n > 40 )                                                        \
		                {                                                                   \
		                    int d = n / 8;                                                  \
		                    a = left, b = left + d, c = left + 2*d;                         \
		                    left = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))     \
		                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
		                                                                                    \
		                    a = pivot - d, b = pivot, c = pivot + d;                        \
		                    pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
		                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
		                                                                                    \
		                    a = right - 2*d, b = right - d, c = right;                      \
		                    right = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))    \
		                                      : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));    \
		                }                                                                   \
		                                                                                    \
		                a = left, b = pivot, c = right;                                     \
		                pivot = LT(*a, *b) ? (LT(*b, *c) ? b : (LT(*a, *c) ? c : a))        \
		                                   : (LT(*c, *b) ? b : (LT(*a, *c) ? a : c));       \
		                if( pivot != left0 )                                                \
		                {                                                                   \
		                    CV_SWAP( *pivot, *left0, t );                                   \
		                    pivot = left0;                                                  \
		                }                                                                   \
		                left = left1 = left0 + 1;                                           \
		                right = right1 = right0;                                            \
		                                                                                    \
		                for(;;)                                                             \
		                {                                                                   \
		                    while( left <= right && !LT(*pivot, *left) )                    \
		                    {                                                               \
		                        if( !LT(*left, *pivot) )                                    \
		                        {                                                           \
		                            if( left > left1 )                                      \
		                                CV_SWAP( *left1, *left, t );                        \
		                            swap_cnt = 1;                                           \
		                            left1++;                                                \
		                        }                                                           \
		                        left++;                                                     \
		                    }                                                               \
		                                                                                    \
		                    while( left <= right && !LT(*right, *pivot) )                   \
		                    {                                                               \
		                        if( !LT(*pivot, *right) )                                   \
		                        {                                                           \
		                            if( right < right1 )                                    \
		                                CV_SWAP( *right1, *right, t );                      \
		                            swap_cnt = 1;                                           \
		                            right1--;                                               \
		                        }                                                           \
		                        right--;                                                    \
		                    }                                                               \
		                                                                                    \
		                    if( left > right )                                              \
		                        break;                                                      \
		                    CV_SWAP( *left, *right, t );                                    \
		                    swap_cnt = 1;                                                   \
		                    left++;                                                         \
		                    right--;                                                        \
		                }                                                                   \
		                                                                                    \
		                if( swap_cnt == 0 )                                                 \
		                {                                                                   \
		                    left = left0, right = right0;                                   \
		                    goto insert_sort;                                               \
		                }                                                                   \
		                                                                                    \
		                n = MIN( (int)(left1 - left0), (int)(left - left1) );               \
		                for( i = 0; i < n; i++ )                                            \
		                    CV_SWAP( left0[i], left[i-n], t );                              \
		                                                                                    \
		                n = MIN( (int)(right0 - right1), (int)(right1 - right) );           \
		                for( i = 0; i < n; i++ )                                            \
		                    CV_SWAP( left[i], right0[i-n+1], t );                           \
		                n = (int)(left - left1);                                            \
		                m = (int)(right1 - right);                                          \
		                if( n > 1 )                                                         \
		                {                                                                   \
		                    if( m > 1 )                                                     \
		                    {                                                               \
		                        if( n > m )                                                 \
		                        {                                                           \
		                            stack[++sp].lb = left0;                                 \
		                            stack[sp].ub = left0 + n - 1;                           \
		                            left = right0 - m + 1, right = right0;                  \
		                        }                                                           \
		                        else                                                        \
		                        {                                                           \
		                            stack[++sp].lb = right0 - m + 1;                        \
		                            stack[sp].ub = right0;                                  \
		                            left = left0, right = left0 + n - 1;                    \
		                        }                                                           \
		                    }                                                               \
		                    else                                                            \
		                        left = left0, right = left0 + n - 1;                        \
		                }                                                                   \
		                else if( m > 1 )                                                    \
		                    left = right0 - m + 1, right = right0;                          \
		                else                                                                \
		                    break;                                                          \
		            }                                                                       \
		        }                                                                           \
		    }                                                                               \
		}
static CV_IMPLEMENT_QSORT_EX( icvHoughSortDescent32s, int, hough_cmp_gt, const int* )


cv::Point2f center(0,0);

cv::Point2d computeIntersect(cv::Vec4i a,
                             cv::Vec4i b)
{
	double x1 = a[0]+0.0, y1 = a[1]+0.0, x2 = a[2]+0.0, y2 = a[3]+0.0, x3 = b[0]+0.0, y3 = b[1]+0.0, x4 = b[2]+0.0, y4 = b[3]+0.0;
	//std::cout<<"x1: "<<x1<<",y1: "<<y1<<",x2: "<<x2<<",y2: "<<y2<<",x3: "<<x3<<",y3: "<<y3<<",x4: "<<x4<<",y4: "<<y4<<std::endl;
	float denom;

	if (double d = ((double)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2d pt;
		pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
		pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
		//std::cout<<"see: "<<(x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)<<std::endl;
		//std::cout<<"intersect: ("<<pt.x<<","<<pt.y<<")"<<std::endl;
		return pt;
	}
	else
		return cv::Point2f(-12345, -12345);
}

static void
icvHoughLinesStandard2( const CvMat* img, float rho, float theta,
                       int threshold, CvSeq *lines, int linesMax )
{
    cv::AutoBuffer<int> _accum, _sort_buf;
    cv::AutoBuffer<float> _tabSin, _tabCos;

    const uchar* image;
    int step, width, height;
    int numangle, numrho;
    int total = 0;
    float ang;
    int r, n;
    int i, j;
    float irho = 1 / rho;
    double scale;

    CV_Assert( CV_IS_MAT(img) && CV_MAT_TYPE(img->type) == CV_8UC1 );

    image = img->data.ptr;
    step = img->step;
    width = img->cols;
    height = img->rows;

    numangle = cvRound(CV_PI / theta);
    numrho = cvRound(((width + height) * 2 + 1) / rho);

    _accum.allocate((numangle+2) * (numrho+2));
    _sort_buf.allocate(numangle * numrho);
    _tabSin.allocate(numangle);
    _tabCos.allocate(numangle);
    int *accum = _accum, *sort_buf = _sort_buf;
    float *tabSin = _tabSin, *tabCos = _tabCos;

    memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );

    for( ang = 0, n = 0; n < numangle; ang += theta, n++ )
    {
        tabSin[n] = (float)(sin(ang) * irho);
        tabCos[n] = (float)(cos(ang) * irho);
    }

    // stage 1. fill accumulator
    for( i = 0; i < height; i++ )
        for( j = 0; j < width; j++ )
        {
            if( image[i * step + j] != 0 )
                for( n = 0; n < numangle; n++ )
                {
                    r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }

    // stage 2. find local maximums
    for( r = 0; r < numrho; r++ )
        for( n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf[total++] = base;
        }

    // stage 3. sort the detected lines by accumulator value
    icvHoughSortDescent32s( sort_buf, total, accum );

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = MIN(linesMax, total);
    scale = 1./(numrho+2);
    for( i = 0; i < linesMax; i++ )
    {
        CvLinePolar2 line;
        int idx = sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = n * theta;
        line.votes = accum[idx];
        cvSeqPush( lines, &line );
    }
    //TODO garbage collection
}

void sortCorners(std::vector<cv::Point2f>& corners,
                 cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}


	if (top.size() == 2 && bot.size() == 2){
		corners.clear();
		cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
		cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
		cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
		cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];


		corners.push_back(tl);
		corners.push_back(tr);
		corners.push_back(br);
		corners.push_back(bl);
	}
	else{
		std::vector<cv::Point2f> left, right;

		for (int i = 0; i < corners.size(); i++)
		{
			if (corners[i].x < center.x)
				left.push_back(corners[i]);
			else
				right.push_back(corners[i]);
		}
		corners.clear();
		if (left.size() == 2 && right.size() == 2){
			cv::Point2f tl = left[0].y > left[1].y ? left[1] : left[0];
			cv::Point2f tr = right[0].y > right[1].y ? right[1] : right[0];
			cv::Point2f bl = left[0].y > left[1].y ? left[0] : left[1];
			cv::Point2f br = right[0].y > right[1].y ? right[0] : right[1];


			corners.push_back(tl);
			corners.push_back(tr);
			corners.push_back(br);
			corners.push_back(bl);
		}
	}
}

double diancheng(double* v1, double* v2, int c){
	double ret = 0.0;
	for(int i=0;i<c;i++)
		ret+=v1[i]*v2[i];
	//std::cout<<"diancheng "<<ret<<std::endl;
	return ret;
}

void chacheng(double* v1, double* v2, double* dist, int c){
	dist[0] = (v1[1]*v2[2]-v1[2]*v2[1]);
	dist[1] = (v1[2]*v2[0]-v1[0]*v2[2]);
	dist[2] = (v1[0]*v2[1]-v1[1]*v2[0]);
	//std::cout<<"chacheng: "<<dist[0]<<", "<<dist[1]<<", "<<dist[2]<<std::endl;
}

void pointToVecP(cv::Point2f pt, double* v3){
	//double v3[3];
	v3[0] = pt.x;
	v3[1] = pt.y;
	v3[2] = 1.0;
	//return v3;
}

CV_IMPL CvSeq*
cvHoughLines3( CvArr* src_image, void* lineStorage, int method,
               double rho, double theta, int threshold,
               double param1, double param2 )
{

    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* lines = 0;
    CvSeq lines_header;
    CvSeqBlock lines_block;
    int lineType, elemSize;
    int linesMax = INT_MAX;
    int iparam1, iparam2;

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !lineStorage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( rho <= 0 || theta <= 0 || threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "rho, theta and threshold must be positive" );

    if( method != CV_HOUGH_PROBABILISTIC )
    {
        lineType = CV_32FC3;
        elemSize = sizeof(float)*3;
    }
    else
    {
        lineType = CV_32SC4;
        elemSize = sizeof(int)*4;
    }

    if( CV_IS_STORAGE( lineStorage ))
    {
        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
    }
    else if( CV_IS_MAT( lineStorage ))
    {
        mat = (CvMat*)lineStorage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        if( CV_MAT_TYPE( mat->type ) != lineType )
            CV_Error( CV_StsBadArg,
            "The destination matrix data type is inappropriate, see the manual" );

        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
        linesMax = lines->total;
        cvClearSeq( lines );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    iparam1 = cvRound(param1);
    iparam2 = cvRound(param2);

    switch( method )
    {
    case CV_HOUGH_STANDARD:

          icvHoughLinesStandard2( img, (float)rho,
                (float)theta, threshold, lines, linesMax );
          break;

    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }
    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = lines->total;
        else
            mat->rows = lines->total;
    }
    else
        result = lines;
    return result;
}

double dist(cv::Point2d p1, cv::Point2d p2){
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

double normalizeAngle(CvLinePolar2* line,int w,int h,int k,int l){
	double rho = line ->rho;
	double theta0 = line->angle;
	if(rho<0){
		rho = -rho;
		theta0 += CV_PI;
	}
	double dx = w/2.0, dy = h/2.0;
	double xjd = rho/(cos(theta0)+dy*sin(theta0)/dx);
	double res = theta0;
	//if(k==0&&l==13) std::cout<<"xjd! "<<xjd<<std::endl;

	if(xjd>=0&&xjd<dx){

		//if(k==0&&l==13)
		//	std::cout<<"intersect! "<<std::endl;

		if(fabs(CV_PI-theta0)<=0.0000001)
			res = CV_PI;
		if(theta0<CV_PI)
			res = theta0+CV_PI;
		else
			res = theta0-CV_PI;
	}

	//if(k==0&&l==13)
		//std::cout<<"normailize "<<res<<std::endl;

	return res;
}

using namespace std;
cv::Mat grad_x, grad_y,grad_x0, grad_y0;
cv::Mat abs_grad_x, abs_grad_y, abs_grad_x0, abs_grad_y0;
cv::Mat grad,grad0;
std::vector<cv::Vec4i> finalines(4);

bool hasPixel(cv::Mat& mat, int thresh) {
	int channels = mat.channels();
	int nRows = mat.rows * channels;
	int nCols = mat.cols;
	//cout<<"size: " << nRows * nCols<<endl;
	if (mat.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; ++i) {
		uchar* p = mat.ptr<uchar>(i);
		for (int j = 0; j < nCols; ++j) {
			//cout<<"value: "<<p[j]<<endl;
			if (p[j] > thresh) {
				return true;
			}
		}
	}
	return false;
}

bool hasInterBlack(cv::Mat& mat, bool debug) {

	bool flag1 = true;
	Mat rowTop = mat.row(0);
	uchar* p = rowTop.ptr<uchar>(0);

	for (int j = 0; j < mat.cols; ++j) {

		if (p[j] > 40) {
			flag1 = false;
			break;
		}
	}

	bool flag2 = true;
	Mat rowBot = mat.row(mat.rows-1);

	p = rowBot.ptr<uchar>(0);
	for (int j = 0; j < mat.cols; ++j) {

		if (p[j] > 40) {
			flag2 = false;
			break;
		}
	}

	if(flag1&&flag2) return true;

	bool flag3 = true;
	Mat colL = mat.col(0);

	p = colL.ptr<uchar>(0);
	for (int j = 0; j < mat.rows*mat.cols; j+=mat.cols) {

		if (p[j] > 40) {
			flag3 = false;
			break;
		}
	}

	bool flag4 = true;
	Mat colR = mat.col(mat.cols-1);
	p = colR.ptr<uchar>(0);
	for (int j = 0; j < mat.rows*mat.cols; j+=mat.cols) {

		if (p[j] > 40) {
			flag4 = false;
			break;
		}
	}
	if(flag3&&flag4) return true;
	return false;
}

bool hasBlackBorder(cv::Mat& mat, bool debug) {

	bool flag = true;
	Mat rowTop = mat.row(0);
	uchar* p = rowTop.ptr<uchar>(0);

	for (int j = 0; j < mat.cols; ++j) {

		if (p[j] > 40) {
			flag = false;
			break;
		}
	}

	if(flag) return true;

	flag = true;
	Mat rowBot = mat.row(mat.rows-1);

	p = rowBot.ptr<uchar>(0);
	for (int j = 0; j < mat.cols; ++j) {

		if (p[j] > 40) {
			flag = false;
			break;
		}
	}

	if(flag) return true;
	flag = true;
	Mat colL = mat.col(0);

	p = colL.ptr<uchar>(0);
	for (int j = 0; j < mat.rows*mat.cols; j+=mat.cols) {

		if (p[j] > 40) {
			flag = false;
			break;
		}
	}
	if(flag) return true;

	flag = true;
	Mat colR = mat.col(mat.cols-1);
	p = colR.ptr<uchar>(0);
	for (int j = 0; j < mat.rows*mat.cols; j+=mat.cols) {

		if (p[j] > 40) {
			flag = false;
			break;
		}
	}
	return flag;
}

map<int, int> blackScore;
bool isBlackBorder(cv::Mat& mat, cv::Point pt1, cv::Point pt2, int threshhold, int size, int mode, int id) {
	if (mat.channels() != 1) {
		CV_Error(CV_StsBadArg, "Mat should be gray image.");
	}
	if (size < 1) {
		CV_Error(CV_StsBadArg, "Size should be positive");
	}
	//std::cout<<"points "<<pt1.x<<"-"<<pt1.y<<","<<pt2.x<<"-"<<pt2.y<<std::endl;

	CvMat src = mat;
	cv::Mat dstMat = cv::Mat::zeros(size, size, CV_8UC1);
	CvMat dst = dstMat;
	CvLineIterator iterator;
	int count = cvInitLineIterator(&src, pt1, pt2, &iterator, 8);

	int maxLink = 0;
	int link = 0;

	int interLink=0;
	int maxInter =0;
	//cout<<"test grad "<<(int)grad_y.at<char>(843,0)<<" "<<(int)grad_y.at<char>(843,1)<<" "<<(int)grad_y.at<char>(843,2)<<" "<<(int)grad_y.at<char>(843,3)<<endl;
	//cout<<count<<" "<<size<<endl;
	while (count--) {
		CV_NEXT_LINE_POINT(iterator);
		/* print the pixel coordinates: demonstrates how to calculate the coordinates */
		int offset, x, y;
		/* assume that ROI is not set, otherwise need to take it into account. */
		offset = iterator.ptr - src.data.ptr;
		y = offset / src.step;
		x = (offset - y * src.step) / sizeof(uchar);

		CvPoint2D32f ptr = cvPoint2D32f(x, y);
		cvGetRectSubPix(&src, &dst, ptr);

		cv::Mat I = cv::cvarrToMat(&dst);
		//cout<<"x: " << x << ", y:" <<y<<endl;


		bool debug = false;

		if (!hasBlackBorder(I,debug)||hasInterBlack(I,debug)) {

			maxLink = std::max(maxLink, link);
			link = 0;

		} else {
			link++;
		}

		if (!hasInterBlack(I,debug)) {

			maxInter = std::max(maxInter, interLink);
			interLink = 0;

		} else {
			interLink++;
		}
	}

	maxLink =max(maxLink,link);
	maxInter=max(maxInter,interLink);
	if(mode==1&&maxInter>=10){//20
		cout<<"inter black "<<maxInter<<endl;
		return false;
	}
	if (mode==1&&maxLink >= 65)//20
	{
		std::cout<<"yes black_border "<<maxLink<<"; inter black "<<maxInter<<std::endl;
		blackScore[id]=maxLink;
		return true;
	}
	cout<<"no black_border "<<maxLink<<std::endl;
	return false;
}

bool isLine(cv::Mat& mat, double& linkScore, double& linkSpace, cv::Point pt1, cv::Point pt2, int threshhold, int size, int mode, int procMode, bool debug) {
	if (mat.channels() != 1) {
		CV_Error(CV_StsBadArg, "Mat should be gray image.");
	}
	if (size < 1) {
		CV_Error(CV_StsBadArg, "Size should be positive");
	}
	//std::cout<<"points "<<pt1.x<<"-"<<pt1.y<<","<<pt2.x<<"-"<<pt2.y<<std::endl;

	CvMat src = mat;
	cv::Mat dstMat = cv::Mat::zeros(size, size, CV_8UC1);
	CvMat dst = dstMat;
	CvLineIterator iterator;
	int count = cvInitLineIterator(&src, pt1, pt2, &iterator, 8);

	int space = 0;
	int maxSpace = 0;
	int link = 0;
	int maxLink = 0;

	int ag_x = -9999;
	int ag_y = -9999;
	int lr = 0;
	int tb = 0;
	int maxLr = 0;
	int maxTb = 0;
	int lastLv = -1;
	//cout<<"test grad "<<(int)grad_y.at<char>(843,0)<<" "<<(int)grad_y.at<char>(843,1)<<" "<<(int)grad_y.at<char>(843,2)<<" "<<(int)grad_y.at<char>(843,3)<<endl;
	//cout<<count<<endl;
	int lastx = -1;
	int lasty = -1;

	while (--count) {
		CV_NEXT_LINE_POINT(iterator);

		/* print the pixel coordinates: demonstrates how to calculate the coordinates */
		int offset, x, y;
		/* assume that ROI is not set, otherwise need to take it into account. */
		offset = iterator.ptr - src.data.ptr;
		y = offset / src.step;
		x = (offset - y * src.step) / sizeof(uchar);

		int g_x = 0;
		int g_y = 0;

		if(procMode==3)
		try{
			g_x = (int)grad_x.at<char>(x,2*y);
			g_y = (int)grad_y.at<char>(x,2*y);
		}
		catch(...){

		}
//		if (debug)
//			cout<<"x: " << x << ", y:" <<y<<endl;

		if(x==lastx&&y==lasty)
			return false;

		lastx = x;
		lasty = y;

		//cout<<"gx: " << g_x << ", gy:" <<g_y<<endl;
		CvPoint2D32f ptr = cvPoint2D32f(x, y);
		cvGetRectSubPix(&src, &dst, ptr);

		cv::Mat I = cv::cvarrToMat(&dst);
		//cout<<"Mat: "<<I<<endl;
		int greyLv = mat.at<uchar>(ptr);
		if (!hasPixel(I,THRESHSCALE)||lastLv==-1||abs(lastLv-greyLv)>50) {
			space++;

			maxLink = std::max(maxLink, link);
			link = 0;

		} else {
			maxSpace = std::max(maxSpace, space);
			space = 0;

			link++;
		}

		lastLv = greyLv;

		if(mode==1&&procMode==3){
		//continuous points of same-directed gradient on direction x, e.g. left, left, left...
		if(ag_x == -9999 ||g_x*ag_x>=0){
			//cout<<"update ag_x "<<ag_x<<" "<<g_x*ag_x<<" "<<lr+1<<endl;
			ag_x = g_x;
			lr++;
		}
		else{
			ag_x = -9999;
		    maxLr=max(maxLr,lr);
		    //cout<<"update max lr "<<maxLr<<endl;
		    lr=0;
		}

		//continuous points of same-directed gradient on direction y, e.g. down, down, down...
		//TODO we need to consider the integration gradient of (x,y) at same time, approximated calculation now
		if(ag_y == -9999 ||g_y*ag_y>=0){
			//cout<<"update ag_y "<<ag_y<<" "<<g_y*ag_y<<" "<<tb+1<<endl;
			ag_y = g_y;
			tb++;
		}
		else{
			ag_y = -9999;
			maxTb=max(maxTb,tb);
			//cout<<"update max tb "<<maxTb<<endl;
			tb=0;
		}
		}
	}
	//cout<<"xxxxx"<<endl;
	maxSpace = std::max(maxSpace, space);
	maxLink =max(maxLink,link);
	maxLr=max(maxLr,lr);
	maxTb=max(maxTb,tb);
	//cout<<"grad "<<maxLr<<" "<<maxTb<<endl;
	if(procMode==3&&mode==1&&maxLr<11&&maxTb<7){
		//if (debug) cout<<"false0"<<endl;
		return false;
	}
	if (mode<=2&&maxLink >= MAXLINK)
	{
		//if (debug) std::cout<<"true1 "<<maxLink<<std::endl;
		linkScore = maxLink;
		linkSpace = maxSpace;
		if(linkSpace==0) linkSpace=1;
		return true;
	}
	if (maxSpace >= threshhold)
	{
		//if (debug) std::cout<<"false1 "<<maxSpace<<" "<<maxLink<<" "<<threshhold<<std::endl;
		return false;
	}
	//if (debug) std::cout<<"true2 "<<maxSpace<<" "<<maxLink<<std::endl;
	linkScore = maxLink;
	return true;
}

int lineSorted[5000];
CvSeq* lines = 0;

int compareLineScore (const void * a, const void * b)
{
	int ai = *(int*)a;
	int bi = *(int*)b;
	CvLinePolar2* l1 = (CvLinePolar2*)cvGetSeqElem(lines,ai);
	CvLinePolar2* l2 = (CvLinePolar2*)cvGetSeqElem(lines,bi);
	return -(l1->score-l2->score);
}

void sortLines(CvSeq* lines){

	for(int i=0;i<min(5000,lines->total);i++){
		lineSorted[i]=i;
	}

	qsort(lineSorted, min(5000,lines->total), sizeof(int), compareLineScore);
}

CV_IMPL CvSeq*
convertToPolar(std::vector<cv::Vec4i> lines0, CvMemStorage* storage, cv::Mat pic1){
	int count = 0;
	int lineType = CV_32FC(8);
	int elemSize = sizeof(float)*8;

	lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, storage );
	int recMaxLink = MAXLINK;
	for(int i=0;i<lines0.size();i++){
		CvLinePolar2 line;
		line.x1 = lines0[i][0];
		line.y1 = lines0[i][1];
		line.x2 = lines0[i][2];
		line.y2 = lines0[i][3];
		line.votes = 1;

		//cout<<"x-y: "<<line.x1<<","<<line.y1<<","<<line.x2<<","<<line.y2<<endl;
		if(lines0[i][1]==lines0[i][3]){
			line.angle = lines0[0][1]>=0?CV_PI/2:3*CV_PI/2;
			line.rho = fabs(line.y1);
		}
		else{
			double k = -(line.x2-line.x1)/(line.y2-line.y1);
			double x3 = (line.x1+k*line.y1)/(1+k*k);
			double y3 = k*x3;
			line.rho = sqrt(x3*x3+y3*y3);
			line.angle = atan(k);

			if(x3>0&&y3>0)
				;
			if(x3<0&&y3<0)
				line.angle = CV_PI + line.angle;
			if(x3<0&&y3>0)
				line.angle = CV_PI + line.angle;
			if(x3>0&&y3<0)
				line.angle = 2*CV_PI+line.angle;
			//cout<<"poi: "<<x3<<" "<<y3<<" "<<k<<" "<<atan(k)<<endl;
			//cout<<"pol: "<<line.rho<<" "<<line.angle<<endl;
		}

		MAXLINK = 20;
		double linkScore = 0.0;
		double linkSpace = 0.0;
		if(isLine(pic1, linkScore, linkSpace, cv::Point(lines0[i][0],lines0[i][1]), cv::Point(lines0[i][2],lines0[i][3]), 2, 1, 1, 0,true))
		{

			line.score = linkScore + sqrt((line.x1-line.x2)*(line.x1-line.x2)+(line.y1-line.y2)*(line.y1-line.y2));
			//cout<<"score: "<<line.score<<endl;
			/*
			Mat pic11 = pic1.clone();
			cv::line( pic11, cv::Point(lines0[i][0],lines0[i][1]), cv::Point(lines0[i][2],lines0[i][3]), CV_RGB(255,255,255),2);

			//Size sz = Size(pic11.cols*2/5,pic11.rows*2/5);
			//Mat pic3 = Mat(sz,CV_32S);
			//cv::resize(pic11, pic3, sz);
			//cv::imshow("image", pic3);
			cv::imshow("image", pic11);
			cv::waitKey();
*/
			cvSeqPush( lines, &line );
		}
	}
	MAXLINK = recMaxLink;
	sortLines(lines);

	return lines;
}

bool notRectLineLv2(float angle,float rho){
	if(rho<0)
		angle += CV_PI;
	if((angle>=CV_PI/3&&angle<=2*CV_PI/3)||(angle>=CV_PI*4/3&&angle<=5*CV_PI/3))
		return false;
	if((angle>=0&&angle<=CV_PI/6)||(angle>=CV_PI*11/6&&angle<=2*CV_PI))
		return false;
	return true;
}
//TODO detect the qudrangle a real one or fake one, with the continuing points
bool isRealQuadr(cv::Mat pic, cv::Vec4i xylines[], int thresh, int size, int procMode, double& score, bool debug){

	cv::Point2f pt[4];

	if(debug)
		cout<<"[DEBUG] "<<thresh<<" "<<size<<endl;
	for(int n=0;n<4;n++)
		pt[n] = computeIntersect(xylines[n/2], xylines[2+n%2]);

	int THRESHOLD = thresh;
	int SIZE = size;
	double dd1,dd2,dd3,dd4;
	double dx1,dx2,dx3,dx4;

	if(!isLine(pic,dd1,dx1,pt[0],pt[1],THRESHOLD,SIZE,2,procMode,debug))
		return false;
	if(!isLine(pic,dd2,dx2,pt[2],pt[3],THRESHOLD,SIZE,2,procMode,debug))
		return false;
	if(!isLine(pic,dd3,dx3,pt[0],pt[2],THRESHOLD,SIZE,2,procMode,debug))
		return false;
	if(!isLine(pic,dd4,dx4,pt[1],pt[3],THRESHOLD,SIZE,2,procMode,debug))
		return false;

	score = (dd1+dd2+dd3+dd4)/(dx1+dx2+dx3+dx4);
	if(debug) cout<<(dd1+dd2+dd3+dd4)<<" "<<(dx1+dx2+dx3+dx4)<<endl;
	//std::cout<<"true quadr"<<std::endl;
	return true;
}

bool verti(CvLinePolar2* line,bool debug){
	if(fabs(line->angle)<CV_PI/10.0)
		return true;

	return false;
}

bool horiz(CvLinePolar2* line,bool debug){
	if(fabs(line->angle-CV_PI/2)<CV_PI/10.0)
		return true;

	return false;
}

bool toushi(CvLinePolar2* line1, CvLinePolar2* line2,bool debug){

	CvLinePolar2* line_r;
	CvLinePolar2* line_l;
	bool find = false;

	if(debug)
		cout<<"[debug toushi] "<<line1->angle<<" "<<line2->angle<<endl;

	if(line1->angle>=0&&line1->angle<=CV_PI/2+0.0001&&line2->angle>=CV_PI/2-0.0001&&line2->angle<=CV_PI){

		if(debug)
			cout<<"[debug toushi] TRUE"<<endl;
		find = true;
		line_r = line1;
		line_l = line2;
	}

	if(line2->angle>=0&&line2->angle<=CV_PI/2+0.0001&&line1->angle>=CV_PI/2-0.0001&&line1->angle<=CV_PI){
		if(debug)
			cout<<"[debug toushi] TRUE"<<endl;
		find = true;
		line_r = line2;
		line_l = line1;
	}

	if(find){
		double ang1 = CV_PI/2 - line_r->angle;
		double ang2 = line_l->angle - CV_PI/2;

		if(fabs(ang1-ang2)>CV_PI/10)
			return false;
	}
	return true;
}

bool shuzhi(CvLinePolar2* line1, CvLinePolar2* line2,bool debug){

	CvLinePolar2* line_t;
	CvLinePolar2* line_b;
	bool find = false;

	if(debug)
		cout<<"[debug shuzhi] "<<line1->angle<<" "<<line2->angle<<endl;

	if(line1->angle>=0&&line1->angle<=CV_PI/2+0.0001&&line2->angle>=CV_PI*3/2-0.0001&&line2->angle<=2*CV_PI){

		if(debug)
			cout<<"[debug shuzhi] TRUE"<<endl;
		find = true;
		line_t = line2;
		line_b = line1;
	}

	if(line2->angle>=0&&line2->angle<=CV_PI/2+0.0001&&line1->angle>=CV_PI*3/2-0.0001&&line1->angle<=2*CV_PI){
		if(debug)
			cout<<"[debug shuzhi] TRUE"<<endl;
		find = true;
		line_t = line1;
		line_b = line2;
	}

	if(find){
		double ang1 = 2*CV_PI/2 - line_t->angle;
		double ang2 = line_b->angle;

		if(fabs(ang1-ang2)>CV_PI/10)
			return false;
	}
	return false;
}

bool isLikeRect(CvLinePolar2 ** clines,bool debug){
	CvLinePolar2* line1 = clines[0];
	CvLinePolar2* line2 = clines[1];
	CvLinePolar2* line3 = clines[2];
	CvLinePolar2* line4 = clines[3];

	if(debug){
		cout<<"[debug like rect] "<<line1->angle<<" "<<line2->angle<<" "<<line3->angle<<" "<<line4->angle<<endl;
	}

	if(verti(line1,debug)&&verti(line2,debug)&&!toushi(line3,line4,debug))
		return false;

	if(verti(line3,debug)&&verti(line4,debug)&&!toushi(line1,line2,debug))
		return false;
/*
	if(horiz(line1,debug)&&horiz(line2,debug)&&!shuzhi(line3,line4,debug))
		return false;

	if(horiz(line3,debug)&&horiz(line4,debug)&&!shuzhi(line1,line2,debug))
		return false;
*/
	if(debug) cout<<"[debug like rect] TRUE"<<endl;
	return true;
}

void modifyAttr(int procMode, int run){

	if(procMode==0){
		THRESHOLD[0] = 2;
		THRESHOLD[1] = 7;
		SIZE[0] = 3;
		SIZE[1] = 7;
		VOTERATE = 2;
		OPPOANG = 1.0/6;
		THRESHSCALE = 40;//210;
		MAXLINK = 20;
	}

	if(procMode==1){
		THRESHOLD[0] = 4*(run+1);
		THRESHOLD[1] = 20;
		SIZE[0] = 6*(run+1);
		SIZE[1] = 20;
		THRESHSCALE = 40;
		MAXLINK = 50;
		VOTERATE = 2;
		OPPOANG = 1.0/6;
	}

	if(procMode==2){
		THRESHOLD[0] = 2;
		THRESHOLD[1] = 15-5*run;
		SIZE[0] = 3;
		SIZE[1] = 15-5*run;
		THRESHSCALE = 80;
		MAXLINK = 50;
		VOTERATE = 2;
		OPPOANG = 1.0/6;
	}

	if(procMode==3){
		THRESHOLD[0] = 6;//2
		THRESHOLD[1] = 60;//20
		SIZE[0] = 3;
		SIZE[1] = 2;
		THRESHSCALE = 40;
		MAXLINK = 10;
		VOTERATE = 5;
		OPPOANG = 2.0/9;
	}
}

void drawInnerBorder(Mat& src, double k, int x0, int y0){
	double b = y0 - k*x0;
	Point points[4];
	points[0] = Point(0,b);
	points[1] = Point(-b/k,0);
	points[2] = Point(src.cols,k*src.cols+b);
	points[3] = Point((src.rows-b)/k,src.rows);

	int p1,p2;
	bool find = false;
	for(p1=0;p1<4;p1++){
		if(points[p1].x>=0&&points[p1].x<=src.cols&&points[p1].y>=0&&points[p1].y<=src.rows){
			for(p2=p1+1;p2<4;p2++){
				if(points[p2].x>=0&&points[p2].x<=src.cols&&points[p2].y>=0&&points[p2].y<=src.rows){
					find = true;
					break;
				}
			}
		}
		if(find)
			break;
	}
	line( src, points[p1], points[p2], CV_RGB(0,255,0),6);
}

void showResult(Mat& src, Mat& cross, Mat& turned){
	//output it
	std::vector<cv::Point2f> corners;
	//cout<<"size "<<src.cols<<" "<<src.rows<<endl;
	for (int i = 0; i < finalines.size(); i++)
	{
		for (int j = i+1; j < finalines.size(); j++)
		{
			cv::Point2f pt = computeIntersect(finalines[i], finalines[j]);

			if(pt.x<0&&pt.x>-10) pt.x = 0;
			if(pt.y<0&&pt.y>-10) pt.y = 0;
			if(pt.x>src.cols&&pt.x<src.cols+10) pt.x = src.cols;
			if(pt.y>src.rows&&pt.y<src.rows+10) pt.y = src.rows;
			if (pt.x >= 0 && pt.y >= 0&&pt.x<=src.cols&&pt.y<=src.rows){

				corners.push_back(pt);
			}

		}
	}

	std::vector<cv::Point2f> approx;
	cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true) * 0.02, true);

	if (approx.size() != 4)
	{
		std::cout << "The object is not quadrilateral!" << std::endl;
		return;
	}

	// Get mass center
	for (int i = 0; i < corners.size(); i++)
		center += corners[i];
	center *= (1. / corners.size());

	sortCorners(corners, center);
	//cout<<"center "<<center.x<<" "<<center.y<<endl;
	for(int i=0;i<4;i++){
		//cout<<"corner "<<corners[i].x<<" "<<corners[i].y<<endl;
	}
	if (corners.size() == 0){
		std::cout << "The corners were not sorted correctly!" << std::endl;
		return;
	}

	cv::Mat dst = src.clone();

	// Draw lines
	for (int i = 0; i < finalines.size(); i++)
	{
		cv::Vec4i v = finalines[i];
		Point pt1(v[0], v[1]),pt2(v[2], v[3]);
		if(pt2.x!=pt1.x)
			drawInnerBorder(dst,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		else{
			pt1.y = 0;
			pt2.y = dst.rows;
			cv::line( dst, pt1, pt2, CV_RGB(0,255,0),6);
		}
		//cv::line(dst, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0,255,0));
	}

	// Draw corner points
	cv::circle(dst, corners[0], 3, CV_RGB(255,0,0), 6);
	cv::circle(dst, corners[1], 3, CV_RGB(0,255,0), 6);
	cv::circle(dst, corners[2], 3, CV_RGB(0,0,255), 6);
	cv::circle(dst, corners[3], 3, CV_RGB(255,255,255), 6);

	//turn the angle
    double u0 = src.cols/2.0;
    double v0 = src.rows/2.0;

	double mp1[3];
	pointToVecP(corners[3],mp1);
	double mp2[3];
	pointToVecP(corners[2],mp2);
	double mp3[3];
	pointToVecP(corners[0],mp3);
	double mp4[3];
	pointToVecP(corners[1],mp4);

	double cha14[3],cha24[3],cha34[3];
	//cout<<"mp1: "<<mp1[0]<<", "<<mp1[1]<<", "<<mp1[2]<<endl;
	//cout<<"mp2: "<<mp2[0]<<", "<<mp2[1]<<", "<<mp2[2]<<endl;
	//cout<<"mp3: "<<mp3[0]<<", "<<mp3[1]<<", "<<mp3[2]<<endl;
	//cout<<"mp4: "<<mp4[0]<<", "<<mp4[1]<<", "<<mp4[2]<<endl;
	chacheng(mp1,mp4,cha14,3);
	chacheng(mp2,mp4,cha24,3);
	chacheng(mp3,mp4,cha34,3);
	double k2 = diancheng(cha14,mp3,3)/diancheng(cha24,mp3,3);
	double k3 = diancheng(cha14,mp2,3)/diancheng(cha34,mp2,3);

	//std::cout<<"k2k3 "<<k2<<" "<<k3<<std::endl;
	double n2[3],n3[3];
	for(int i=0;i<3;i++){
		n2[i] = k2*mp2[i]-mp1[i];
		n3[i] = k3*mp3[i]-mp1[i];
	}
	/*
	//cout<<"n2: ";
	for(int i=0;i<3;i++)
		//cout<<n2[i]<<",";
	//cout<<endl;
	//cout<<"n3: ";
	for(int i=0;i<3;i++)
		//cout<<n3[i]<<",";
	//cout<<endl;
	 * */
	double fk1 = -(1.0/(n2[2]*n3[2]));
	double fk2 = n2[0]*n3[0]-(n2[0]*n3[2]+n2[2]*n3[0])*u0+n2[2]*n3[2]*u0*u0;
	double fk3 = n2[1]*n3[1]-(n2[1]*n3[2]+n2[2]*n3[1])*v0+n2[2]*n3[2]*v0*v0;
	double f2 = fk1*(fk2+fk3);

	//std::cout<<"f2 "<<f2<<std::endl;

	double bl1 = (n2[0]-n2[2]*u0)*n2[0]+(n2[1]-n2[2]*v0)*n2[1]+(u0*u0+v0*v0+f2)*n2[2]*n2[2]-(u0*n2[0]+v0*n2[1])*n2[2];
	double bl2 = (n3[0]-n3[2]*u0)*n3[0]+(n3[1]-n3[2]*v0)*n3[1]+(u0*u0+v0*v0+f2)*n3[2]*n3[2]-(u0*n3[0]+v0*n3[1])*n3[2];

	double factor = 2;

	double bl0 = 2*max(fabs(mp1[0]-mp2[0]),fabs(mp3[0]-mp4[0]))/max(fabs(mp1[1]-mp3[1]),fabs(mp2[1]-mp4[1]));
	double bl = bl0;

	if(bl1*bl2>0)
		bl = sqrt(bl1/bl2);

	/**/
	if(bl0>0&&bl<0||bl0<0&&bl>0)
		bl = bl0;

	if(bl>3||bl<0.3)
		bl = bl0;

	//std::cout<<"bl0 "<<bl0<<",bl1 "<<bl1<<",bl2 "<<bl2<<",bl "<<bl<<std::endl;

	int width = src.cols>bl*src.rows?src.cols:bl*src.rows;
	int height = (int)(width/bl);

	cv::Mat quad = cv::Mat::zeros(height, width, CV_8UC3);
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, 0));
	quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
	quad_pts.push_back(cv::Point2f(0, quad.rows));

	cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);
	cv::warpPerspective(src, quad, transmtx, quad.size());

	cross = dst.clone();
	turned = quad.clone();

	//cv::imshow("imageF", dst);
	//cv::imshow("quadrilateral", quad);
	//waitKey();
}

//procMode: 0, default; 1, big; 2, micro; 3, deep1
int process(cv::Mat src0,cv::Mat src, int procMode, Mat& cross, Mat& turned){

	//step0: to gray picture
    cv::Mat bw,bw0;

	cv::cvtColor(src, bw, CV_BGR2GRAY);
	//blur(bw,bw, Size(5,5));
	//step1: edge detection
	cv::Mat pic1;
	int ddepth = 3;

	cv::Sobel(bw,grad_x,ddepth,1,0);
	cv::convertScaleAbs(grad_x,abs_grad_x);

	cv::Sobel(bw,grad_y,ddepth,0,1);
	cv::convertScaleAbs(grad_y,abs_grad_y);

	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);

	cv::threshold(grad,pic1,40.0,255,CV_THRESH_TOZERO);
	//cv::threshold(grad,pic1,180.0,255,CV_THRESH_BINARY);
	/**************************************************/
//	for(int i=0;i<2;i++){
//	bw = pic1.clone();
//	cv::Sobel(bw,grad_x,ddepth,1,0);
//	cv::convertScaleAbs(grad_x,abs_grad_x);
//
//	cv::Sobel(bw,grad_y,ddepth,0,1);
//	cv::convertScaleAbs(grad_y,abs_grad_y);
//
//	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);
//
//	cv::threshold(grad,pic1,40.0,255,CV_THRESH_TOZERO);
//	}
	/**************************************************/

	//cv::threshold(grad,pic1,40.0,255,CV_THRESH_BINARY);
	std::cout<<"img size: "<<pic1.cols<<" "<<pic1.rows<<std::endl;
/*
	Size sz = Size(pic1.cols*3/5,grad.rows*3/5);
	Mat pic3 = Mat(sz,CV_32S);
	cv::resize(pic1, pic3, sz);
    cv::imshow("grad", pic3);
	cv::waitKey();
*/
	//step2: Hough transform
	IplImage iplimg = pic1;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	//lines = cvHoughLines3( &iplimg, storage, CV_HOUGH_STANDARD, 5, CV_PI/90, 70, 30, 10 );

	std::vector<cv::Vec4i> lines0;
	cv::HoughLinesP(pic1, lines0, 5, CV_PI/90, 100, 70, 20);

	lines = convertToPolar(lines0,storage, pic1);
	//step3: quadrangle formation
	int i = 0;
	std::vector<int> fakeLines(lines->total);
	int fakes = 0;
	for( ; i < lines->total; i++ )
	{
		//std::cout<<i<<" of "<<lines->total<<std::endl;
		CvLinePolar2* line = (CvLinePolar2*)cvGetSeqElem(lines,lineSorted[i]);

		if((procMode==1&&i<31)||line->votes*VOTERATE>((CvLinePolar2*)cvGetSeqElem(lines,0))->votes){

			cv::Point pt1, pt2;
			float rho = line->rho, theta = line->angle;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = line->x1;//cvRound(x0 + 1000*(-b));
			pt1.y = line->y1;//cvRound(y0 + 1000*(a));
			pt2.x = line->x2;//cvRound(x0 - 1000*(-b));
			pt2.y = line->y2;//cvRound(y0 - 1000*(a));
			cv::Mat pic11 = pic1.clone();

			/*
			if(!isLine(pic11,pt1, pt2, THRESHOLD[0], SIZE[0], 1, procMode))
			{
				fakeLines[i]=0;
				fakes++;
			}
			else
				fakeLines[i]=1;
			*/
			///
			fakeLines[i] = 1;
			//std::cout<<"test line "<<fakeLines[i]<<std::endl;
			if(fakeLines[i]==1&&(procMode==3&&(notRectLineLv2(line->angle,line->rho)||!isBlackBorder(pic11,pt1, pt2, 17*3-2, 11*3-2, 1,i)))){
				fakeLines[i]=0;
				fakes++;
			}

/*  //if(fabs(line->angle-CV_PI/2)<CV_PI/6){
			cv::line( pic11, pt1, pt2, CV_RGB(255,255,255),6);

			std::cout<<"LINE "<<i<<": "<<line->score<<" "<<line->rho<<" "<<line->angle<<std::endl;

			Size sz = Size(pic11.cols*3/5,pic11.rows*3/5);
			Mat pic3 = Mat(sz,CV_32S);
			cv::resize(pic11, pic3, sz);
			cv::imshow("image", pic3);
			//cv::imshow("image", pic11);
			cv::waitKey();
		//}
*/

		}
		else{
			if(procMode==1&&i<31)
				continue;
			else
				break;
		}

	}
	int cut = i;
	std::cout<<"lines number "<<cut<<std::endl;
	std::cout<<"fake lines "<<fakes<<std::endl;
	if(procMode<3&&cut>1000) cut=1000;

	std::vector<OppositeLines> opplineVector;
	std::vector<int> lines2;

	int width = grad.cols;
	int height = grad.rows;

	//iteration can be merged with the previous one
	for(i=0;i<cut;i++){
		if(fakeLines[i]==0)
			continue;
		CvLinePolar2* l1 = (CvLinePolar2*)cvGetSeqElem(lines,lineSorted[i]);
		//if(fabs(fabs(l1->angle)-CV_PI/2)<0.00001) continue;

		for(int j=i+1;j<cut;j++){
			if(fakeLines[j]==0)
				continue;
			CvLinePolar2* l2 = (CvLinePolar2*)cvGetSeqElem(lines,lineSorted[j]);
			//if(fabs(fabs(l2->angle)-CV_PI/2)<0.00001) continue;

			double dangle = fabs(l1->angle-l2->angle);//TODO when rho is minus...
			double drho = fabs(fabs(l1->rho)-fabs(l2->rho));
			if(l1->rho*l2->rho<0&&dangle<CV_PI)
				drho=fabs(l1->rho-l2->rho);
			double rho1 = l1->rho, rho2 = l2->rho;
			double theta1 = l1->angle, theta2 = l2->angle;
/*
			if(i==5&&j==8){
				cout<<"l1 "<<l1->angle<<" "<<l1->rho<<endl;
				cout<<"l2 "<<l2->angle<<" "<<l2->rho<<endl;
				std::cout<<"why not oppsite "<<lineSorted[i]<<" "<<lineSorted[j]<<" "<<dangle<<" "<<drho<<" "<<width<<" "<<height<<" "
						<<(rho1/(cos(theta1)+height*sin(theta1)/width)-0.5*width)*(rho2/(cos(theta2)+height*sin(theta2)/width)-0.5*width)<<std::endl;
			}
*/

			/*if((dangle>=CV_PI*5.0/6.0&&dangle<=CV_PI*7.0/6.0||
					dangle<=CV_PI*1.0/6.0&&(rho1/(cos(theta1)+height*sin(theta1)/width)-0.5*width)*(rho2/(cos(theta2)+height*sin(theta2)/width)-0.5*width)<0)
					&&drho>0.2*width&&drho>0.2*height)*/

			/*if((dangle>=CV_PI*7.0/9.0&&dangle<=CV_PI*11.0/9.0||
								dangle<=CV_PI*2.0/9.0&&(rho1/(cos(theta1)+height*sin(theta1)/width)-0.5*width)*(rho2/(cos(theta2)+height*sin(theta2)/width)-0.5*width)<0)
								&&drho>0.2*width&&drho>0.2*height)*/
			OPPOANG = 0.25;
			if((dangle>=CV_PI*(1.0-OPPOANG)&&dangle<=CV_PI*(OPPOANG+1.0)&&(drho=l1->rho+l2->rho)||
				(dangle<=CV_PI*OPPOANG||(dangle>=(2.0-OPPOANG)*CV_PI&&dangle<=2.0*CV_PI))&&(rho1/(cos(theta1)+height*sin(theta1)/width)-0.5*width)*(rho2/(cos(theta2)+height*sin(theta2)/width)-0.5*width)<0)
				&&(drho>0.02*width&&drho>0.02*height)){

				OppositeLines oppLines;
				oppLines.one = lineSorted[i];
				oppLines.two = lineSorted[j];
/**/
				if(i==5&&j==8)
					std::cout<<"carrot: "<<opplineVector.size()<<std::endl;

				if(std::find(lines2.begin(),lines2.end(),i)==lines2.end())
					lines2.push_back(i);

				if(std::find(lines2.begin(),lines2.end(),j)==lines2.end())
					lines2.push_back(j);

				opplineVector.push_back(oppLines);
			}

		}
	}

	std::cout<<"opposite pair "<<opplineVector.size()<<std::endl;
	int vecsize = opplineVector.size()>500?500:opplineVector.size();
	/*
	for(int s=0;s<lines2.size();s++){
		CvLinePolar2* line = (CvLinePolar2*)cvGetSeqElem(lines,lines2[s]);
		float rho = line->rho, theta = line->angle;
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		cv::line( pic1, pt1, pt2, CV_RGB(255,255,255));
	}
*/
	/*
	for(int s=0;s<opplineVector.size();s++){
		cv::Mat pic2 = pic1.clone();
		CvLinePolar2* line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[s].one);
		float rho = line->rho, theta = line->angle;
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),2);

		//if(pt2.x!=pt1.x)
		//	drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//else{
		//	pt1.y = 0;
		//	pt2.y = pic2.rows;
		//	cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
		//}


		line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[s].two);
		rho = line->rho;
		theta = line->angle;
		a = cos(theta);
		b = sin(theta);
		x0 = a*rho;
		y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),2);
		//drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//Size sz = Size(pic2.cols/3,pic2.rows/3);
		//Mat pic3 = Mat(sz,CV_32S);
		//resize(pic2, pic3, sz);

		std::cout<<"s,one,two "<<s<<" "<<opplineVector[s].one<<" "<<opplineVector[s].two<<std::endl;
		if(fabs(line->angle-CV_PI/2)<CV_PI/6){
			cv::imshow("image", pic2);
			cv::waitKey();
		}

	}
*/

	std::vector<Quadrangle> quadVector;
	int finalK = -1;
	int finalL = -1;
	double mcirc = -1;
	double minDistToM = 9999;
	double minAngleSum = 9999;
	double maxScore = -1;
	int vecsize1 = 120;
	int vecsize2 = vecsize;
	int lStart = 0;

	for(int k=0;k<vecsize1&&k<opplineVector.size();k++){
		OppositeLines pair1 = opplineVector.at(k);
		CvLinePolar2 *clines[4];
		clines[0] = (CvLinePolar2*)cvGetSeqElem(lines,pair1.one);
		clines[1] = (CvLinePolar2*)cvGetSeqElem(lines,pair1.two);
		cv::Vec4i xylines[4];
		for(int m=0;m<2;m++){
			float rho = clines[m]->rho, theta = clines[m]->angle;
			cv::Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			xylines[m][0] = cvRound(x0 + 1000*(-b));
			xylines[m][1] = cvRound(y0 + 1000*(a));
			xylines[m][2] = cvRound(x0 - 1000*(-b));
			xylines[m][3] = cvRound(y0 - 1000*(a));
		}

		for(int l=max(k+1,lStart);l<vecsize2&&l<opplineVector.size();l++){

			//cout<<"KL: "<<k<<" "<<l<<endl;
			if(finalK<0&&finalL<0&&l==vecsize2-1&&k==vecsize1-2){
				cout<<"BIGBIG"<<endl;
				lStart = vecsize2;
				vecsize2 += 150;
				k = 0;
			}
			OppositeLines pair2 = opplineVector.at(l);
			if(pair1.one==pair2.one||pair1.one==pair2.two||pair1.two==pair2.one||pair1.two==pair2.two)
				continue;

			//std::cout<<"here"<<std::endl;
			clines[2] = (CvLinePolar2*)cvGetSeqElem(lines,pair2.one);
			clines[3] = (CvLinePolar2*)cvGetSeqElem(lines,pair2.two);

			for(int m=2;m<4;m++){
				float rho = clines[m]->rho, theta = clines[m]->angle;
				cv::Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				xylines[m][0] = cvRound(x0 + 1000*(-b));
				xylines[m][1] = cvRound(y0 + 1000*(a));
				xylines[m][2] = cvRound(x0 - 1000*(-b));
				xylines[m][3] = cvRound(y0 - 1000*(a));
			}
/*
			if(k==0&&l==9){
				cv::Mat pic2 = pic1.clone();
				for(int n=0;n<4;n++){
					cv::Point pt1, pt2;
					pt1.x = xylines[n][0];
					pt1.y = xylines[n][1];
					pt2.x = xylines[n][2];
					pt2.y = xylines[n][3];
					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
					cv::imshow("image", pic2);
					std::cout<<clines[n]->angle<<" "<<clines[n]->rho<<std::endl;
					cv::waitKey();
				}

			}
*/
//			if(k==0&&l==12)
//				cout<<"here1"<<endl;

			int padding = 10.0;
			cv::Point2f pt[4];
			bool pass1 = true;
			for(int n=0;n<4;n++){

				pt[n] = computeIntersect(xylines[n/2], xylines[2+n%2]);
				/*
				if(k==1&&l==5){
					cv::Mat pic2 = pic1.clone();
					cv::Point pt1, pt2;
					pt1.x = xylines[n/2][0];
					pt1.y = xylines[n/2][1];
					std::cout<<"("<<pt1.x<<","<<pt1.y<<")"<<std::endl;
					pt2.x = xylines[n/2][2];
					pt2.y = xylines[n/2][3];
					std::cout<<"("<<pt2.x<<","<<pt2.y<<")"<<std::endl;
					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
					pt1.x = xylines[2+n%2][0];
					pt1.y = xylines[2+n%2][1];
					std::cout<<"("<<pt1.x<<","<<pt1.y<<")"<<std::endl;
					pt2.x = xylines[2+n%2][2];
					pt2.y = xylines[2+n%2][3];
					std::cout<<"("<<pt2.x<<","<<pt2.y<<")"<<std::endl;
					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
					std::cout<<"("<<pt[n].x<<","<<pt[n].y<<")"<<std::endl;
					cv::circle(pic2, pt[n], 3, CV_RGB(255,255,255), 2);
					cv::imshow("image", pic2);
					cv::waitKey();
				}
*/
				double dnangle = fabs(clines[n/2]->angle-clines[2+n%2]->angle);
//				if(k==0&&l==12)
//					std::cout<<"DANAGEL "<<n<<" "<<dnangle<<std::endl;
/*
				if(n>0)
					std::cout<<n<<" "<<dnangle<<std::endl;
				*/
				/**/

				if(pt[n].x<-padding||pt[n].y<-padding||pt[n].x>width+padding||pt[n].y>height+padding){
					pass1 = false;
//					if(k==0&&l==12)
//						std::cout<<"dead1"<<std::endl;
					break;
				}

				//Magic number
				if(!((dnangle>=CV_PI/3.0&&dnangle<=CV_PI*2.0/3.0)||(dnangle>=CV_PI*4.0/3.0-0.04&&dnangle<=CV_PI*5.0/3.0))){
					pass1 = false;
//					if(k==0&&l==12)
//						std::cout<<"dead2"<<std::endl;
					break;
				}
			}

			if(!pass1)
				continue;

//			if(k==0&&l==12)
//				cout<<"here2"<<endl;

			double lnab = dist(pt[0],pt[1]);
			double lnbd = dist(pt[1],pt[3]);
			double lncd = dist(pt[2],pt[3]);
			double lnac = dist(pt[0],pt[2]);
			double circ = lnab+lnbd+lncd+lnac;
			if(circ<=(width+height)*0.5)
				continue;

			int sort_arr[4] = {0,1,2,3};
			for(int sort_cur1 = 0;sort_cur1<4;sort_cur1++){
				for(int sort_cur2=sort_cur1+1;sort_cur2<4;sort_cur2++){
					int ix1 = sort_arr[sort_cur1];
					int ix2 = sort_arr[sort_cur2];
					//if(k==0&&l==13)
					//	std::cout<<"angle: "<<ix1<<std::endl;
					double angv1 = normalizeAngle(clines[ix1],width,height,k,l);
					//if(k==0&&l==13)
					//	std::cout<<"angle: "<<ix2<<std::endl;
					double angv2 = normalizeAngle(clines[ix2],width,height,k,l);

					if(angv1>angv2){
						sort_arr[sort_cur1]=ix2;
						sort_arr[sort_cur2]=ix1;
					}
				}
			}
//			if(k==48&&l==63)
//				std::cout<<"herek"<<std::endl;
			if(sort_arr[0]/2!=sort_arr[1]/2&&sort_arr[1]/2!=sort_arr[2]/2){
				//success to get a quadrangle
				//if(k==270&&l==380)
				//	std::cout<<"here2"<<std::endl;
				double rho0 = clines[0]->rho;
				double rho1 = clines[1]->rho;
				double rho2 = clines[2]->rho;
				double rho3 = clines[3]->rho;

				double theta0 = rho0>=0?clines[0]->angle:-CV_PI+clines[0]->angle;
				double theta1 = rho1>=0?clines[1]->angle:-CV_PI+clines[1]->angle;
				double theta2 = rho2>=0?clines[2]->angle:-CV_PI+clines[2]->angle;
				double theta3 = rho3>=0?clines[3]->angle:-CV_PI+clines[3]->angle;

				cv::Vec4i chaline1, chaline2;
				chaline1[0] = pt[0].x;
				chaline1[1] = pt[0].y;
				chaline1[2] = pt[3].x;
				chaline1[3] = pt[3].y;

				chaline2[0] = pt[1].x;
				chaline2[1] = pt[1].y;
				chaline2[2] = pt[2].x;
				chaline2[3] = pt[2].y;

				cv::Point2d jiao = computeIntersect(chaline1, chaline2);
				cv::Point2d zhon;
				zhon.x = width/2;
				zhon.y = height/2;

				double distToM = dist(jiao,zhon);
				double angleSum = fabs(theta0-theta1)+fabs(theta2-theta3);//TODO this is a problem!That when angleSum is nearly PI...like the cat book!
                //int score = clines[0]->score+clines[1]->score+clines[2]->score+clines[3]->score;//blackScore[opplineVector.at(l).one]+blackScore[opplineVector.at(l).two]+blackScore[opplineVector.at(k).one]+blackScore[opplineVector.at(k).two];
				double score = 0;

                if(angleSum>CV_PI&&!(2*CV_PI-angleSum<CV_PI/6))
                	angleSum -= CV_PI;

                bool debug = false;
                if(k==412&&l==657)
                	debug = true;

                if(isRealQuadr(pic1,xylines,THRESHOLD[1],SIZE[1],procMode,score,debug)&&isLikeRect(clines,debug)&&
                  (procMode<3&&((angleSum<CV_PI/6||2*CV_PI-angleSum<CV_PI/6)&&maxScore<=score/*(distToM<minDistToM||distToM<20)&&(circ>1.0*mcirc||(circ>0.95*mcirc&&circ<mcirc))*/)||
				   procMode==3&&angleSum<CV_PI/3&&maxScore<=score))
				{
					finalK = k;
					finalL = l;
					mcirc = circ;
					minAngleSum=angleSum;
					minDistToM = distToM;
					maxScore=score;
				}
/*
				if((k==11&&l==20||k==32&&l==251||k==38&&l==49))
				{
					cout<<"rhos "<<rho0<<" "<<rho1<<" "<<rho2<<" "<<rho3<<endl;
					cout<<"angles "<<clines[0]->angle<<" "<<clines[1]->angle<<" "<<clines[2]->angle<<" "<<clines[3]->angle<<endl;
					cout<<"thetas "<<theta0<<" "<<theta1<<" "<<theta2<<" "<<theta3<<endl;
					std::cout<<maxScore<<" "<<score<<" "<<angleSum<<" "<<distToM<<" "<<circ<<" "<<k<<" "<<l<<" "<<finalK<<" "<<finalL<<std::endl;
              //Quadrangle quad;
				//quad.a = pt[0];
				//quad.b = pt[1];
				//quad.c = pt[2];
				//quad.d = pt[3];
				//quadVector.push_back(quad);

				//to show quadrangles
				cv::Mat pic2 = pic1.clone();
				cv::Point pt01, pt02, pt03, pt04;
				pt01.x = 0; pt01.y=height/2;
				pt02.x = width; pt02.y=height/2;
				pt03.x = width/2; pt03.y=0;
				pt04.x = width/2; pt04.y=height;
				//cv::line( pic2, pt01, pt02, CV_RGB(255,255,255));
				//cv::line( pic2, pt03, pt04, CV_RGB(255,255,255));

				CvLinePolar2* line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[k].one);
				float rho = line->rho, theta = line->angle;
				cv::Point pt1, pt2;
				double a = cos(theta), b = sin(theta);
				double x0 = a*rho, y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
//				if(pt2.x!=pt1.x)
//					drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
//				else{
//					pt1.y = 0;
//					pt2.y = pic2.rows;
//					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
//				}
				cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));

				line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[k].two);
				rho = line->rho;
				theta = line->angle;
				a = cos(theta);
				b = sin(theta);
				x0 = a*rho;
				y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
//				if(pt2.x!=pt1.x)
//					drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
//				else{
//					pt1.y = 0;
//					pt2.y = pic2.rows;
//					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
//				}
				cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));

				line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[l].one);
				rho = line->rho;
				theta = line->angle;
				a = cos(theta);
				b = sin(theta);
				x0 = a*rho;
				y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
//				if(pt2.x!=pt1.x)
//					drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
//				else{
//					pt1.y = 0;
//					pt2.y = pic2.rows;
//					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
//				}
				cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));

				line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[l].two);
				rho = line->rho;
				theta = line->angle;
				a = cos(theta);
				b = sin(theta);
				x0 = a*rho;
				y0 = b*rho;
				pt1.x = cvRound(x0 + 1000*(-b));
				pt1.y = cvRound(y0 + 1000*(a));
				pt2.x = cvRound(x0 - 1000*(-b));
				pt2.y = cvRound(y0 - 1000*(a));
//				if(pt2.x!=pt1.x)
//					drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
//				else{
//					pt1.y = 0;
//					pt2.y = pic2.rows;
//					cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
//				}
				cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
//				Size sz = Size(pic2.cols/3,pic2.rows/3);
//				Mat pic3 = Mat(sz,CV_32S);
//				resize(pic2, pic3, sz);
//				cv::imshow("imageComp", pic3);
				cv::imshow("imageComp", pic2);
				cv::waitKey();

				}*/
			}
		}
	}
	/**/

	if(finalK>=0&&finalL>=0)
	{
		std::cout<<"final "<<finalK<<" "<<finalL<<std::endl;

		cv::Mat pic2 = pic1.clone();
		CvLinePolar2* line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[finalK].one);
		float rho = line->rho, theta = line->angle;
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		finalines[0][0] = pt1.x; finalines[0][1]=pt1.y; finalines[0][2] = pt2.x; finalines[0][3]=pt2.y;
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
		//if(pt2.x!=pt1.x)
		//	drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//else{
		//	pt1.y = 0;
		//	pt2.y = pic2.rows;
		//	cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
		//}
		//std::cout<<"line "<<line->angle<<" "<<line->rho<<std::endl;
		//imshow("imageF", pic2);
		//waitKey();

		line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[finalK].two);
		rho = line->rho;
		theta = line->angle;
		a = cos(theta);
		b = sin(theta);
		x0 = a*rho;
		y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		finalines[1][0] = pt1.x; finalines[1][1]=pt1.y; finalines[1][2] = pt2.x; finalines[1][3]=pt2.y;
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
		//if(pt2.x!=pt1.x)
		//	drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//else{
		//	pt1.y = 0;
		//	pt2.y = pic2.rows;
		//	cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
		//}
		//std::cout<<"line "<<line->angle<<" "<<line->rho<<std::endl;
		//imshow("imageF", pic2);
		//waitKey();

		line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[finalL].one);
		rho = line->rho;
		theta = line->angle;
		a = cos(theta);
		b = sin(theta);
		x0 = a*rho;
		y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		finalines[2][0] = pt1.x; finalines[2][1]=pt1.y; finalines[2][2] = pt2.x; finalines[2][3]=pt2.y;
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
		//if(pt2.x!=pt1.x)
		//	drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//else{
		//	pt1.y = 0;
		//	pt2.y = pic2.rows;
		//	cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
		//}
		//std::cout<<"line "<<line->angle<<" "<<line->rho<<std::endl;
		//imshow("imageF", pic2);
		//waitKey();

		line = (CvLinePolar2*)cvGetSeqElem(lines,opplineVector[finalL].two);
		rho = line->rho;
		theta = line->angle;
		a = cos(theta);
		b = sin(theta);
		x0 = a*rho;
		y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		finalines[3][0] = pt1.x; finalines[3][1]=pt1.y; finalines[3][2] = pt2.x; finalines[3][3]=pt2.y;
		cv::line( pic2, pt1, pt2, CV_RGB(255,255,255));
		//if(pt2.x!=pt1.x)
		//	drawInnerBorder(pic2,(0.0+pt2.y-pt1.y)/(0.0+pt2.x-pt1.x),pt1.x,pt1.y);
		//else{
		//	pt1.y = 0;
		//	pt2.y = pic2.rows;
		//	cv::line( pic2, pt1, pt2, CV_RGB(255,255,255),6);
		//}
		//std::cout<<"line "<<line->angle<<" "<<line->rho<<std::endl;
		showResult(src0,cross,turned);

		//Size sz = Size(pic2.cols/3,pic2.rows/3);
		//Mat pic3 = Mat(sz,CV_32S);
		//resize(pic2, pic3, sz);
//		cv::imshow("imageF", pic2);
//		cv::waitKey();
		return 0;
	}

	return -1;
}

//procMode: 0, default; 1, big; 2, micro; 3, deep1
int mainProc(cv::Mat src, int procMode, Mat& cross, Mat& turned){

	for(int run=0;run<1;run++){

		modifyAttr(procMode,run);
		int result = process(src, src, procMode,cross,turned);

		if(result!=-1)
			return result;
	}
	return -1;
}

int mainProc2(Mat input, cv::Mat src, int procMode, Mat& cross, Mat& turned){

	for(int run=0;run<1;run++){

		modifyAttr(procMode,run);
		int result = process(input, src, procMode,cross,turned);

		if(result!=-1)
			return result;
	}
	return -1;
}

void getCountHistoGram(Mat& src){
	//step0: to gray picture
	cv::Mat bw,bw0;

	cv::cvtColor(src, bw, CV_BGR2GRAY);

	//step1: edge detection
	cv::Mat pic1;
	int ddepth = 3;

	cv::Sobel(bw,grad_x,ddepth,1,0);
	cv::convertScaleAbs(grad_x,abs_grad_x);

	cv::Sobel(bw,grad_y,ddepth,0,1);
	cv::convertScaleAbs(grad_y,abs_grad_y);

	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);

	int histSize = 30;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

    int hist_w = grad.cols; int hist_h = grad.rows;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat hist = Mat::zeros(1,histSize,CV_16UC1);

    int dim[] = {0};
    vector<Mat> mats;
    mats.push_back(grad);

    for(int i=0;i<histSize;i++){
    	int start_x = i*bin_w;
    	unsigned short count0 = 0;
    	for(int j=start_x;j<start_x+bin_w;j++){
    		for(int k=0;k<grad.rows;k++){
    			CvPoint2D32f ptr = cvPoint2D32f(j, k);
    			int val = grad.at<uchar>(ptr);
    			if(val>40)
    				count0++;
    		}
    	}


    	hist.at<unsigned short>(i)=count0;
    	//cout<<i<<", "<<count0<<" "<<hist.at<unsigned short>(i)<<endl;
    }
    /*
    for( int i = 1; i < histSize; i++ )
    {
    	cout<<i<<" "<<hist.at<unsigned short>(i-1)<<endl;
    }
    */
	//calcHist(&mats[0],1,&dim[0],Mat(),hist,1,&histSize, &histRange, uniform, accumulate);

	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	cout<<"out size: "<<hist_h<<" "<<hist_w<<endl;

    for( int i = 1; i < histSize; i++ )
    {
    	cout<<"Point: "<<i<<" "<<bin_w*(i-1)<<" "<<hist.at<unsigned short>(i-1)<<endl;

        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<unsigned short>(i-1)) ) ,
	    	    	     Point( bin_w*(i), hist_h - cvRound(hist.at<unsigned short>(i)) ),
					     Scalar( 255, 255, 255), 2, 8, 0  );

    }

    imshow("grad",grad);
    namedWindow("calcHist Demo");
    imshow("calcHist Demo", histImage );

    waitKey(0);
}

int getGradient(Mat& src,Mat& grad2){
	cv::Mat bw,bw0;
	Mat grad1;

	cv::cvtColor(src, bw, CV_BGR2GRAY);

	cv::Mat pic1;
	int ddepth = 3;

	cv::Sobel(bw,grad_x,ddepth,1,0);
	cv::convertScaleAbs(grad_x,abs_grad_x);

	cv::Sobel(bw,grad_y,ddepth,0,1);
	cv::convertScaleAbs(grad_y,abs_grad_y);

	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad1);

	cv::threshold(grad1,grad2,40.0,255,CV_THRESH_TOZERO);
	return 0;
}

int getCropped(Mat& src, vector<Mat>& grads){

	int size = 16;
	if(src.rows<size||src.cols<size) return -1;

	Mat src1;
	cvtColor(src, src1, CV_BGR2GRAY);
	for(int i=0;i<src1.cols/size;i++){
		int left = i*size;
		for(int j=0;j<src1.rows/size;j++){
			int top = j*size;

			Rect roi(left, top, size, size);
			Mat image_roi = src1(roi);
			Mat cropimage;
			image_roi.copyTo(cropimage);
			if(countNonZero(cropimage)>30)
				grads.push_back(cropimage);
		}
		if(src1.rows%size!=0){
			Rect roi(left, src1.rows-size, size, size);
			Mat image_roi = src1(roi);
			Mat cropimage;
			image_roi.copyTo(cropimage);
			if(countNonZero(cropimage)>30)
				grads.push_back(cropimage);
		}
	}
	if(src1.cols%size!=0){
		int left = src1.cols-size;
		for(int j=0;j<src1.rows/size;j++){
			int top = j*size;

			Rect roi(left, top, size, size);
			Mat image_roi = src1(roi);
			Mat cropimage;
			image_roi.copyTo(cropimage);
			if(countNonZero(cropimage)>30)
				grads.push_back(cropimage);
		}
		if(src1.rows%size!=0){
			Rect roi(left, src1.rows-size, size, size);
			Mat image_roi = src1(roi);
			Mat cropimage;
			image_roi.copyTo(cropimage);
			if(countNonZero(cropimage)>30)
				grads.push_back(cropimage);
		}
	}
}
