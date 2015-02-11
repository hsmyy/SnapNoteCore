
//#ifndef TEXTAREA_H
//#define TEXTAREA_H
//

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/types_c.h"
//#include "opencv2/imgproc/imgproc_c.h"
//#include <opencv2/opencv.hpp>
//#include <cmath>
//#include <iostream>
//#include "opencv2/text.hpp"
//#include "opencv2/core/utility.hpp"
//#include <string.h>
//#include <stdlib.h>
//#include <dirent.h>
//#include <sys/stat.h>
//#include <unistd.h>
//
//using namespace std;
//using namespace cv;
//using namespace cv::text;
//
//size_t min(size_t x, size_t y, size_t z)
//{
//    return x < y ? min(x,z) : min(y,z);
//}
//
//size_t edit_distance(const string& A, const string& B)
//{
//    size_t NA = A.size();
//    size_t NB = B.size();
//
//    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));
//
//    for (size_t a = 0; a <= NA; ++a)
//        M[a][0] = a;
//
//    for (size_t b = 0; b <= NB; ++b)
//        M[0][b] = b;
//
//    for (size_t a = 1; a <= NA; ++a)
//        for (size_t b = 1; b <= NB; ++b)
//        {
//            size_t x = M[a-1][b] + 1;
//            size_t y = M[a][b-1] + 1;
//            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
//            M[a][b] = min(x,y,z);
//        }
//
//    return M[A.size()][B.size()];
//}
//
//bool isRepetitive(const string& s)
//{
//    int count = 0;
//    for (int i=0; i<(int)s.size(); i++)
//    {
//        if ((s[i] == 'i') ||
//                (s[i] == 'l') ||
//                (s[i] == 'I'))
//            count++;
//    }
//    if (count > ((int)s.size()+1)/2)
//    {
//        return true;
//    }
//    return false;
//}
//
//
//void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
//{
//    for (int r=0; r<(int)group.size(); r++)
//    {
//        ERStat er = regions[group[r][0]][group[r][1]];
//        if (er.parent != NULL) // deprecate the root region
//        {
//            int newMaskVal = 255;
//            int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
//            floodFill(channels[group[r][0]],segmentation,Point(er.pixel%channels[group[r][0]].cols,er.pixel/channels[group[r][0]].cols),
//                      Scalar(255),0,Scalar(er.level),Scalar(0),flags);
//        }
//    }
//}
//
//bool   sort_by_lenght(const string &a, const string &b){return (a.size()>b.size());}
//
//int detectText1(Mat& src, Mat& rst){
//	vector<Mat> channels;
//	Mat image,grad_x,abs_grad_x,grad_y,abs_grad_y,grad;
//	image = src.clone();
//	Mat grey;
//	cv::cvtColor(image, grey, CV_BGR2GRAY);
//	//threshold(img, img, 180, 255, cv::THRESH_BINARY);
//	channels.push_back(grey);
//    channels.push_back(255-grey);
///*
//
//	int ddepth = 3;
//
//	cv::Sobel(grey,grad_x,ddepth,1,0);
//	cv::convertScaleAbs(grad_x,abs_grad_x);
//
//	cv::Sobel(grey,grad_y,ddepth,0,1);
//	cv::convertScaleAbs(grad_y,abs_grad_y);
//
//	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);
//	cv::threshold(grad,grad,40.0,255,CV_THRESH_TOZERO);
//	//bitwise_not(img, img);
//	channels.push_back(grad);
//     */
//    Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("/home/litton/opencv_contrib/opencv_contrib-master/modules/text/samples/trained_classifierNM1.xml"),8,0.00015f,0.13f,0.2f,true,0.1f);
//	Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("/home/litton/opencv_contrib/opencv_contrib-master/modules/text/samples/trained_classifierNM2.xml"),0.5);
//
//	vector<vector<ERStat> > regions(channels.size());
//
//	// Apply the default cascade classifier to each independent channel (could be done in parallel)
//	for (int c=0; c<(int)channels.size(); c++)
//	{
//		er_filter1->run(channels[c], regions[c]);
//		er_filter2->run(channels[c], regions[c]);
//	}
//
//    Mat out_img_decomposition= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
//    vector<Vec2i> tmp_group;
//    for (int i=0; i<(int)regions.size(); i++)
//    {
//        for (int j=0; j<(int)regions[i].size();j++)
//        {
//            tmp_group.push_back(Vec2i(i,j));
//        }
//        Mat tmp= Mat::zeros(image.rows+2, image.cols+2, CV_8UC1);
//        er_draw(channels, regions, tmp_group, tmp);
//        if (i > 0)
//            tmp = tmp / 2;
//        out_img_decomposition = out_img_decomposition | tmp;
//        tmp_group.clear();
//    }
//
//    //grouping
//    vector< vector<Vec2i> > nm_region_groups;
//    vector<Rect> nm_boxes;
//    erGrouping(image, channels, regions, nm_region_groups, nm_boxes,ERGROUPING_ORIENTATION_HORIZ);
//    cout<<"rect num "<<nm_boxes.size()<<endl;
//    int left = 9999;
//    int up = 9999;
//    int right = -1;
//    int bottom = -1;
//
//    for(int i=0;i<nm_boxes.size();i++)
//    {
//    	if(left>nm_boxes[i].x)
//    		left = nm_boxes[i].x;
//    	if(right<nm_boxes[i].x+nm_boxes[i].width)
//    		right = nm_boxes[i].x+nm_boxes[i].width;
//    	if(up>nm_boxes[i].y)
//			up = nm_boxes[i].y;
//		if(bottom<nm_boxes[i].y+nm_boxes[i].height)
//			bottom = nm_boxes[i].y+nm_boxes[i].height;
//    	rectangle(image,nm_boxes[i],CV_RGB(0,255,0),2);
//    }
//
//    Rect rect;
//    rect.x = left;
//    rect.y = up;
//    rect.width = abs(right-left);
//    rect.height = abs(bottom-up);
//    //rectangle(image,rect,CV_RGB(0,255,0),2);
//	rst = image.clone();
//    imshow("imageF", image);
//	waitKey();
//}
//
//int changfang(cv::Vec4i line){
//	return (line[0]-line[2])*(line[0]-line[2])+(line[1]-line[3])*(line[1]-line[3]);
//}
//
//bool isQuadr2(vector<cv::Vec4i> lines,int minB1,int maxB1,int i,int j,bool smallAng, int maxGap){
//
//	if(i==41&&j==48){
//		cout<<fabs(fabs(lines[i][1]-lines[i][3])-maxGap)<<endl;
//		cout<<fabs(fabs(lines[j][1]-lines[j][3])-maxGap)<<endl;
//		cout<<min(fabs(lines[minB1][0]-lines[minB1][2]),fabs(lines[maxB1][0]-lines[maxB1][2]))<<endl;
//	}
//	if(smallAng){
//		if(fabs(fabs(lines[i][1]-lines[i][3])-maxGap)<20&&
//				fabs(fabs(lines[j][1]-lines[j][3])-maxGap)<20&&
//				fabs(lines[i][0]-lines[j][0])>=(min(fabs(lines[minB1][0]-lines[minB1][2]),fabs(lines[maxB1][0]-lines[maxB1][2]))-20)&&
//				(fabs(lines[i][0]-lines[minB1][0])<20||fabs(lines[i][0]-lines[minB1][2])<20||fabs(lines[i][0]-lines[maxB1][0])<20||fabs(lines[i][0]-lines[maxB1][2])<20)&&
//				(fabs(lines[j][0]-lines[minB1][0])<20||fabs(lines[j][0]-lines[minB1][2])<20||fabs(lines[j][0]-lines[maxB1][0])<20||fabs(lines[j][0]-lines[maxB1][2])<20)){
//			return true;
//		}
//	}
//	else{
//
//	}
//	return false;
//}
//
//void drawInnerBorder(Mat& src, double k, int x0, int y0){
//	double b = y0 - k*x0;
//	Point points[4];
//	points[0] = Point(0,b);
//	points[1] = Point(-b/k,0);
//	points[2] = Point(src.cols,k*src.cols+b);
//	points[3] = Point((src.rows-b)/k,src.rows);
//
//	int p1,p2;
//	bool find = false;
//	for(p1=0;p1<4;p1++){
//		if(points[p1].x>=0&&points[p1].x<=src.cols&&points[p1].y>=0&&points[p1].y<=src.rows){
//			for(p2=p1+1;p2<4;p2++){
//				if(points[p2].x>=0&&points[p2].x<=src.cols&&points[p2].y>=0&&points[p2].y<=src.rows){
//					find = true;
//					break;
//				}
//			}
//		}
//		if(find)
//			break;
//	}
//	line( src, points[p1], points[p2], CV_RGB(255,0,0),1);
//}
//
//int detectText2(Mat& src){
//	Mat image,grad_x,abs_grad_x,grad_y,abs_grad_y,grad;
//	image = src.clone();
//	Mat grey;
//	cv::cvtColor(image, grey, CV_BGR2GRAY);
//
//	int ddepth = 3;
//
//	cv::Sobel(grey,grad_x,ddepth,1,0);
//	cv::convertScaleAbs(grad_x,abs_grad_x);
//
//	cv::Sobel(grey,grad_y,ddepth,0,1);
//	cv::convertScaleAbs(grad_y,abs_grad_y);
//
//	cv::addWeighted( abs_grad_x, 1, abs_grad_y, 1, 0, grad);
//	cv::threshold(grad,grad,40.0,255,CV_THRESH_TOZERO);
//
//	std::vector<cv::Vec4i> lines;
//	cv::HoughLinesP(grad, lines, 1, CV_PI/180, 100, 70, 20);
//
//	//imshow("grad", grad);
//	//waitKey();
//
//	int vsize = 0;
//	vector<double> tpK(1000);
//	vector<int> tpN(1000);
//	vector<vector<int> > tpV(1000);
//
//	int vertN = 0;
//
//
//	for(int i=0;i<lines.size();i++){
//		cv::Vec4i v = lines[i];
//		if(v[0]>=v[2]-2&&v[0]<=v[2]+2)
//			vertN ++;
//		else{
//			double k = (0.0+v[3]-v[1])/(0.0+v[2]-v[0]);
//			bool found = false;
//			for(int j=0;j<vsize;j++){
//				if(fabs(k-tpK[j])<0.18){
//					found = true;
//					tpK[j] = (tpK[j]*tpN[j]+k)/(tpN[j]+1);
//					tpN[j] = tpN[j]+1;
//					tpV[j].push_back(i);
//				}
//			}
//			if(!found){
//				tpK[vsize]=k;
//				tpN[vsize]=1;
//				vector<int> v(1000);
//				v.push_back(i);
//				tpV.push_back(v);
//				vsize++;
//			}
//		}
//
//		//line( grad, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(255,255,255));
//	}
//	//cout<<lines.size()<<endl;
//	//imshow("grad", grad);
//	//waitKey();
//
//	int maxTp = -1;
//	int maxCt = 0;
//	for(int i=0;i<vsize;i++){
//		if(tpN[i]>maxCt){
//			maxCt = tpN[i];
//			maxTp = i;
//		}
//	}
//
//	int upM = 0;
//	int downM = 0;
//	int minB1 = -1;
//	int maxB1 = -1;
//
//	for(int i=0;i<tpV[maxTp].size();i++){
//		int ln = tpV[maxTp][i];
//		if(lines[ln][1]<src.rows/2&&changfang(lines[ln])>upM)
//		{
//			upM = changfang(lines[ln]);
//			minB1 = ln;
//		}
//		if(lines[ln][1]>src.rows/2&&changfang(lines[ln])>downM)
//		{
//			downM = changfang(lines[ln]);
//			maxB1 = ln;
//		}
//
//		line( src, cv::Point(lines[ln][0], lines[ln][1]), cv::Point(lines[ln][2], lines[ln][3]), CV_RGB(0,255,0));
//	}
//
//	/**/
//	imshow("imageF", src);
//	waitKey();
//
//	bool smallAng = false;
//	int maxGap = max(fabs(lines[minB1][1]-lines[maxB1][1]),fabs(lines[minB1][3]-lines[maxB1][3]));
//	if(fabs(tpK[maxTp])<1){
//		smallAng = true;
//	}
//
//	int ano1, ano2;
//	bool find = false;
//
//
//
//	for(int i=0;i<lines.size();i++){
//		if(i!=minB1&&i!=maxB1){
//			for(int j=0;j<lines.size();j++){
//				if(j!=minB1&&j!=maxB1){
//					if(isQuadr2(lines,minB1,maxB1,i,j,smallAng,maxGap))
//					{
//						find = true;
//						ano1=i;
//						ano2=j;
//						break;
//					}
//				}
//			}
//			if(find)
//				break;
//		}
//	}
//
//	Mat backg = Mat::zeros(src.rows,src.cols,CV_8UC1);
//	int up = 9999, bt = -1, left = 9999, right = -1;
//	int upx, btx, lefty, righty;
//
//	for(int i=0;i<tpV[maxTp].size();i++){
//		int ln = tpV[maxTp][i];
//		if(ln!=minB1&&ln!=maxB1){
//			if(smallAng&&lines[ln][1]>lines[minB1][1]+20&&lines[ln][1]<lines[maxB1][1]-20&&
//					min(lines[ln][0],lines[ln][2])>min(lines[minB1][0],lines[minB1][2])&&
//					max(lines[ln][0],lines[ln][2])<max(lines[maxB1][0],lines[maxB1][2])){
//
//				if(min(lines[ln][0],lines[ln][2])<left)
//				{
//					left = min(lines[ln][0],lines[ln][2]);
//					if(left==lines[ln][0])
//						lefty = lines[ln][1];
//					else
//						lefty = lines[ln][3];
//
//				}
//				if(max(lines[ln][0],lines[ln][2])>right){
//					right = max(lines[ln][0],lines[ln][2]);
//					if(right==lines[ln][0])
//						righty = lines[ln][1];
//					else
//						righty = lines[ln][3];
//
//				}
//
//				if(min(lines[ln][1],lines[ln][3])<up){
//					up = min(lines[ln][1],lines[ln][3]);
//					if(up==lines[ln][1])
//						upx = lines[ln][0];
//					else
//						upx = lines[ln][2];
//
//				}
//
//				if(max(lines[ln][1],lines[ln][3])>bt){
//					bt = max(lines[ln][1],lines[ln][3]);
//					if(bt==lines[ln][1])
//						btx = lines[ln][0];
//					else
//						btx = lines[ln][2];
//
//				}
//
//				line(backg, cv::Point(lines[ln][0], lines[ln][1]), cv::Point(lines[ln][2], lines[ln][3]), CV_RGB(255,255,255),2);
//				cout<<lines[ln][0]<<" "<<lines[ln][1]<<" "<<lines[ln][2]<<" "<<lines[ln][3]<<endl;
//				imshow("lk",backg);
//				//waitKey();
//			}
//			else{
//
//			}
//		}
//	}
//
//
//	if(find){
//		//circle(backg, Point(left,lefty), 2, CV_RGB(255,255,255));
//		//circle(backg, Point(right,righty), 2, CV_RGB(255,255,255));
//		//circle(backg, Point(upx,up), 2, CV_RGB(255,255,255));
//		//circle(backg, Point(btx,bt), 2, CV_RGB(255,255,255));
//		int lefti = max(lines[ano1][0],lines[ano1][2])<min(lines[ano2][0],lines[ano2][2])?ano1:ano2;
//		int righti = lefti==ano1?ano2:ano1;
//
//		double kl = (0.0+lines[lefti][3]-lines[lefti][1])/(0.0+lines[lefti][2]-lines[lefti][0]);
//		double kr = (0.0+lines[righti][3]-lines[righti][1])/(0.0+lines[righti][2]-lines[righti][0]);
//		double ku = (0.0+lines[minB1][3]-lines[minB1][1])/(0.0+lines[minB1][2]-lines[minB1][0]);
//		double kb = (0.0+lines[maxB1][3]-lines[maxB1][1])/(0.0+lines[maxB1][2]-lines[maxB1][0]);
//
//		drawInnerBorder(src, kl, max(0,left-30), lefty);
//		drawInnerBorder(src, kr, min(src.cols, right+30), righty);
//		drawInnerBorder(src, ku, max(0, upx-30), up);
//		drawInnerBorder(src, kb, min(src.rows, btx+30), bt);
//
//		line( src, cv::Point(lines[minB1][0], lines[minB1][1]), cv::Point(lines[minB1][2], lines[minB1][3]), CV_RGB(0,255,0),2);
//		line( src, cv::Point(lines[maxB1][0], lines[maxB1][1]), cv::Point(lines[maxB1][2], lines[maxB1][3]), CV_RGB(0,255,0),2);
//		line( src, cv::Point(lines[ano1][0], lines[ano1][1]), cv::Point(lines[ano1][2], lines[ano1][3]), CV_RGB(0,255,0),2);
//		line( src, cv::Point(lines[ano2][0], lines[ano2][1]), cv::Point(lines[ano2][2], lines[ano2][3]), CV_RGB(0,255,0),2);
//
//		cout<<"Found K "<<tpK[maxTp]<<endl;
//		cout<<min(lines[minB1][0],lines[minB1][2])<<" "<<max(lines[maxB1][0],lines[maxB1][2])<<endl;
//		cout<<left<<", "<<lefty<<endl;
//		cout<<right<<", "<<righty<<endl;
//		cout<<upx<<", "<<up<<endl;
//		cout<<btx<<", "<<bt<<endl;
//		//imshow("lk",backg);
//		imshow("imageF", src);
//		waitKey();
//	}
//	else{
//		line( src, cv::Point(lines[minB1][0], lines[minB1][1]), cv::Point(lines[minB1][2], lines[minB1][3]), CV_RGB(0,255,0),2);
//		line( src, cv::Point(lines[maxB1][0], lines[maxB1][1]), cv::Point(lines[maxB1][2], lines[maxB1][3]), CV_RGB(0,255,0),2);
//		cout<<"Can't find region";
//		imshow("imageF", src);
//		waitKey();
//	}
//}
//
//Mat mat2gray(const Mat& src)
//{
//    Mat dst;
//    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
//    return dst;
//}
//
//void detectText3(Mat& src){
//	int w = 11;
//	Mat grey;
//	src.convertTo(grey, CV_32F);
//
//	Mat mu0;
//	blur(grey,mu0, Size(w,w));
//
//	Mat mu1;
//	blur(grey,mu1, Size(w,w));
//
//	Mat mu2;
//	blur(mu0,mu2, Size(2*w+1,2*w+1));
//
//	Mat mu3;
//	blur(mu0.mul(mu0),mu3, Size(2*w+1,2*w+1));
//
//	Mat sigma;
//	cv::sqrt(mu3-mu2.mul(mu2),sigma);
//
//	imshow("mean",mat2gray(mu1));
//	imshow("sigma",mat2gray(sigma));
//
//	Mat tv1 = 0.3*sigma+16;
//	//imshow("tv1",mat2gray(tv1));
//	Mat mask;
//	threshold(sigma - tv1, mask,0,255,THRESH_BINARY_INV);
//	//imshow("mask1",mat2gray(mask));
//
//	Mat rem,remb;
//	bitwise_and(sigma,mask,rem);
//	threshold(rem, remb,0,255,THRESH_BINARY);
//
//	double fz = 0.0, fm = 0.0;
//	double avg1=cv::mean(rem).val[0],avg2=cv::mean(remb).val[0];
//
//	double noise=avg1*255/avg2;
//	cout<<avg1<<" "<<avg2<<" "<<noise<<endl;
//
//	Mat tv2 = 0.3*sigma+noise;
//	Mat mask2;
//
//	threshold(sigma - tv2, mask2,0,255,THRESH_BINARY);
//
//	imshow("tv2",mat2gray(tv2));
//	imshow("mask2",mat2gray(mask2));
//
//
//	Mat rmv;
//	bitwise_and(mu1,mask2,rmv);
//
//	imshow("rmv",mat2gray(rmv));
//
//	Mat rem2 = mu1 - rmv;
//	imshow("rem",mat2gray(rem2));
///**/
//	Mat proc0;
//	threshold(rem2, proc0,240,255,THRESH_BINARY);
//	imshow("proc0",mat2gray(proc0));
//
//	Mat proc1;
//	threshold(rem2, proc1,10,255,THRESH_BINARY_INV);
//	imshow("proc1",mat2gray(proc1));
//
//	Mat proc2;
//	bitwise_or(proc0,proc1,proc2);
//	imshow("proc2",mat2gray(proc2));
//	waitKey();
//}
//
//void myNormalSize(Mat& src, Mat& tsrc, int type){
//
//	double bili = src.cols>src.rows?(src.cols>1000?1000.0/src.cols:1):(src.rows>1000?1000.0/src.rows:1);
//	Size sz = Size(src.cols*bili,src.rows*bili);
//	tsrc = Mat(sz,type);
//	cv::resize(src, tsrc, sz);
//}
//

//int main_textArea(){

//	/*
//	DIR              *pDir ;
//	struct dirent    *ent  ;
//	int               i=0  ;
//	char              childpath[512];
//
//	pDir=opendir("/home/litton/test_set_1");
//
//	while((ent=readdir(pDir))!=NULL)  {
//
//		if(ent->d_name[0]!='.'){
//			cout<<ent->d_name<<endl;
//			string fname(ent->d_name);
//			cv::Mat src = cv::imread("/home/litton/test_set_1/"+fname);
//			Mat res;
//			if(!src.empty()){
//
//				detectText1(src,res);
//
//				imwrite("/home/litton/test_text_region/"+fname+".jpg",res);
//			}
//		}
//	}
//	*/
//	/**/
//	cv::Mat src = cv::imread("/home/litton/upload/nai.jpg");//("/home/litton/opencv_contrib/opencv_contrib-master/modules/text/samples/scenetext02.jpg");
//	Mat tsrc;
//	if (src.empty())
//		return -1;
//	myNormalSize(src,tsrc,CV_32S);
//	Mat rst;
//	detectText2(tsrc);
//	//detectText3(src);
//
//}

//

