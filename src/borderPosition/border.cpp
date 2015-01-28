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

using namespace std;

int main2()
{
	/*batch test*/
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
//			Mat cross, turned;
//			if(!src.empty()){
//				for(int i=0;i<1;i++){
//					center.x = 0.0;
//					center.y = 0.0;
//					int ret = mainProc(src,i,cross,turned);
//
//					if(ret!=-1){
//
//					    string is;
//					    strstream ss;
//					    ss << i;
//					    ss >> is;
//
//						imwrite("/home/litton/test_result_2/cross_"+fname+"_"+is+".jpg",cross);
//						cout<<"..."<<endl;
//					}
//				}
//			}
//		}
//	}

	
	/*
	cv::Mat src = cv::imread("/home/litton/test_set_1/note5.jpg");

	//cv::Mat src = cv::imread("/home/litton/imagec.jpg");
	Mat cross, turned;
	if (src.empty())
		return -1;

	for(int i=0;i<1;i++){
		int ret = mainProc(src,i,cross,turned);
		if(ret!=-1){
			imwrite("/home/litton/cross.jpg",cross);
			imwrite("/home/litton/turned.jpg",turned);
			return ret;
		}
	}
*/
	return 0;
}
