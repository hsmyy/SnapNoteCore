//
//  ConnectedComponent.h
//  ConnectedComponent
//
//  Created by Saburo Okita on 06/06/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#ifndef __RobustTextDetection__ConnectedComponent__
#define __RobustTextDetection__ConnectedComponent__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

/**
 * Structure that describes the property of the connected component
 */
struct ComponentProperty {
    int labelID;
    int area;
    float eccentricity;
    float solidity;
    Point2f centroid;
    Rect boundingBox;
    
    friend std::ostream &operator <<( ostream& os, const ComponentProperty & prop ) {
        os << "     Label ID: " << prop.labelID      << "\n";
        os << "         Area: " << prop.area         << "\n";
        os << "     Centroid: " << prop.centroid     << "\n";
        os << " Eccentricity: " << prop.eccentricity << "\n";
        os << "     Solidity: " << prop.solidity     << "\n";
        return os;
    }
};


/**
 * Connected component labeling using 8-connected neighbors, based on
 * http://en.wikipedia.org/wiki/Connected-component_labeling
 *
 * with disjoint union and find functions adapted from :
 * https://courses.cs.washington.edu/courses/cse576/02au/homework/hw3/ConnectComponent.java
 */
class ConnectedComponent {
public:
    ConnectedComponent( int max_component = 1000, int connectivity_type = 8 );
    virtual ~ConnectedComponent();
    
    cv::Mat apply( const cv::Mat& image );
    
    int getComponentsCount();
    const std::vector<ComponentProperty>& getComponentsProperties();
    
    std::vector<int> get8Neighbors( int * curr_ptr, int * prev_ptr, int x );
    std::vector<int> get4Neighbors( int * curr_ptr, int * prev_ptr, int x );
    void debugCC(cv::Mat &labelRegionImg1i, cv::Mat &originImg1f, int regionSize, char * title);
protected:
    float calculateBlobEccentricity( const cv::Moments& moment );
    cv::Point2f calculateBlobCentroid( const cv::Moments& moment );
    
    void disjointUnion( int a, int b, std::vector<int>& parent  );
    int disjointFind( int a, std::vector<int>& parent, std::vector<int>& labels  );

private:
    int connectivityType;
    int maxComponent;
    int nextLabel;
    std::vector<ComponentProperty> properties;
};

ConnectedComponent::ConnectedComponent( int max_component, int connectivity_type )
: maxComponent( max_component ),
connectivityType( connectivity_type ){
	nextLabel = 0;
}

ConnectedComponent::~ConnectedComponent(){
}

bool componentCompare(const ComponentProperty& a, const ComponentProperty& b){
        return a.area > b.area;
    }

/**
 * Apply connected component labeling
 * it only works for predefined maximum no of connected components
 * and currently treat black color as background
 */
Mat ConnectedComponent::apply( const Mat& image ) {
    CV_Assert( !image.empty() );
    CV_Assert( image.channels() == 1 );

    /* Padding the image with 1 pixel border, just to remove boundary checks */
    Mat result( image.rows + 2, image.cols + 2, image.type(), Scalar(0) );
    image.copyTo( Mat( result, Rect(1, 1, image.cols, image.rows) ) );
    result.convertTo( result, CV_32SC1 );

    /* 1st pass: labeling the regions incrementally */
    nextLabel = 1;
    vector<int> linked(maxComponent);


    /* Function pointer, it makes everything hard to read... */
    /* Basically use function pointer to decide whether to use 4 or 8 neighbors connectivity */
//    function<vector<int>, int*, int*, int> func_ptr;
//    if( connectivityType == 8 )
//        func_ptr = std::bind(&ConnectedComponent::get8Neighbors, this, placeholders::_1, placeholders::_2, placeholders::_3);
//    else
//        func_ptr = std::bind(&ConnectedComponent::get4Neighbors, this, placeholders::_1, placeholders::_2, placeholders::_3);


    /* Preparing the pointers */
    int * prev_ptr = result.ptr<int>(0);
    int * curr_ptr = result.ptr<int>(1);

    for( int y = 1; y < result.rows - 1; y++ ) {
        int * next_ptr = result.ptr<int>(y + 1);

        for( int x = 1; x < result.cols - 1; x++ ) {

            if( curr_ptr[x] != 0 ) {
                vector<int> neighbors;
                if(connectivityType == 8){
                	neighbors = get8Neighbors( curr_ptr, prev_ptr, x );
                }else{
                	neighbors = get4Neighbors( curr_ptr, prev_ptr, x );
                }

                if( neighbors.empty() ) {
                    if( connectivityType == 8 && curr_ptr[x+1] == 0 && next_ptr[x] == 0 && next_ptr[x-1] == 0 && next_ptr[x+1] == 0 ) {
                        /* If it's single isolated pixel, why even bother */
                        curr_ptr[x] = 0;
                    }
                    else if( connectivityType == 4 && curr_ptr[x+1] == 0 && next_ptr[x] == 0 ) {
                        /* If it's single isolated pixel, why even bother */
                        curr_ptr[x] = 0;
                    }
                    else {
                        /* If it's new unconnected blob */
                        curr_ptr[x] = nextLabel;
                        nextLabel++;

                        if( nextLabel >= maxComponent ) {
                            stringstream ss;
                            ss  << "Current label count [" << (int) nextLabel
                            << "] exceeds maximum no of components [" << maxComponent << "]";
                            //throw std::runtime_error( ss.str() );
                            exit(-1);
                        }
                    }
                }
                else {
                    /* Use the minimum label out from the neighbors */
                    int min_index = (int) (min_element( neighbors.begin(), neighbors.end() ) - neighbors.begin());
                    curr_ptr[x]   = neighbors[min_index];

                    for( unsigned int i = 0, len = neighbors.size(); i < len; ++i ){
                    	int neighbor = neighbors[i];
                        disjointUnion( curr_ptr[x], neighbor, linked );
                    }
                }
            }
        }

        /* Shift the pointers */
        prev_ptr = curr_ptr;
        curr_ptr = next_ptr;
    }


    /* Remove our padding borders */
    result = Mat( result, Rect(1, 1, image.cols, image.rows) );

    /* 2nd pass: merge the equivalent labels */
    nextLabel = 1;
    vector<int> temp, labels_set(maxComponent);
    for( int y = 0; y < result.rows; y++ ) {
        int * curr_ptr = result.ptr<int>(y);

        for( int x = 0; x < result.cols; x++ ) {
            if( curr_ptr[x] != 0 ) {
                curr_ptr[x] = disjointFind( curr_ptr[x], linked, labels_set );
                temp.push_back( curr_ptr[x] );
            }
        }
    }


    /* Get the unique labels */
    vector<int> labels;
    if( !temp.empty() ) {
        std::sort( temp.begin(), temp.end() );
        std::unique_copy( temp.begin(), temp.end(), std::back_inserter( labels ) );
    }

    /* Gather the properties of each blob */
    properties.resize( labels.size() );
    for( int i = 0; i < labels.size(); i++ ) {
        Mat blob        = result == labels[i];

        Moments moment  = cv::moments( blob );

        properties[i].labelID   = labels[i];
        properties[i].area      = countNonZero( blob );

        properties[i].eccentricity = calculateBlobEccentricity( moment );
        properties[i].centroid     = calculateBlobCentroid( moment );

        /* Find the solidity of the blob from blob area / convex area */
        vector<vector<Point> > contours;
        findContours( blob, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );

        if( !contours.empty() ) {
            vector<vector<Point> > hull(1);
            convexHull( contours[0], hull[0] );

            /* ... I hope this is correct ... */
            properties[i].solidity = properties[i].area / contourArea( hull[0] );

            //calculate range;
            if(hull[0].size() > 0){
            	properties[i].boundingBox.x = hull[0][0].x;
            	properties[i].boundingBox.y = hull[0][0].y;
            	properties[i].boundingBox.width = hull[0][0].x;
            	properties[i].boundingBox.height = hull[0][0].y;
            	for(unsigned int j = 0, len = hull[0].size(); j < len;++j){
            		if(properties[i].boundingBox.x > hull[0][j].x){
            			properties[i].boundingBox.x = hull[0][j].x;
            		}else if(properties[i].boundingBox.width < hull[0][j].x){
            			properties[i].boundingBox.width = hull[0][j].x;
            		}
            		if(properties[i].boundingBox.y > hull[0][j].y){
            			properties[i].boundingBox.y = hull[0][j].y;
            		}else if(properties[i].boundingBox.height < hull[0][j].y){
            			properties[i].boundingBox.height = hull[0][j].y;
            		}
            	}
            	properties[i].boundingBox.width -= properties[i].boundingBox.x;
            	properties[i].boundingBox.height -= properties[i].boundingBox.y;
            }
        }
    }


    /* By default, sort the properties from the area size in descending order */
    sort( properties.begin(), properties.end(), componentCompare);


    return result;
}

void ConnectedComponent::debugCC(cv::Mat &labelRegionImg1i, cv::Mat &originImg1f, int regionSize, char * title){
	//debug
	Mat debug1st = Mat(originImg1f.size(), CV_8UC3, Scalar(0));
	vector<Vec3b> randomColor(regionSize);
	for(unsigned int i = 0, len = randomColor.size(); i < len; ++i){
		randomColor[i][0] = rand() % 255;
		randomColor[i][1] = rand() % 255;
		randomColor[i][2] = rand() % 255;
	}
	for(int y = 0; y < debug1st.rows; ++y){
		for(int x = 0; x < debug1st.cols; ++x){
			if(originImg1f.at<int>(y,x) > 0){
				debug1st.at<Vec3b>(y,x)[0] = randomColor[labelRegionImg1i.at<int>(y,x) - 1][0];
				debug1st.at<Vec3b>(y,x)[1] = randomColor[labelRegionImg1i.at<int>(y,x) - 1][1];
				debug1st.at<Vec3b>(y,x)[2] = randomColor[labelRegionImg1i.at<int>(y,x) - 1][2];
			}
		}
	}
	namedWindow(title);
	imshow(title, debug1st);
}

/**
 * From the given blob's moments, calculate its eccentricity
 * It's implemented based on the formula shown on http://en.wikipedia.org/wiki/Image_moment#Examples_2
 * which includes using the blob's central moments to find the eigenvalues
 */
float ConnectedComponent::calculateBlobEccentricity( const Moments& moment ) {
    double left_comp  = (moment.nu20 + moment.nu02) / 2.0;
    double right_comp = sqrt( (4 * moment.nu11 * moment.nu11) + (moment.nu20 - moment.nu02)*(moment.nu20 - moment.nu02) ) / 2.0;

    double eig_val_1 = left_comp + right_comp;
    double eig_val_2 = left_comp - right_comp;

    return sqrtf( 1.0 - (eig_val_2 / eig_val_1) );
}

/**
 * From the given blob moment, calculate its centroid
 */
Point2f ConnectedComponent::calculateBlobCentroid( const Moments& moment ) {
    return Point2f( moment.m10 / moment.m00, moment.m01 / moment.m00 );
}

/**
 * Returns the number of connected components found
 */
int ConnectedComponent::getComponentsCount() {
    return static_cast<int>( properties.size() );
}

const vector<ComponentProperty>& ConnectedComponent::getComponentsProperties() {
    return properties;
}

/**
 * Disjoint set union function, taken from
 * https://courses.cs.washington.edu/courses/cse576/02au/homework/hw3/ConnectComponent.java
 */
void ConnectedComponent::disjointUnion( int a, int b, vector<int>& parent  ) {
    while( parent[a] > 0 )
        a = parent[a];
    while( parent[b] > 0 )
        b = parent[b];

    if( a != b ) {
        if( a < b )
            parent[a] = b;
        else
            parent[b] = a;
    }
}

/**
 * Disjoint set find function, taken from
 * https://courses.cs.washington.edu/courses/cse576/02au/homework/hw3/ConnectComponent.java
 */
int ConnectedComponent::disjointFind( int a, vector<int>& parent, vector<int>& labels ) {
    while( parent[a] > 0 )
        a = parent[a];
    if( labels[a] == 0 )
        labels[a] = nextLabel++;
    return labels[a];
}

/**
 * Get the labels of 8 point neighbors from the given pixel
 *   | 2 | 3 | 4 |
 *   | 1 | 0 | 5 |
 *   | 8 | 7 | 6 |
 *
 * returns a vector of that contains unique neighbor labels
 */
vector<int> ConnectedComponent::get8Neighbors( int * curr_ptr, int * prev_ptr, int x ) {
    vector<int> neighbors;

    /* Actually we only consider pixel 1, 2, 3, and 4 */
    /* At this point of time, the logic hasn't traversed thru 5, 6, 7, 8 */
    if( prev_ptr[x-1] != 0 )
        neighbors.push_back( prev_ptr[x-1] );

    if( prev_ptr[x] != 0 )
        neighbors.push_back( prev_ptr[x] );

    if( prev_ptr[x+1] != 0 )
        neighbors.push_back( prev_ptr[x+1] );

    if( curr_ptr[x-1] != 0 )
        neighbors.push_back( curr_ptr[x-1] );

    /* Reduce to unique labels */
    /* This is because I'm not using set (it doesn't have random element access) */
    vector<int> result;
    if( !neighbors.empty() ) {
        std::sort( neighbors.begin(), neighbors.end() );
        std::unique_copy( neighbors.begin(), neighbors.end(), std::back_inserter( result ) );
    }

    return result;
}

/**
 * Similar to the 8 neighbors, but now only considering two pixels (the top and left ones)
 */
vector<int> ConnectedComponent::get4Neighbors( int * curr_ptr, int * prev_ptr, int x ) {
    vector<int> neighbors;

    /* Actually we only consider pixel 1, 2, 3, and 4 */
    /* At this point of time, the logic hasn't traversed thru 5, 6, 7, 8 */
    if( prev_ptr[x] != 0 )
        neighbors.push_back( prev_ptr[x] );

    if( curr_ptr[x-1] != 0 )
        neighbors.push_back( curr_ptr[x-1] );

    /* Reduce to unique labels */
    /* This is because I'm not using set (it doesn't have random element access) */
    vector<int> result;
    if( !neighbors.empty() ) {
        std::sort( neighbors.begin(), neighbors.end() );
        std::unique_copy( neighbors.begin(), neighbors.end(), std::back_inserter( result ) );
    }

    return result;
}

#endif /* defined(__RobustTextDetection__ConnectedComponent__) */
