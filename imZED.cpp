//============================================================================
// Name        : imZED.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <iomanip>
#include <stddef.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


//#include <cuda.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>


#include <sstream>

#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>

//New includes



using namespace std;
using namespace cv;

void lineIntersection(double x1, double y1, double x2, double y2,
					  double x3, double y3, double x4, double y4,
					  double& px, double&);

void undistortImagePointTordoff(
	Point2f &undistortedImgPoint,
	Point2f &distortedImgPoint,
	Point2f principlePoint,
	double radialDistortionCoeff);

void planextraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr& result){
	//create cloud data
	  pcl::PointIndices::Ptr inliers_ransac_b (new pcl::PointIndices);
	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

	  pcl::SACSegmentation<pcl::PointXYZ> seg;
	  // Optional

	  seg.setOptimizeCoefficients (true);
	  // Mandatory
	  seg.setModelType (pcl::SACMODEL_PLANE);
	  seg.setMethodType (pcl::SAC_RANSAC);
	  seg.setMaxIterations(200);
	  seg.setDistanceThreshold (5);//0.06 //3.58

	  // Create the filtering object
	  pcl::ExtractIndices<pcl::PointXYZ> extract;

	  seg.setInputCloud (cloud);
	  seg.segment (*inliers_ransac_b, *coefficients);

	  extract.setInputCloud (cloud);
	  extract.setIndices (inliers_ransac_b);
	  extract.setNegative (false);
	  extract.filter (*result);

	  extract.setNegative (true);
	  extract.filter (*cloud_f);
	  cloud.swap (cloud_f);
}



void compute_orb(cuda::GpuMat img1, cuda::GpuMat img2, vector<KeyPoint>& kpts_1, vector<KeyPoint>& kpts_2, vector<DMatch>& good_matches){
	//std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::cuda::GpuMat keypoints1, keypoints2;
	cv::cuda::GpuMat descriptors1, descriptors2;
	cv::Ptr<cuda::ORB> d_orb = cuda::ORB::create(1500,1.2,8,2,2,2,ORB::FAST_SCORE,31,5);

	//good_matches.clear();

	d_orb->detectAndComputeAsync(img1, cv::cuda::GpuMat(), keypoints1, descriptors1);
	d_orb->detectAndComputeAsync(img2, cv::cuda::GpuMat(), keypoints2, descriptors2);

	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	//cv::BFMatcher matcher(NORM_HAMMING);
	std::vector<std::vector<DMatch>> matches;
	matcher->knnMatch(descriptors1, descriptors2, matches, 2, cv::cuda::GpuMat());


	for(int k=0; k < min(descriptors1.rows-1, (int)matches.size()); k++){
		if((matches[k][0].distance<0.5*(matches[k][1].distance)) && ((int)matches[k].size()<=2 &&(int)matches[k].size()>0)){
			good_matches.push_back(matches[k][0]);  ;
		}
	}
	d_orb->convert(keypoints1, kpts_1);
	d_orb->convert(keypoints2, kpts_2);

	matches.clear();

}

void compute_depth(Mat mCMlInv,Mat mCMrInv,vector<Point2f> undistortL,vector<Point2f> undistortR,
				   vector<Point3f>& stereoPoints,vector<Point2f> distortL,pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
	{
	//************ calibration files*************
	Mat mF,mR,mT;
	Mat mRwc,mTwc;
	FileStorage cf("camera_config/stereo.yaml", FileStorage::READ);
    ////printjmcv(cf.isOpened());
	cf["F"]>> mF;
	cf["R"]>> mR;
	cf["T"]>> mT;
	cf.release();


	mRwc = mR.inv();
	mTwc = -mRwc*mT;
	//*******************************************

	 // Fill in the cloud data
  cloud->width    = undistortL.size();
  cloud->height   = 1;
  cloud->is_dense = true;
  cloud->points.resize (cloud->width * cloud->height);
  int mind=50;

	for (int i=0; i< undistortL.size(); i++){

				Point3d u(undistortL[i].x, undistortL[i].y, 1.0);

				Mat_<double> um = mCMlInv * Mat_<double>(u);
				u.x = um(0,0);
				u.y = um(1,0);
				u.z = um(2,0);

				Point3d v(undistortR[i].x, undistortR[i].y, 1.0);

				Mat_<double> vm = mCMrInv * Mat_<double>(v);
				v.x = vm(0,0);
				v.y = vm(1,0);
				v.z = vm(2,0);

				//printjmcv(vm);
				vm = mRwc * Mat_<double>(v) + mTwc;
				v.x = vm(0,0);
				v.y = vm(1,0);
				v.z = vm(2,0);
				//printjmcv(vm);


				double x1 = 0;
				double y1 = 0;
				double x2 = u.x;
				double y2 = u.z;

				double x3 = mTwc.at<double>(0);
				double y3 = mTwc.at<double>(2);
				double x4 = v.x;
				double y4 = v.z;

				double px,py;

		lineIntersection(x1,y1,x2,y2,
						 x3,y3,x4,y4,
						 px,py);
        double depth=py;
		Point3f pointsWC;

		pointsWC.x=distortL[i].x;
		pointsWC.y=distortL[i].y;
        pointsWC.z=depth;
        stereoPoints.push_back(pointsWC);

        cloud->points[i].x = distortL[i].x;//((distortL[i].x/752)*2)-1;
        cloud->points[i].y = distortL[i].y;//((distortL[i].y /480)*-2)+1;
        cloud->points[i].z = depth;//(((depth-mind)/dis)*-2)+1;

        //cout <<"X: " << cloud->points[i].x << "X: " << cloud->points[i].y << "Z: " << cloud->points[i].z <<endl;

        }

}

void depth_undistort(vector<KeyPoint> kpts_1,vector<KeyPoint> kpts_2,vector<DMatch> good_matches, vector<Point2f>& undistortL,vector<Point2f>& undistortR,
					 Mat& mCMlInv, Mat& mCMrInv,vector<Point2f>& distortL, vector<Point2f>& distortR, Mat& mcMl,Mat& mcMr, Mat& mcDl, Mat& mcDr){
	//***************************** read calibration

	//undistortL.clear();
	//undistortR.clear();


	FileStorage cl("camera_config/ZED_L.yaml", FileStorage::READ);

	cl["CM"]>> mcMl;
	cl["D"]>> mcDl;
	//std::cout<<"matrix K: "<<mcMl<< std::endl;
	cl.release();
	mCMlInv=mcMl.inv();
	FileStorage cr("camera_config/ZED_R.yaml", FileStorage::READ);
    //CM es la matriz 3x3
	cr["CM"]>> mcMr;
	//D son los parámetros de distorsión k1, k2, p1, p2, k3
	cr["D"]>> mcDr;
	cr.release();
	mCMrInv=mcMr.inv();
	//*******************************
	Point2f d_kpts_l, d_kpts_r;
	Point2f d_kpts_Ul, d_kpts_Ur;

	Point2f principalPtL, principalPtR;

	double coeffL, coeffR;
	//		[0]		[1]		[2]
	//[0]	fx	-	0	-	PPtx
	//[1]	0	-	fy	-	PPty
	//[2]	0	-	0	-	1

	principalPtL.x = mcMl.at<double>(0,2);
	principalPtL.y = mcMl.at<double>(1,2);

	principalPtR.x = mcMr.at<double>(0,2);
	principalPtR.y = mcMr.at<double>(1,2);


	coeffL = mcDl.at<double>(0,0)/(mcMl.at<double>(0,0)*mcMl.at<double>(0,0));
	coeffR = mcDr.at<double>(0,0)/(mcMr.at<double>(0,0)*mcMr.at<double>(0,0));

	for (int k=0; k < good_matches.size(); k++){

			d_kpts_l.x=kpts_1[good_matches[k].queryIdx].pt.x;
			d_kpts_l.y=kpts_1[good_matches[k].queryIdx].pt.y;

			d_kpts_r.x=kpts_2[good_matches[k].trainIdx].pt.x;
			d_kpts_r.y=kpts_2[good_matches[k].trainIdx].pt.y;

			undistortImagePointTordoff(d_kpts_Ul,d_kpts_l,principalPtL,coeffL);
			undistortL.push_back(d_kpts_Ul);
			distortL.push_back(d_kpts_l);

			undistortImagePointTordoff(d_kpts_Ur,d_kpts_r,principalPtR,coeffR);
			undistortR.push_back(d_kpts_Ur);
			distortR.push_back(d_kpts_r);

	}
}


void out_data(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
	int aux=0;
	int sz=cloud->points.size();
	for(int i=0;i<cloud->points.size();i++){
		aux+=cloud->points[i].z;
		}
	cout<<"depth mean= "<<aux/sz<<endl;

	Mat data_pts = Mat(sz,2,CV_64FC1);

	for (int j=0;j<data_pts.rows;j++){
		data_pts.at<double>(j,0)=cloud->points[j].x;
		data_pts.at<double>(j,1)=cloud->points[j].y;
		}
	PCA pca_analysis(data_pts,Mat(),CV_PCA_DATA_AS_ROW);
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0,0)),
					   static_cast<int>(pca_analysis.mean.at<double>(0,1)));

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);

	for(int k=0;k<2;k++){
		eigen_vecs[k]=Point2d(pca_analysis.eigenvectors.at<double>(k,0),
							  pca_analysis.eigenvectors.at<double>(k,1));

		eigen_val[k]= pca_analysis.eigenvalues.at<double>(0,k);

		}

	cout<<"eigenvectors= "<<eigen_vecs<<endl;
	printf("eigenvalues= %f,%f \n",eigen_val[0],eigen_val[1]);


	eigen_vecs.clear();
	eigen_val.clear();
	}

int main(int argc, char **argv) {

    if (argc > 3) {
        return -1;
    }

    // Quick check input arguments
    bool readSVO = false;
    std::string SVOName;
    bool loadParams = false;
    std::string ParamsName;
    if (argc > 1) {
        std::string _arg;
        for (int i = 1; i < argc; i++) {
            _arg = argv[i];
            if (_arg.find(".svo") != std::string::npos) {
                // If a SVO is given we save its name
                readSVO = true;
                SVOName = _arg;
            }
            if (_arg.find(".ZEDinitParam") != std::string::npos) {
                // If a parameter file is given we save its name
                loadParams = true;
                ParamsName = _arg;
            }
        }
    }

    sl::zed::Camera* zed;

    if (!readSVO) // Live Mode
        zed = new sl::zed::Camera(sl::zed::VGA);
    else // SVO playback mode
        zed = new sl::zed::Camera(SVOName);

    // Define a struct of parameters for the initialization
    sl::zed::InitParams params;

    if (loadParams) // A parameters file was given in argument, we load it
        params.load(ParamsName);

    // Enables verbosity in the console
    params.verbose = true;


    sl::zed::ERRCODE err = zed->init(params);
    if (err != sl::zed::SUCCESS) {
        // Exit if an error occurred
        delete zed;
        return 1;
    }

    // Save the initialization parameters
    // The file can be used later in any zed based application
    params.save("MyParam");

    char key = ' ';


    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;



    cv::Mat ileft(height, width, CV_8UC1);
    cv::Mat iright(height, width, CV_8UC1);
    cv::Mat iconc(376, 2*672, CV_8UC1);

    vector<KeyPoint> kpts_1,kpts_2;
    vector<DMatch> good_matches;

    //Undistorted Points
    vector<Point2f> undistL, undistR;
    //Distorted Points
    vector<Point2f> distL, distR;

    //Inverse Calibration
    Mat mCMlInv,mCMrInv;
    Mat mcMl,mcMr,mcDl,mcDr;

    //undistorted points output
    vector<Point3f> stereoPoints;


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    //planes output
    pcl::PointCloud<pcl::PointXYZ>::Ptr result (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result1 (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::visualization::CloudViewer viewer ("Simple Viewer");

    cv::Size displaySize(2*672, 376);

    sl::zed::SENSING_MODE dm_type = sl::zed::STANDARD;

    cv::namedWindow("VIEW", cv::WINDOW_AUTOSIZE);

    std::cout << "Press 'q' to exit" << std::endl;

    // Jetson only. Execute the calling thread on core 2
    //sl::zed::Camera::sticktoCPUCore(2);


    // Loop until 'q' is pressed
    while (key != 'q') {
        // Disparity Map filtering

        // Get frames and launch the computation
        zed->grab(dm_type,false, false, false);



        slMat2cvMat(zed->retrieveImage(sl::zed::LEFT_UNRECTIFIED_GREY)).copyTo(ileft);
        slMat2cvMat(zed->retrieveImage(sl::zed::RIGHT_UNRECTIFIED_GREY)).copyTo(iright);

        cv::cuda::GpuMat img1(ileft);
        cv::cuda::GpuMat img2(iright);
        cv::cuda::GpuMat img3(iconc);

            //slMat2cvMat(zed->retrieveImage(static_cast<sl::zed::SIDE> (1))).copyTo(iright);

        compute_orb(img1, img2, kpts_1, kpts_2, good_matches);
        //cv::hconcat(ileft,iright, iconc);


            //undistortion
        depth_undistort(kpts_1,kpts_2,good_matches,undistL,undistR,mCMlInv,mCMrInv,distL,distR,mcMl,mcMr,mcDl,mcDr);
        compute_depth(mCMlInv,mCMrInv,undistL,undistR,stereoPoints,distL,cloud); //line intersection method

        cv::drawMatches(ileft, kpts_1, iright, kpts_2, good_matches, iconc);
        good_matches.clear();

        imshow("VIEW", iconc);


        //std::cout << "Cloud: " << cloud->points << std::endl;
        //cv::resize(iconc, iconc, displaySize);

            //plane extraction



        cout<<"plane 1"<<endl;
        if(cloud->points.size()>20){
          	planextraction(cloud,result);
           	out_data(result);
        }else{
           	cout<<"plane can't be detected"<<endl;
        }

        cout << "Puntos: " << cloud->points.size() << endl;

        cout<<"plane 2"<<endl;
        if(cloud->points.size()>50){
          	planextraction(cloud,result1);
           	out_data(result1);
        }else{
           	cout<<"plane can't be detected"<<endl;
        }

        viewer.showCloud(result);
        undistL.clear();
        undistR.clear();
        distR.clear();
        distL.clear();

        key = cv::waitKey(5);
    }


    delete zed;
    return 0;
}

void undistortImagePointTordoff(
	Point2f &undistortedImgPoint,
	Point2f &distortedImgPoint,
	Point2f principlePoint,
	double radialDistortionCoeff){

	//printjmcv(distortedImgPoint);
	Point2f vec = distortedImgPoint - principlePoint;
	double r = cv::norm(vec);
	double invFactor = 1.0 / sqrt(1.0 + 2.0 * radialDistortionCoeff * r * r);
	undistortedImgPoint = principlePoint + vec * invFactor;

}

void lineIntersection(double x1, double y1, double x2, double y2,
					  double x3, double y3, double x4, double y4,
					  double &px, double &py)
{
	double denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);

	if(denom < 0.00001)
	{
		px = 0;
		py = 0;
		return ;
	}

	px = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4);

	px = px/denom;

	py = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4);

	py = py/denom;
}
