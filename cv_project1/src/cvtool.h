#ifndef CVTOOL_H
#define CVTOOL_H
#include    "opencv2/nonfree/features2d.hpp"
#include    "opencv2/features2d/features2d.hpp"
#include    "opencv2/highgui/highgui.hpp"
#include    "opencv2/imgproc/imgproc.hpp"
#include    "opencv2/core/core.hpp"
#include    "opencv2/calib3d/calib3d.hpp"
#include    <iostream>
#include    <stdio.h>
#include    <stdlib.h>
#include    <cmath>
/**
 * @brief The CVTool class
 * This class contains all the functions you should implement in the project.
 * The member functions are initially defined, but you should define the parameters and return value yourself.
 * The name of the public member functions cannot be modified.
 * You can add any protected functions or members.
 */
using namespace cv;
using namespace std;
class CVTool
{
public:
    //CVTool();
    /**
     * @brief for function 1.
     */
    void detectFeatureSIFT(Mat a_img,vector<KeyPoint> &a_keypoint, Mat &adescp);

    void detectFeatureSURF(Mat a_img,vector<KeyPoint> &a_keypoint, Mat &adescp);

    void detectFeatureFAST(Mat a_img,vector<KeyPoint> &a_keypoint, Mat &adescp);

    void detectFeatureMSER(Mat a_img);

    Mat detectFeatureHaris(Mat a_img);

    /**
     * @brief for function 2.
     */
    void matchFeatures(Mat adescp, Mat  bdescp, vector<DMatch> &matches, vector<KeyPoint> a_kp, vector<KeyPoint> b_kp, vector<DMatch> &good_matches); 

    void visualizeMatching(Mat a_img,  Mat b_img, vector<KeyPoint> a_kp, vector<KeyPoint> b_kp, vector<DMatch> good_matches, Mat &img_matches);


    /**
     * @brief repairImage
     * @param damaged_img_a
     * @param complete_img_b
     * @return repaired_img_a
     */
    cv::Mat repairImage( CVTool cvtool, const cv::Mat & damaged_img, const cv::Mat & complete_img);


    /**
     * @brief visualizeDiffence
     * @param repaired_img_a
     * @param complete_img_a
     * compute the difference between the repaired image and the complete image
     */
    void visualizeDiffence(const cv::Mat & repaired_img_a, const cv::Mat & complete_img_a);

    /**
     * @brief for function 4.
     */
    void computeFundMatrix(CVTool cvtool, Mat a_img, Mat b_img, Mat &F, vector<Point2f> &a_pt, vector<Point2f> &b_pt, vector<DMatch> &best_matches);

    Mat visualizeEpipolarLine(CVTool cvtool,Mat &a_img, Mat &b_img, vector<Point2f> a_p, vector<Point2f> b_p, vector<DMatch> match, Mat F, uint32_t NumPlotLines);


    Mat cvShowTwoImage(Mat a_img, Mat b_img);
protected:

    /**
      * You can add any protected function to help.
      */
protected:
    /**
     * @brief image1_, image2_ are the two input images for testing
     */
    cv::Mat image1_, image2_;

    /**
      * You can add any other members to help.
      */
};

#endif 
