#include    <iostream>
#include    <cmath>
#include    <opencv2/opencv.hpp>
#include    <Eigen/Eigen>
#include    <stdio.h>
#include    "opencv2/nonfree/features2d.hpp"
#include    "opencv2/features2d/features2d.hpp"
#include    "opencv2/calib3d/calib3d.hpp"
using namespace     std;
using namespace     Eigen;
using namespace     cv;


int main(int argc, char **argv)
{
    vector<Mat>  srcImage;

    Mat    img = imread("../dataset/DJI_0010.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/DJI_0011.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img  = imread("../dataset/DJI_0012.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    cv::FileStorage  cameraDetail("../config/inspire.yml", FileStorage::READ);

    Mat   cameraMatrix, cameraUndistortion;
    Eigen::Matrix3d   cameraMatrixEigen;
    cameraDetail["camera_matrix"] >> cameraMatrix;
    cameraDetail["distortion_coefficients"] >> cameraUndistortion;
    for (uint32_t i = 0; i < 3; i++)
    {
        for (uint32_t j = 0; j < 3; j++)
            cameraMatrixEigen(i, j) = cameraMatrix.at<double>(i, j);
    }
    cout << "cameraMatrix\n" << cameraMatrixEigen << endl << endl << cameraMatrixEigen.inverse() << endl;
    //return 0;
    vector<Mat>  imageUndistorted;
    for (uint32_t i = 0; i < srcImage.size(); i++)
    {
        Mat   img;
        cv::undistort(srcImage[i], img, cameraMatrix, cameraUndistortion);
        imageUndistorted.push_back(img);
    }

    cv::SiftFeatureDetector detector;
    //cv::FastFeatureDetector fastDetector(40);
    cv::SiftDescriptorExtractor siftDescriptor;
    vector<vector<KeyPoint> >  keyPointAll;
    vector<Mat> descAll;
    for (uint32_t i = 0; i < imageUndistorted.size(); i++)
    {
        vector<KeyPoint>  keyPoint;
        Mat  descImg;
        detector.detect(imageUndistorted[i], keyPoint);
        siftDescriptor.compute(imageUndistorted[i], keyPoint, descImg);
        keyPointAll.push_back(keyPoint);
        descAll.push_back(descImg);
    }
    cv::BFMatcher matcher;
    vector<DMatch>  match, good_match;
    vector<vector<DMatch> >     matchAllImg;
    vector<vector<Point2f> >    featureGoodMatched;
    Mat  mask, H;
    vector<Mat>  H_imgs;
    for (uint32_t i = 0; i < descAll.size() - 1; i = i + 1) //need to refine for more than two images
    {
        match.clear();
        matcher.match(descAll[i], descAll[i + 1], match);
        vector<Point2f>     a_p, b_p, ap_good, bp_good;
        for (uint32_t j = 0; j < match.size(); j++)
        {
            a_p.push_back(keyPointAll[i][match[j].queryIdx].pt);
            b_p.push_back(keyPointAll[i + 1][match[j].trainIdx].pt);
        }
        H = cv::findHomography(b_p, a_p, CV_FM_RANSAC, 3, mask);
        H_imgs.push_back(H);
        for (uint32_t a = 0; a < match.size(); a++)
        {
            if ((unsigned int)mask.at<uchar>(a))
            {
                good_match.push_back(match[a]);
                ap_good.push_back(a_p[a]);
                bp_good.push_back(b_p[a]);
            }
        }
        matchAllImg.push_back(good_match); //good_match[0] --> featureGoodMatched[0/1]
        featureGoodMatched.push_back(ap_good);
        featureGoodMatched.push_back(bp_good);
    }
    Mat     W, U, VT;
    cv::SVDecomp(H_imgs[0], W, U, VT);
    cout << "W: " << W << endl;
    cout << "U: " << U << endl;
    cout << "VT: " << VT << endl;


    Eigen::Vector3d     uvInPixel, xyzInMetric;
    Eigen::Matrix3d     cameraMatrixInv;
    cameraMatrixInv = cameraMatrixEigen.inverse();
    Eigen::Vector2d  featureInSphere;
    vector<Vector2d> featureGMSphereA, featureGMSphereB;
    for (uint32_t a = 0; a < featureGoodMatched.size(); a = a + 2)
    {
        featureGMSphereA.clear();
        for (uint32_t b = 0; b < featureGoodMatched[a].size(); b++)
        {
            uvInPixel.x() = featureGoodMatched[a][b].x;
            uvInPixel.y() = featureGoodMatched[a][b].y;
            uvInPixel.z() = 1;
            xyzInMetric = cameraMatrixInv * uvInPixel;
            featureInSphere.x() = atan2(xyzInMetric.x(), xyzInMetric.z());
            featureInSphere.y() = asin(xyzInMetric.normalized().y());
            featureGMSphereA.push_back(featureInSphere);
        }
        Eigen::Vector3d    deltaXYZ, thetaXYZ;
        Eigen::Vector2d    errRes, thetaXY, residualSum;
        vector<Vector2d>   errResAll;
        Eigen::Vector2d    xyInMetric;
        Eigen::Matrix2d    Rot;
        Eigen::MatrixXd    Joc(2, 3);
        Eigen::Vector3d    solveVecSum;
        Joc << 1, 0, 0,
            0, 1, 0;
        vector<MatrixXd>   JocAll;
        Eigen::Matrix3d    solveMat, solveMatSum;
        solveMat.setIdentity();

        deltaXYZ.setZero();
        thetaXYZ.setZero();
        thetaXYZ << -1, -0.1, -0.1;
        for (uint32_t iterNum = 0; iterNum < 100; iterNum++)
        {
            featureGMSphereB.clear();
            solveMatSum.setZero();
            solveVecSum.setZero();
            residualSum.setZero();
            thetaXY << thetaXYZ.x(), thetaXYZ.y();
            Rot << cos(thetaXYZ.z()), -sin(thetaXYZ.z()), sin(thetaXYZ.z()), cos(thetaXYZ.z());
            for (uint32_t i = 0; i < featureGoodMatched[a].size(); i++)
            {
                uvInPixel.x() = featureGoodMatched[a + 1][i].x;
                uvInPixel.y() = featureGoodMatched[a + 1][i].y;
                uvInPixel.z() = 1;
                xyzInMetric = cameraMatrixInv * uvInPixel;
                xyInMetric.x() = xyzInMetric.x();
                xyInMetric.y() = xyzInMetric.y();
                xyInMetric.noalias() = Rot * xyInMetric;
                featureInSphere.x() = atan2(xyInMetric.x(), xyzInMetric.z());
                featureInSphere.y() = asin(xyInMetric.normalized().y());
               
                featureGMSphereB.push_back(featureInSphere);

                errRes = featureGMSphereA[i] - featureInSphere + thetaXY;
                residualSum = residualSum + Vector2d(errRes.x() * errRes.x(), errRes.y() * errRes.y());

                Joc(0, 2) = (xyzInMetric.x() * sin(thetaXYZ.z()) + xyzInMetric.y() * cos(thetaXYZ.z())) /
                            (1 + pow((xyzInMetric.x() * cos(thetaXYZ.z()) - xyzInMetric.y() * sin(thetaXYZ.z())), 2));
                Joc(1, 2) = (xyzInMetric.y() * sin(thetaXYZ.z()) - xyzInMetric.x() * cos(thetaXYZ.z())) /
                            (1 + pow((xyzInMetric.y() * cos(thetaXYZ.z()) + xyzInMetric.x() * sin(thetaXYZ.z())), 2));
                solveMat(0, 2) = Joc(0, 2);
                solveMat(1, 2) = Joc(1, 2);
                solveMat(2, 2) = pow(Joc(0, 2), 2) + pow(Joc(1, 2), 2);
                solveMatSum.noalias()   = solveMatSum + solveMat;
                solveVecSum.noalias()   = solveVecSum + (errRes.transpose() * Joc).transpose();
            }
            deltaXYZ = solveMatSum.inverse() * (-solveVecSum);
            thetaXYZ.noalias() = thetaXYZ + deltaXYZ;
            cout << thetaXYZ.x() << " " << thetaXYZ.y() << " " << thetaXYZ.z() << "\t" << residualSum.x() << " " << residualSum.y() << endl;
        }
        cout << thetaXYZ.x() << " " << thetaXYZ.y() << " " << thetaXYZ.z() << endl;// "\t"
        //fuse the Image together
        Eigen::Vector3d  uvPixel, xyzMetric;
        Vector2d xyMetric, xyInSphere, uvAfterBlend, u0v0k0AfterBlend;
        double  FOCUS = 1000;
        Mat     imageFused;
        imageFused = Mat::zeros(2000, 2000, imageUndistorted[0].type());
        Matrix2d   Rota;
        Rota << cos(thetaXYZ.z()), -sin(thetaXYZ.z()), sin(thetaXYZ.z()), cos(thetaXYZ.z());
        Vector2d   optiXY;
        optiXY << thetaXYZ.x(), thetaXYZ.y();
        for (uint32_t k = 0; k < imageUndistorted.size(); k++)
        {
            int  i = 0;
            if (k == 1)
                i = imageUndistorted[k].size().width / 2;
            for ( ; i < imageUndistorted[k].size().width; i++)
            {
                for (int j = 0; j < imageUndistorted[k].size().height; j++)
                {
                    uvPixel.x() = i;
                    uvPixel.y() = j;
                    uvPixel.z() = 1;
                    xyzMetric  =   cameraMatrixInv * uvPixel;
                    xyMetric.x() = xyzMetric.x();
                    xyMetric.y() = xyzMetric.y();
                    if (k == 1)
                        xyMetric  = Rota * xyMetric;
                    xyInSphere.x() = atan2(xyMetric.x(), xyzMetric.z());
                    xyInSphere.y() = atan2(xyMetric.y(), xyzMetric.z());
                    if (k == 1)
                        xyInSphere =  -optiXY + xyInSphere;

                    uvAfterBlend.x() = xyInSphere.x() * FOCUS;
                    uvAfterBlend.y() = tan(xyInSphere.y()) * FOCUS;
                    if (k == 0 && i == 0 && j == 0)
                        u0v0k0AfterBlend = uvAfterBlend;
                    uvAfterBlend.noalias() = uvAfterBlend - u0v0k0AfterBlend;
                    if (uvAfterBlend.y() > 0 && uvAfterBlend.x() > 0 &&
                            (imageUndistorted[k].at<Vec3b>(j, i)[0] > 10 &&
                             imageUndistorted[k].at<Vec3b>(j, i)[1] > 10 &&
                             imageUndistorted[k].at<Vec3b>(j, i)[2] > 10))
                        for (int g = 0; g < 3; g++)
                        {
                            imageFused.at<Vec3b>((int)uvAfterBlend.y(), uvAfterBlend.x())[g] = imageUndistorted[k].at<cv::Vec3b>(j, i)[g];
                        }
                }
            }
            cout << k << endl;
        }
        imshow("Fused Image", imageFused);
        waitKey(0);
    }

    waitKey(0);
    return  1;
}


