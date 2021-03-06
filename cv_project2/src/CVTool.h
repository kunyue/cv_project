#include    <iostream>
#include    <fstream>
#include    <sstream>
#include    <cmath>
#include    <opencv2/opencv.hpp>
#include    <Eigen/Eigen>
#include    <stdio.h>
#include    "opencv2/nonfree/features2d.hpp"
#include    "opencv2/features2d/features2d.hpp"
#include    "opencv2/calib3d/calib3d.hpp"
#include    "pose_.h"
using namespace     std;
using namespace     Eigen;
using namespace     cv;

void UndistortImages(vector<Mat> srcImage, FileStorage cameraDetail, Eigen::Matrix3d &cameraMatrixEigen, vector<Mat> &imageUndistorted)
{
    Mat   cameraMatrix, cameraUndistortion;
    cameraDetail["camera_matrix"] >> cameraMatrix;
    cameraDetail["distortion_coefficients"] >> cameraUndistortion;
    for (uint32_t i = 0; i < 3; i++)
    {
        for (uint32_t j = 0; j < 3; j++)
            cameraMatrixEigen(i, j) = cameraMatrix.at<double>(i, j);
    }
    for (uint32_t i = 0; i < srcImage.size(); i++)
    {
        Mat   img;
        cv::undistort(srcImage[i], img, cameraMatrix, cameraUndistortion);
        imageUndistorted.push_back(img);
    }

}
void GetMatchedPointUsingImage(vector<Mat> imageUndistorted, vector<vector<Point2f> > &featureGoodMatched, vector<Mat> &H_imgs)
{
    cv::SiftFeatureDetector detector;
    cv::FastFeatureDetector fast(60);
    cv::SiftDescriptorExtractor siftDescriptor;
    vector<vector<KeyPoint> >  keyPointAll;
    vector<Mat> descAll;
    for (uint32_t i = 0; i < imageUndistorted.size(); i++)
    {
        vector<KeyPoint>  keyPoint;
        Mat  descImg;
        //detector.detect(imageUndistorted[i], keyPoint);
        fast.detect(imageUndistorted[i], keyPoint);
        siftDescriptor.compute(imageUndistorted[i], keyPoint, descImg);
        keyPointAll.push_back(keyPoint);
        descAll.push_back(descImg);
        cout << "the number of detected points " << i << " : " << keyPoint.size() << endl;
    }
    cv::BFMatcher matcher;
    vector<DMatch>  match, good_match;
    vector<vector<DMatch> >     matchAllImg;
    Mat  mask, H;
    std::ofstream  ofile, oofile;
    ofile.open("../dataset/matched_point.txt", std::ostream::trunc);
    oofile.open("../dataset/H.txt", ostream::trunc);
    oofile.precision(std::numeric_limits<long double>::digits);
    ofile.precision(std::numeric_limits<float>::digits);
    for (uint32_t i = 0; i < descAll.size() - 1; i = i + 1)
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
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                oofile << H.at<double>(a, b) << "\t";
            }
            oofile << endl;
        }
        H_imgs.push_back(H);
        for (uint32_t a = 0; a < match.size(); a++)
        {
            if ((unsigned int)mask.at<uchar>(a))
            {
                good_match.push_back(match[a]);
                ap_good.push_back(a_p[a]);
                bp_good.push_back(b_p[a]);
                ofile << a_p[a].x << "\t" << a_p[a].y << "\t" << b_p[a].x << "\t" << b_p[a].y << endl;
            }
        }
        ofile << "6666" << endl;
        matchAllImg.push_back(good_match); //good_match[0] --> featureGoodMatched[0/1]
        featureGoodMatched.push_back(ap_good);
        featureGoodMatched.push_back(bp_good);
    }
    ofile.close();
    ofile.clear();
    oofile.close();
    oofile.clear();
}

void getStitchingPoint(vector<Mat> imageUndistorted, vector<vector<Point2f> > featureGoodMatched, vector<int> &stitchPoint)
{
    VectorXd    cnt(10);
    int width = imageUndistorted[0].cols / 10;
    for (uint32_t i = 0; i < featureGoodMatched.size(); i = i + 2)
    {
        cout << "the Number of matched Points " << i << " : " << featureGoodMatched[i].size() << endl;
        cnt.setZero();
        for (uint32_t j = 0; j < featureGoodMatched[i].size(); j++)
        {
            cnt[featureGoodMatched[i][j].x / width] ++;
        }
        int maxCnt = cnt(1);
        int maxIdx = 1;

        for (int a = 2; a < 9; a++)
        {
            if (maxCnt < cnt(a))
            {
                maxCnt = cnt(a);
                maxIdx = a;
            }
        }
        stitchPoint.push_back(maxIdx);
    }
}
void ReadMatchedPointsInfile(vector<Mat> &H_imgs, vector<vector<Point2f> > &featureGoodMatched)
{
    std::ifstream  ifile("../dataset/matched_point.txt");
    std::ifstream  iifile("../dataset/H.txt");
    float   x;
    Point2f   feature;
    vector<Point2f>  featuresInA, featuresInB;
    int cnt = 0;
    while (ifile >> x)
    {
        if (x != 6666)
        {
            cnt ++;
            if (cnt == 1)
                feature.x = x;
            else if (cnt == 2)
            {
                feature.y = x;
                featuresInA.push_back(feature);
            }
            else if (cnt == 3)
                feature.x = x;
            else if (cnt == 4)
            {
                feature.y = x;
                featuresInB.push_back(feature);
                cnt = 0;
            }
        }
        else
        {
            cout << "zero: " << cnt << endl;
            featureGoodMatched.push_back(featuresInA);
            featureGoodMatched.push_back(featuresInB);
            featuresInA.clear();
            featuresInB.clear();
        }
    }
    Mat H(3, 3, DataType<double>::type);
    cnt = 0;
    double y;
    while (iifile >> y)
    {
        H.at<double>(cnt / 3, cnt % 3) = y;
        cnt ++;
        if (cnt == 9)
        {
            H_imgs.push_back(H);
            cnt = 0;
        }
    }

}

void  MatchPointsFromPixelToMetric(vector<vector<Point2f> > featureGoodMatched, vector<vector<Eigen::Vector3d> > &matchedPointsInMetric, Eigen::Matrix3d cameraMatrixEigen)
{
    vector<Eigen::Vector3d>   featuresInA, featuresInB;
    Eigen::Vector3d   featureInPixel, featureInMetric;
    Eigen::Matrix3d   cameraMatrixEigenInv;
    cameraMatrixEigenInv = cameraMatrixEigen.inverse();
    for (uint32_t i = 0; i < featureGoodMatched.size(); i = i + 2)
    {
        for (uint32_t j = 0; j < featureGoodMatched[i].size(); j++)
        {
            featureInPixel.x() = featureGoodMatched[i][j].x;
            featureInPixel.y() = featureGoodMatched[i][j].y;
            featureInPixel.z() = 1;
            featureInMetric = (cameraMatrixEigenInv * featureInPixel).normalized();
            featuresInA.push_back(featureInMetric);

            featureInPixel.x() = featureGoodMatched[i + 1][j].x;
            featureInPixel.y() = featureGoodMatched[i + 1][j].y;
            featureInPixel.z() = 1;
            featureInMetric = (cameraMatrixEigenInv * featureInPixel).normalized();
            featuresInB.push_back(featureInMetric);
        }
        matchedPointsInMetric.push_back(featuresInA);
        matchedPointsInMetric.push_back(featuresInB);
        featuresInA.clear();
        featuresInB.clear();
    }
}

Eigen::Matrix3d SKEW(Vector3d vec)
{
    Matrix3d   Skew;
    Skew.setZero();
    Skew(0, 1) = -vec(2);
    Skew(0, 2) =  vec(1);
    Skew(1, 2) = -vec(0);
    Skew(1, 0) =  vec(2);
    Skew(2, 0) = -vec(1);
    Skew(2, 1) =  vec(0);
    return  Skew;
}
void GuassNewtonGetRotation(vector<Matrix3d> &Rotation, vector<vector<Vector3d> > matchedPointsInMetric)
{
    Eigen::Matrix3d  RotInit;
    Eigen::Vector3d  bSum;
    Eigen::Vector3d  error, errorSum;
    Matrix3d CofSum, Cof, Joc;
    for (uint32_t a = 0; a < matchedPointsInMetric.size(); a = a + 2)
    {
        CofSum.setZero();
        for (uint32_t c = 0; c < matchedPointsInMetric[a].size(); c++)
        {
            CofSum = CofSum + SKEW(matchedPointsInMetric[a + 1][c]).transpose() * SKEW(matchedPointsInMetric[a + 1][c]);
        }
        //cout << CofSum << endl;
        RotInit.setIdentity();
        for (int i = 0; i < 30; i++)
        {
            bSum.setZero();
            errorSum.setZero();
            for (uint32_t b = 0; b < matchedPointsInMetric[a].size(); b++)
            {
                error = matchedPointsInMetric[a][b] - RotInit * matchedPointsInMetric[a + 1][b];
                errorSum = errorSum + error;
                Joc   = RotInit * SKEW(matchedPointsInMetric[a + 1][b]);
                bSum  = bSum - (error.transpose() * Joc).transpose();
            }
            Vector3d  delta;
            delta = CofSum.inverse() * bSum;
            cout << "Error: " << errorSum.x() << " " << errorSum.y() << " " << errorSum.z() << "\t" << delta.x() << " " << delta.y() << " " << delta.z() << endl;
            Matrix3d  deltaR;
            deltaR = Eigen::AngleAxisd(delta.norm(), delta.normalized());
            RotInit = RotInit * deltaR;//* (MatrixXd::Identity(3, 3) + SKEW(delta));
            RotInit = Quaterniond(RotInit).normalized().toRotationMatrix();
        }
        cout << MatrixXd::Identity(3, 3) << endl;
        cout << "RotInit:" << endl << RotInit << endl;
        Eigen::Vector3d rpy;
        rpy = Qbw2RPY(Quaterniond(RotInit));
        cout << "rpy" << rpy << endl;
        Rotation.push_back(RotInit);
    }
}
void ShowCoImage(vector<Mat> imageUndistorted, Matrix3d cameraMatrixEigen, vector<Matrix3d> Rotation, vector<int> stitchPoint, Mat &coImage)
{
    double  FocusLen = 1000;
    double  widthX   = 1000;
    double  width  = imageUndistorted[0].cols;
    double  height = imageUndistorted[0].rows;
    Matrix3d   cameraMatrixEigenInv;
    cameraMatrixEigenInv = cameraMatrixEigen.inverse();
    Eigen::Vector3d  corners[4];
    corners[0] << 0, 0, 1;
    corners[1] << width - 1, 0, 1;
    corners[2] << 0, height - 1, 1;
    corners[3] << width - 1, height - 1, 1;

    Eigen::Vector3d  FirP;
    FirP = cameraMatrixEigenInv * corners[0];
    double orienTheta = atan2(FirP.x(), FirP.z());
    double orienDelta = asin(FirP.normalized().y());
    Eigen::Vector2d offset;
    offset << orienTheta * FocusLen, tan(orienDelta) * FocusLen;

    uint32_t srcNum = 0;
    int u_d = 0;
    cout << offset.y() << endl;
    coImage = Mat::zeros((int)2 * widthX, (int) (1.15 * M_PI * FocusLen), imageUndistorted[0].type());

    cout << " loop " << stitchPoint[0]  << " " << stitchPoint[1] << endl;

    srcNum = 0;
    Eigen::Vector2d   pixel;
    Eigen::Vector2d   pixel_;
    Eigen::Vector3d   pixelInM, pixelInM_Rot;
    Eigen::Vector3d   pixelInCamera;
    bool  srcChangeDueZero = false;
    Eigen::Matrix3d RotAll;
    RotAll = Rotation[0].inverse();
    for (uint m = 1; m < Rotation.size(); m++)
    {
        RotAll = RotAll * Rotation[m];
    }
    Eigen::Quaterniond Q_RotAll;
    Q_RotAll = RotAll;
    Eigen::Vector3d  rpy;
    rpy = Qbw2RPY(Q_RotAll);

    cout << "rpy:::::: " << rpy.z() << endl;

    Eigen::Matrix3d   Rot_, RotLast;
    //double cs  = cos(20 * M_PI / 180);
    //double sn  = sin(20 * M_PI / 180);
    double cs  = cos(rpy.z() / 2);
    double sn  = sin(rpy.z() / 2);
    Rot_ << cs, -sn, 0, sn, cs, 0, 0, 0, 1;
    RotLast = Rot_;
    int  u_, v_;
    double  PI_half     = M_PI * 0.5;
    double  PI_double   = M_PI * 2;
    double  PI_3half    = M_PI * 1.5;
    double  PI_5half    = M_PI * 2.5;
    int     height_4    = height * 0.4;
    int     height_6    = height * 0.6;
    int     width_1     = width  * 0.1;
    int     width_05    = width  * 0.05;
    for (int u = 0; u < coImage.cols; u++)
    {
        for (int v = 0; v < coImage.rows; v++)
        {
            pixel.x() = u;
            pixel.y() = v;//<< u, v;
            pixel_  = pixel + offset;
            pixelInM.y() = pixel_.y() - widthX / 2;
            double  angleX = pixel_.x() / FocusLen;
            if (angleX <= PI_half )
            {
                pixelInM.z() = FocusLen * cos(angleX);
                pixelInM.x() = FocusLen * sin(angleX);
            }
            else if (angleX <= PI_3half)
            {
                pixelInM.z() = - FocusLen * cos(abs(M_PI - angleX));
                pixelInM.x() = FocusLen * sin(M_PI - angleX);
            }
            else if (angleX <= PI_5half)
            {
                pixelInM.z() = FocusLen * cos(abs(PI_double - angleX));
                pixelInM.x() = FocusLen * sin(angleX - PI_double);
            }

            pixelInM = pixelInM.normalized();
            pixelInM_Rot = Rot_ * pixelInM;
            pixelInM_Rot = pixelInM_Rot / pixelInM_Rot.z();

            pixelInCamera  = cameraMatrixEigen * pixelInM_Rot;
            u_ = pixelInCamera.x();
            v_ = pixelInCamera.y();
            //if (u_ < 0)
            //{
            //    pixelInM_Rot = RotLast * pixelInM;
            //    pixelInM_Rot = pixelInM_Rot / pixelInM.z();
            //    pixelInCamera  = cameraMatrixEigen * pixelInM_Rot;
            //    u_ = pixelInCamera.x();
            //    v_ = pixelInCamera.y();
            //    srcNum --;
            //    srcChangeDueZero = true;
            //}
            if (u_ > 0 && u_ < width && v_ > 0 && v_ < height && u >= u_d)
                for (int a = 0; a < 3; a++)
                    coImage.at<Vec3b>(v, u)[a] = imageUndistorted[srcNum].at<Vec3b>(v_, u_)[a];

            if (srcChangeDueZero)
            {
                srcNum ++;
                srcChangeDueZero = false;
            }
            if (srcNum < stitchPoint.size())
                if ((v_ < height_6) && (v_ > height_4) &&
                        (u_ > ((stitchPoint[srcNum]) * width_1 + width_05)))
                {
                    RotLast = Rot_;
                    Rot_ = Rotation[srcNum].inverse() * Rot_;
                    u_d = u;
                    srcNum ++;
                    cout << "srcNum ++" << srcNum << " width:" << u << endl;
                }
        }
    }
}

void SVDHomoMatrixToRotation(vector<Mat> H_imgs, vector<Matrix3d> &Rotation)
{
    Eigen::Matrix3d    H_eigen, Rot, W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, -1;
    for (uint32_t i = 0; i < H_imgs.size(); i++)
    {
        for (int a = 0; a < 9; a++)
        {
            H_eigen(a / 3, a % 3) = H_imgs[i].at<double>(a / 3, a % 3);
        }
        JacobiSVD<MatrixXd> svd(H_eigen, ComputeThinU | ComputeThinV);
        cout << "U" << svd.matrixU() << endl << "V" << svd.matrixV() << endl << "E" << svd.singularValues() << endl;
        Rot = svd.matrixU() * W.inverse() * svd.matrixV().transpose();
        Eigen::Vector3d rpy;
        rpy = Qbw2RPY(Quaterniond(Rot));
        cout << "H rpy" << rpy << endl;
        Rotation.push_back(Rot);
    }
}

void DirectSolveRotation(vector<vector<Vector3d> > matchedPointsInMetric, vector<Matrix3d> &Rotation)
{
    Eigen::Matrix3d W, Rot;
    W << 0, -1, 0, 1, 0, 0, 0, 0, -1;
    for (uint32_t i = 0; i < matchedPointsInMetric.size(); i = i + 2)
    {
        Eigen::MatrixXd  Cof(3 * matchedPointsInMetric[i].size(), 9);
        Cof = Eigen::MatrixXd::Zero(3 * matchedPointsInMetric[i].size(), 9);
        Eigen::VectorXd  b(3 * matchedPointsInMetric[i].size());
        for (uint32_t j = 0; j < matchedPointsInMetric[i].size(); j++)
        {
            b[3 * j]   = matchedPointsInMetric[i][j].x();
            b[3 * j + 1] = matchedPointsInMetric[i][j].y();
            b[3 * j + 2] = matchedPointsInMetric[i][j].z();
            Cof(3 * j, 0)  = matchedPointsInMetric[i + 1][j].x();
            Cof(3 * j, 1)  = matchedPointsInMetric[i + 1][j].y();
            Cof(3 * j, 2)  = matchedPointsInMetric[i + 1][j].z();
            Cof(3 * j + 1, 3)  = matchedPointsInMetric[i + 1][j].x();
            Cof(3 * j + 1, 4)  = matchedPointsInMetric[i + 1][j].y();
            Cof(3 * j + 1, 5)  = matchedPointsInMetric[i + 1][j].z();
            Cof(3 * j + 2, 6)  = matchedPointsInMetric[i + 1][j].x();
            Cof(3 * j + 2, 7)  = matchedPointsInMetric[i + 1][j].y();
            Cof(3 * j + 2, 8)  = matchedPointsInMetric[i + 1][j].z();
        }
        Eigen::MatrixXd CofS(9, 9);
        CofS = Cof.transpose() * Cof;

        //cout << "CofS: " << CofS << endl;

        Eigen::VectorXd bS(9), ba(9);
        bS   = Cof.transpose() * b;
        ba   = CofS.inverse() * bS;
        Eigen::Matrix3d  E;
        for (int a = 0; a < 9; a++)
        {
            E(a / 3, a % 3) = ba(a);
        }
        JacobiSVD<MatrixXd> svd(E, ComputeThinU | ComputeThinV);
        cout << "U" << svd.matrixU() << endl << "V" << svd.matrixV() << endl << "E" << svd.singularValues() << endl;
        //Rot = svd.matrixU() * W.inverse() * svd.matrixV().transpose();
        Rot = svd.matrixU() * svd.matrixV().transpose();
        Eigen::Vector3d rpy;
        rpy = Qbw2RPY(Quaterniond(Rot));
        cout << "H rpy" << rpy.x() << " " << rpy.y() << " " << rpy.z() << endl;
        Rotation.push_back(Rot);
    }
}
