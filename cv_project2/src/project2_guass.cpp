#include "CVTool.h"

int main(int argc, char **argv)
{
    vector<Mat>  srcImage;
    vector<Mat>  imageUndistorted;
    vector<Mat>  H_imgs;
    vector<vector<Point2f> > featureGoodMatched;
    Eigen::Matrix3d     cameraMatrixEigen;

    Mat  img = imread("../dataset/phantom/DJI_0034.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/phantom/DJI_0035.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/phantom/DJI_0036.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/phantom/DJI_0038.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/phantom/DJI_0039.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/phantom/DJI_0040.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    //img = imread("../dataset/phantom/DJI_0041.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    cv::FileStorage  cameraDetail("../config/phantom3.yml", FileStorage::READ);
    //Mat  img = imread("../dataset/DJI_0010.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //img = imread("../dataset/DJI_0011.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //img  = imread("../dataset/DJI_0012.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //img  = imread("../dataset/DJI_0013.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //img  = imread("../dataset/DJI_0014.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    ////img  = imread("../dataset/DJI_0015.JPG", CV_LOAD_IMAGE_COLOR);
    ////srcImage.push_back(img);
    //cv::FileStorage  cameraDetail("../config/inspire.yml", FileStorage::READ);
    //Mat    img = imread("../dataset/tailei/019.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //img  = imread("../dataset/tailei/020.JPG", CV_LOAD_IMAGE_COLOR);
    //srcImage.push_back(img);
    //cv::FileStorage  cameraDetail("../config/tai.yml", FileStorage::READ);
    UndistortImages(srcImage, cameraDetail, cameraMatrixEigen, imageUndistorted);
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0)
        {
            cout << "read file" << endl;
            ReadMatchedPointsInfile(H_imgs, featureGoodMatched);
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            GetMatchedPointUsingImage(imageUndistorted, featureGoodMatched, H_imgs);
        }
        else
            cout << "Error: -d OR -f "  << endl;
    }
    cout << "cM" << cameraMatrixEigen << endl;
    vector<vector<Eigen::Vector3d> >  matchedPointsInMetric;
    vector<int> stitchPoint;
    getStitchingPoint(imageUndistorted, featureGoodMatched, stitchPoint);

    MatchPointsFromPixelToMetric(featureGoodMatched, matchedPointsInMetric, cameraMatrixEigen);
    cout << "Size:" << matchedPointsInMetric.size() <<endl;
    vector<Matrix3d> Rotation;
    GuassNewtonGetRotation(Rotation, matchedPointsInMetric);
    vector<Matrix3d>  Rot;
    DirectSolveRotation(matchedPointsInMetric, Rot); 
    Mat coImage;
    ShowCoImage(imageUndistorted, cameraMatrixEigen, Rotation, stitchPoint, coImage);
    namedWindow("Gauss", WINDOW_NORMAL);
    imshow("Gauss", coImage);
    Mat coImage_;
    ShowCoImage(imageUndistorted, cameraMatrixEigen, Rot, stitchPoint, coImage_);
    namedWindow("DLT", WINDOW_NORMAL);
    imshow("DLT", coImage);
    waitKey(0);
    return  1;
}
