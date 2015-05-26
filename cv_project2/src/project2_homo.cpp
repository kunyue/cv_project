#include "CVTool.h"

int main(int argc, char **argv)
{
    vector<Mat>  srcImage;
    vector<Mat>  imageUndistorted;
    vector<Mat>  H_imgs;
    vector<vector<Point2f> > featureGoodMatched;
    Eigen::Matrix3d     cameraMatrixEigen;

    Mat    img = imread("../dataset/DJI_0010.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img = imread("../dataset/DJI_0011.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    img  = imread("../dataset/DJI_0012.JPG", CV_LOAD_IMAGE_COLOR);
    srcImage.push_back(img);
    cv::FileStorage  cameraDetail("../config/inspire.yml", FileStorage::READ);
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
    for(uint32_t i=0; i<H_imgs.size(); i++)
        cout << "H" << H_imgs[i] << endl;
    //Mat     W, U, VT;
    //cv::SVDecomp(H_imgs[0], W, U, VT);
    //cout << "W: " << W << endl;
    //cout << "U: " << U << endl;
    //cout << "VT: " << VT << endl;
    int imageRows = imageUndistorted[0].rows;
    int imageCols = imageUndistorted[0].cols;
    Mat onePic = Mat::zeros(imageRows, imageCols * 2, imageUndistorted[0].type());
    Mat srcImg;
    vector<Point2f>  pointTar(imageRows);
    vector<Point2f>  pointSrc(imageRows);
    int  numImg;
    for (int i = 0; i < onePic.cols; i++)
    {
        for (int n = 0; n < onePic.rows; n++)
            pointTar[n] = Point2f(i, n);

        if (i < 2 * imageCols / 3)
        {
            for (int m = 0; m < onePic.rows; m++)
            {
                pointSrc[m].x = pointTar[m].x;
                pointSrc[m].y = pointTar[m].y;
            }
            numImg = 0;
        }
        else if (i < 4 * imageCols / 3)
        {
            numImg = 1;
            cv::perspectiveTransform(pointTar, pointSrc, H_imgs[0].inv());
        }
        else if (i > 4 * imageCols / 3)
        {
            numImg = 2;
            cv::perspectiveTransform(pointTar, pointSrc, H_imgs[0].inv() * H_imgs[1].inv());
        }
        for (int j = 0; j < onePic.rows; j++)
        {
            if (pointSrc[j].x >= 0 && pointSrc[j].x < imageCols && pointSrc[j].y >= 0 && pointSrc[j].y < imageRows)
                for (int a = 0; a < 3; a++)
                    onePic.at<Vec3b>(j, i)[a] = imageUndistorted[numImg].at<cv::Vec3b>(pointSrc[j].y, pointSrc[j].x)[a];
        }
    }
    cv::namedWindow("one image", WINDOW_NORMAL);
    imshow("one image", onePic);
    waitKey(0);
    return  1;
}
