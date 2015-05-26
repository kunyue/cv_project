#include    "cvtool.h"


void CVTool::detectFeatureSURF(Mat a_img, vector<KeyPoint> &a_keypoint, Mat &adescp)
{
    int     minHessian = 400;
    SurfFeatureDetector     surfDetector(minHessian);
    surfDetector.detect(a_img, a_keypoint);
    cv::SurfDescriptorExtractor surfExtractor;
    surfExtractor.compute(a_img, a_keypoint, adescp);
}

void CVTool::detectFeatureFAST(Mat a_img,vector<KeyPoint> &a_keypoint, Mat &adescp)
{
    cv::FastFeatureDetector  fast(15);
    fast.detect(a_img, a_keypoint);
    cv::SurfDescriptorExtractor extra;
    extra.compute(a_img, a_keypoint, adescp);
}

void CVTool::detectFeatureSIFT(Mat a_img, vector<KeyPoint> &a_keypoint, Mat &adescp)
{
    cv::SiftFeatureDetector     detector;
    detector.detect(a_img, a_keypoint);
    cv::SiftDescriptorExtractor siftExtractor;
    siftExtractor.compute(a_img, a_keypoint, adescp);
}

void CVTool::detectFeatureMSER(Mat a_img)
{
    Mat     a_gray;
    cvtColor(a_img, a_gray, CV_RGB2GRAY);
}
Mat CVTool::detectFeatureHaris(Mat a_img)
{
    Mat     a_gray, b_gray;
    cv::cvtColor(a_img, a_gray, CV_RGB2GRAY);
    Mat     dst, dst_norm, dst_norm_scaled;
    dst     = Mat::zeros(a_img.size(), CV_32FC1);

    int     blockSize   = 2;
    int     apertureSize = 5;
    double  k   = 0.041;
    int     thresh = 183;
    cv::cornerHarris( a_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs(dst_norm, dst_norm_scaled);
    // Drawing a circle around corners
    for ( int j = 0; j < dst_norm.rows ; j++ )
    {
        for ( int i = 0; i < dst_norm.cols; i++ )
        {
            if ( (int) dst_norm.at<float>(j, i) > thresh )
            {
                circle( dst_norm_scaled, Point( i, j ), 5, Scalar(0), 1, 8, 0 );
            }
        }
    }
    return dst_norm_scaled;
}


void CVTool::matchFeatures(Mat adescp, Mat  bdescp, vector<DMatch> &matches, vector<KeyPoint> a_kp, vector<KeyPoint> b_kp, vector<DMatch> &good_matches)
{
    cv::BFMatcher   matcher;
    matcher.match(adescp, bdescp, matches);

    vector<Point2f>     a_p, b_p;
    for (uint32_t i = 0; i < matches.size(); i++)
    {
        a_p.push_back(a_kp[matches[i].queryIdx].pt);
        b_p.push_back(b_kp[matches[i].trainIdx].pt);
    }
    Mat     mask, H;
    //cv::findFundamentalMat(a_p, b_p, CV_FM_RANSAC, 3, 0.99, mask);
    H = findHomography(a_p, b_p, CV_RANSAC, 3, mask);
    for (uint32_t i = 0; i < matches.size(); i++)
    {
        if ((unsigned int)mask.at<uchar>(i))
        {
            good_matches.push_back(matches[i]);
        }
    }
}

void CVTool::visualizeMatching(Mat a_img, Mat b_img,  vector<KeyPoint> a_kp, vector<KeyPoint> b_kp, vector<DMatch>  good_matches,   Mat &img_matches)
{
    //drawMatches(a_img, a_kp, b_img, b_kp, good_matches, img_matches);
    drawMatches( a_img, a_kp, b_img, b_kp, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //imshow("Matches", img_matches);
}


Mat CVTool::repairImage(CVTool cvtool, const cv::Mat & damaged_img, const cv::Mat & complete_img)
{
    vector<KeyPoint>    kp_dam, kp_cmp;
    Mat     des_dam, des_cmp;
    cvtool.detectFeatureSIFT(damaged_img, kp_dam, des_dam);
    cvtool.detectFeatureSIFT(complete_img, kp_cmp, des_cmp);
    vector<DMatch>      matches, good_matches;
    vector<DMatch>   best_matches;
    cvtool.matchFeatures(des_dam, des_cmp, matches, kp_dam, kp_cmp, good_matches);

    Mat     img_matches;
    vector<DMatch>      match_use;
    vector<Point2f>     dam_img, cmp_img;
    if (0) //(good_matches.size() >= 8) //
    {
        for (unsigned long i = 0; i < good_matches.size(); i++)
        {
            match_use.push_back( good_matches[i] );
            dam_img.push_back( kp_dam[good_matches[i].queryIdx].pt);
            cmp_img.push_back( kp_cmp[good_matches[i].trainIdx].pt);
        }
        cout << "good_matches" << endl;
    }
    else if (matches.size() >= 8)
    {
        for (unsigned long i = 0; i < matches.size(); i++)
        {
            match_use.push_back( matches[i] );
            dam_img.push_back( kp_dam[matches[i].queryIdx].pt);
            cmp_img.push_back( kp_cmp[matches[i].trainIdx].pt);
        }
        cout << "general_matches" << endl;
    }
    cvtool.visualizeMatching(damaged_img, complete_img, kp_dam, kp_cmp, match_use, img_matches);
    Mat     H;
    H   = findHomography(dam_img, cmp_img, CV_RANSAC);

    //draw lines between the corners
    vector<Point2f>   a_corners(4);
    a_corners[0]    = cvPoint(0, 0);
    a_corners[1]    = cvPoint(damaged_img.cols, 0);
    a_corners[2]    = cvPoint(damaged_img.cols, damaged_img.rows);
    a_corners[3]    = cvPoint(0, damaged_img.rows);

    vector<Point2f>     b_corners(4);
    cv::perspectiveTransform(a_corners, b_corners, H);

    line( img_matches, b_corners[0] + Point2f( damaged_img.cols, 0), b_corners[1] + Point2f( damaged_img.cols, 0), Scalar(0, 255, 0), 1 );
    line( img_matches, b_corners[1] + Point2f( damaged_img.cols, 0), b_corners[2] + Point2f( damaged_img.cols, 0), Scalar( 0, 255, 0), 1 );
    line( img_matches, b_corners[2] + Point2f( damaged_img.cols, 0), b_corners[3] + Point2f( damaged_img.cols, 0), Scalar( 0, 255, 0), 1 );
    line( img_matches, b_corners[3] + Point2f( damaged_img.cols, 0), b_corners[0] + Point2f( damaged_img.cols, 0), Scalar( 0, 255, 0), 1 );
    imshow("show matches", img_matches);

    Mat   rep_img;
    damaged_img.copyTo(rep_img);
    for (int i = 0; i < damaged_img.cols; i++)
    {
        vector<Point2f>     ap(damaged_img.rows);
        vector<Point2f>     bp(damaged_img.rows);
        for (int j = 0; j < damaged_img.rows; j++)
        {
            ap[j]    = Point2f(i, j);
        }
        perspectiveTransform(ap, bp, H);
        for (int j = 0; j < damaged_img.rows; j++)
        {
            for ( int a = 0; a < 3; a ++)
            {
                rep_img.at<Vec3b>(j, i)[a]   = complete_img.at<Vec3b>((int)bp[j].y, (int)bp[j].x)[a];
            }
        }
    }
    return rep_img;
}

void CVTool::computeFundMatrix(CVTool cvtool, Mat a_img, Mat b_img, Mat &F, vector<Point2f> &a_p, vector<Point2f> &b_p, vector<DMatch> &best_matches)
{
    vector<KeyPoint>    a_kp, b_kp;
    vector<DMatch>      matches, good_matches;
    Mat     ades, bdes;
    cvtool.detectFeatureSURF(a_img, a_kp, ades);
    cvtool.detectFeatureSURF(b_img, b_kp, bdes);
    cvtool.matchFeatures(ades, bdes, matches, a_kp, b_kp, good_matches);
    vector<Point2f>     a_pt, b_pt;
    for (uint32_t i = 0; i < matches.size(); i++)
    {
        a_pt.push_back(a_kp[matches[i].queryIdx].pt);
        b_pt.push_back(b_kp[matches[i].trainIdx].pt);
    }
    Mat  mask;
    vector<DMatch>  f_match;
    vector<Point2f>  a_pt_fransac, b_pt_fransac;
    F   = findFundamentalMat(a_pt, b_pt, cv::FM_RANSAC, 2, 0.99, mask);
    for (uint32_t i = 0; i < matches.size(); i++ )
    {
        if ((unsigned int)mask.at<uchar>(i))
        {
            f_match.push_back(matches[i]);
            a_pt_fransac.push_back(a_pt[i]);
            b_pt_fransac.push_back(b_pt[i]);
        }
    }
    F   = findFundamentalMat(a_pt_fransac, b_pt_fransac, cv::FM_8POINT);
    vector<Vec3f>   a_lines, b_lines;
    computeCorrespondEpilines(a_pt_fransac, 1, F, a_lines);
    computeCorrespondEpilines(b_pt_fransac, 2, F, b_lines);
    cout << "FM:" << F << endl;

    double  match_quality[f_match.size()];
    int     idx_match[f_match.size()];
    for (uint32_t i = 0; i < f_match.size(); i++)
    {
        match_quality[i] = pow( b_pt_fransac[i].x * a_lines[i][0] / a_lines[i][2] + b_pt_fransac[i].y * a_lines[i][1] / a_lines[i][2] + 1, 2)
                           + pow( a_pt_fransac[i].x * b_lines[i][0] / b_lines[i][2] + a_pt_fransac[i].y * b_lines[i][1] / b_lines[i][2] + 1,  2);
        idx_match[i] = i;
    }

    //cout << "*****************************"<< endl;
    for (uint32_t i = 0; i < f_match.size(); i++)
    {
        double changle_q;
        int    changle_idx;
        for (uint32_t j = 0; j < f_match.size() - 1 - i; j++)
        {
            if (match_quality[j] < match_quality[j + 1])
            {
                changle_idx     = idx_match[j];
                changle_q       = match_quality[j];
                idx_match[j]    = idx_match[j + 1];
                match_quality[j] = match_quality[j + 1];
                idx_match[j + 1]  = changle_idx;
                match_quality[j + 1]  = changle_q;
            }
        }
        if (i < 20)
            cout << "best_match_in_order:" << idx_match[f_match.size() - 1 - i]
                 << "\t" << match_quality[f_match.size() - i - 1] << endl;
        best_matches.push_back(f_match[idx_match[f_match.size() - 1 - i]]);
    }
    for (uint32_t i = 0; i < best_matches.size(); i++)
    {
        a_p.push_back(a_kp[best_matches[i].queryIdx].pt);
        b_p.push_back(b_kp[best_matches[i].trainIdx].pt);
    }
}

Mat CVTool::visualizeEpipolarLine(CVTool cvtool, Mat &a_img, Mat &b_img, vector<Point2f> a_p, vector<Point2f> b_p, vector<DMatch> match, Mat F, uint32_t NumPlotLines)
{
    vector<Vec3f>    a_lines, b_lines;
    cv::computeCorrespondEpilines(a_p, 1, F, a_lines);
    cv::computeCorrespondEpilines(b_p, 2, F, b_lines);
    //for(uint32_t i=0; i < match.size() && i<NumPlotLines; i++)
    if (NumPlotLines < match.size())
    {
        int i = NumPlotLines;
        cv::circle(b_img, b_p[i], 3, Scalar(0, 0, 255));
        cv::circle(a_img, a_p[i], 3, Scalar(0, 0, 255));
        cv::line(b_img,
                 Point2f(0, -a_lines[i][2] / a_lines[i][1]),
                 Point2f( b_img.cols, (-a_lines[i][2] - a_lines[i][0] * b_img.cols) / a_lines[i][1]),
                 Scalar(0, 255, 0), 1);
        cv::line(a_img,
                 Point2f(0, -b_lines[i][2] / b_lines[i][1]),
                 Point2f( a_img.cols, (-b_lines[i][2] - b_lines[i][0] * a_img.cols) / b_lines[i][1]),
                 Scalar(0, 255, 0), 1);
    }
    Mat towImg;
    towImg = cvtool.cvShowTwoImage(a_img, b_img);
    return  towImg;
}

Mat CVTool::cvShowTwoImage(Mat a_img, Mat b_img)
{
    int     rows = a_img.rows;
    if ( rows < b_img.rows)  rows = b_img.rows;
    Mat     testImg(rows, 2 * a_img.cols, CV_8UC3);
    Rect    rect1(0, 0, a_img.cols, a_img.rows);
    Rect    rect2(a_img.cols, 0, a_img.cols, a_img.rows);
    a_img.copyTo(testImg(rect1));
    b_img.copyTo(testImg(rect2));
    return  testImg;
}
