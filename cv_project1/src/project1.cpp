#include    "cvtool.h"

CVTool  cvtool;

Mat     dst, src, d_img, c_img, ac_img;
CVTool  tool, rep;
void    repair_triger(Point pre_pt, Point cur_pt)
{
    Mat    d_part   = d_img(Rect(pre_pt.x, pre_pt.y, (cur_pt.x-pre_pt.x), (cur_pt.y-pre_pt.y)));
    Mat    rep_img  = tool.repairImage(rep, d_part, c_img);
    Mat    rep_full;
    d_img.copyTo(rep_full);
    rep_img.copyTo(rep_full.rowRange(pre_pt.y, cur_pt.y).colRange(pre_pt.x, cur_pt.x));
    imshow("full_IMAGE_Repaired", rep_full);
    imshow("a_cmp_image",ac_img);
}
void    on_mouse(int event, int x, int y, int flags, void* )
{
    static Point pre_pt;
    static Point cur_pt;
    CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
    char temp[50];
    Vec3b intensity = src.at<Vec3b>(Point(x, y));
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        dst.copyTo(src);	
        sprintf(temp,"(%d,%d,%d,%d,%d)",x,y,intensity.val[0],intensity.val[1],intensity.val[2]);
        pre_pt = cvPoint(x,y);
        putText(src,temp, pre_pt, FONT_HERSHEY_SIMPLEX, 0.5,cvScalar(0,0, 0, 255),1,8);
        circle( src, pre_pt, 3,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
        imshow( "the_damaged_image", src );
        src.copyTo(dst);
    }
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
    {
        dst.copyTo(src);
        sprintf(temp,"(%d,%d,%d,%d,%d)",x,y,intensity.val[0],intensity.val[1],intensity.val[2]);
        cur_pt = cvPoint(x,y);		
        putText(src,temp, cur_pt,FONT_HERSHEY_SIMPLEX, 0.5,cvScalar(0,0, 0, 255),1,8);
        cv::rectangle(src, pre_pt, cur_pt, cvScalar(0,255,0,0), 1, CV_AA, 0 );
        imshow( "the_damaged_image", src );
    }
    else if( event == CV_EVENT_LBUTTONUP )
    {
        dst.copyTo(src);
        sprintf(temp,"(%d,%d,%d,%d,%d)",x,y,intensity.val[0],intensity.val[1],intensity.val[2]);
        cur_pt = Point(x,y);		
        putText(src,temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5,cvScalar(0,0, 0, 255),1,8);
        circle( src, cur_pt, 3,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
        cv::rectangle( src, pre_pt, cur_pt, cvScalar(0,255,0,0), 1, CV_AA, 0 );
        imshow( "the_damaged_image", src );
        src.copyTo(dst);
        
        repair_triger(pre_pt, cur_pt);
    }
}


int     main(int argc, char **argv)
{
    bool    surf_detect     = false;
    bool    sift_detect     = false;
    bool    fast_detect     = false;
    bool    harris      = false;
    bool    visual      = false;

    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "--surf") == 0)
        {
            surf_detect = true;
        }
        if(strcmp(argv[i], "--sift") == 0)
        {
            sift_detect = true;
        }

        if(strcmp(argv[i], "-f") == 0)
        {
            fast_detect = true;
        }
        if(strcmp(argv[i], "--fast") == 0)
        {
            fast_detect = true;
        }

        if(strcmp(argv[i], "--harris") == 0)
        {
            harris  = true;
        }
        if(strcmp(argv[i], "--visual") == 0)
        {
            visual  = true;
        }
    }
    Mat a_img, b_img;
    a_img   = imread("../src/data/a_damage.png", CV_LOAD_IMAGE_COLOR);
    b_img   = imread("../src/data/b.png",    CV_LOAD_IMAGE_COLOR);
    ac_img  = imread("../src/data/a_complete.png",    CV_LOAD_IMAGE_COLOR);
    if(b_img.empty() || a_img.empty())
    {
        cout << "read b.png failed!" << endl;
        return -1;
    }

    //use surf detect the features
    std::vector<KeyPoint>   a_keypoint, b_keypoint;
    Mat     adescp, bdescp;
    vector< DMatch>     matches, good_matches;
    vector <DMatch> best_matches;  
    Mat     img_matches;
    Mat     a_key_img, b_key_img; 
    if(sift_detect)
    {
        cvtool.detectFeatureSIFT(a_img, a_keypoint, adescp);
        cvtool.detectFeatureSIFT(b_img, b_keypoint, bdescp);
        drawKeypoints(b_img, b_keypoint, b_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(a_img, a_keypoint, a_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("SIFT b.png", b_key_img);
        imshow("SIFT a.png", a_key_img);
    }
    if(surf_detect)
    {
        cvtool.detectFeatureSURF(a_img, a_keypoint, adescp);
        cvtool.detectFeatureSURF(b_img, b_keypoint, bdescp);
        drawKeypoints(b_img, b_keypoint, b_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(a_img, a_keypoint, a_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("a.png", a_key_img);
        imshow("b.png", b_key_img);
    }
    if(fast_detect)
    {
        cvtool.detectFeatureFAST(a_img, a_keypoint, adescp);
        cvtool.detectFeatureFAST(b_img, b_keypoint, bdescp);
        drawKeypoints(b_img, b_keypoint, b_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(a_img, a_keypoint, a_key_img, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        imshow("FAST b.png", b_key_img);
        imshow("FAST a.png", a_key_img);
        cvtool.matchFeatures(adescp, bdescp, matches, a_keypoint, b_keypoint,good_matches);
        cvtool.visualizeMatching(a_img, b_img, a_keypoint, b_keypoint, good_matches, img_matches);
        imshow("Matches", img_matches);
    }
    if(harris) 
    {
        Mat a, b;
        a = cvtool.detectFeatureHaris(ac_img);
        b = cvtool.detectFeatureHaris(b_img);
        imshow("a",a);
        imshow("b",b);
    }
    if(visual)
    {
        cvtool.detectFeatureSURF(a_img, a_keypoint, adescp);
        cvtool.detectFeatureSURF(b_img, b_keypoint, bdescp);
        cvtool.matchFeatures(adescp, bdescp, matches, a_keypoint, b_keypoint,good_matches);
        cvtool.visualizeMatching(a_img, b_img, a_keypoint, b_keypoint, good_matches, img_matches);
        imshow("Matches", img_matches);
    }
/*    
    a_img.copyTo(src);
    a_img.copyTo(dst);
    a_img.copyTo(d_img);
    b_img.copyTo(c_img);
    cvNamedWindow("the_damaged_image",CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback("the_damaged_image", on_mouse, 0);
    imshow("the_damaged_image", src);
    CVTool cv_t;
    Mat F, CoImg1;
    vector<Point2f> a_p, b_p;
    Mat     ImgAC=a_img.clone();
    Mat     ImgB=b_img.clone();
    
    cv_t.computeFundMatrix(cv_t, ImgAC, ImgB, F, a_p, b_p, best_matches);
    for(uint32_t i=0; i<15; i++)
    {
        CoImg1 = cv_t.visualizeEpipolarLine(cv_t, ImgAC, ImgB, a_p, b_p, best_matches, F, i);
    }
    imshow("a_dam & b", CoImg1);
    
    
    CVTool  cv_t_;
    Mat F_, CoImg2; 
    Mat ImgAC_=ac_img.clone();  
    Mat ImgB_ =b_img.clone();  
    vector<Point2f> a_p_, b_p_;
    vector<DMatch> best_matches_;
    cv_t_.computeFundMatrix(cv_t, ImgAC_, ImgB_, F_, a_p_, b_p_, best_matches_);
    for(uint32_t i=0; i<15; i++)
    {
     CoImg2 = cv_t_.visualizeEpipolarLine(cv_t_, ImgAC_, ImgB_, a_p_, b_p_, best_matches_, F_, i);
    }
    imshow("a_com & b", CoImg2);
  */  
    waitKey(0);
    //while(1);
    return 0;
}
