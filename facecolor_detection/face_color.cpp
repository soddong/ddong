#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

bool R1(int R, int G, int B) { //skin range
    bool e1 = (R > 95) && (G > 40) && (B > 20) && ((max(R, max(G, B)) - min(R, min(G, B))) > 15) && (abs(R - G) > 15) && (R > G) && (R > B);
    bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15) && (R > B) && (G > B);
    return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) { //skin range
    bool e3 = (Y >= 0) && (Y <= 255);
    bool e4 = (Cr >= 133) && (Cr <= 173);
    bool e5 = (Cb >= 77) && (Cb <= 127);

    return e3 && e4 && e5;
}
//database retrieved skin thresholds for HUE
bool R3(float H, float S, float V) { //skin range
    bool e6 = ((H > 0) && (H < 18));
    bool e7 = (S > 0.17)&& (S < 0.30) || (S > 0.7)&&(S < 0.8);
    return e7 || e6;
}


Mat GetSkinRGB(Mat const& src) {
    // allocate the result matrix
    Mat dst = src.clone();
    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            // RGB define
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // determine if rgb thresold is skin
            bool a = R1(R, G, B);

            if (!a) // if not skin color
                dst.ptr<Vec3b>(i)[j] = cblack;
            else // if skin color
                dst.ptr<Vec3b>(i)[j] = cwhite;
        }
    }
    return dst;
}

Mat GetSkinHSV(Mat const& src) {
    Mat dst = src.clone();
    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);
    Mat src_hsv;

    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision
    src.convertTo(src_hsv, CV_32FC3);

    // convert rgb color space -> hsv
    cvtColor(src_hsv, src_hsv, COLOR_BGR2HSV);

    // scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // determine if YCrCb thresold is skin
            bool c = R3(H, S, V);

            if (!c) // if not skin color
                dst.ptr<Vec3b>(i)[j] = cblack;
            else  // if skin color
                dst.ptr<Vec3b>(i)[j] = cwhite;
        }
    }
    return dst;
}

Mat GetSkinYCbCr(Mat const& src) {
    // allocate the result matrix
    Mat dst = src.clone();

    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);
    Mat src_ycrcb;

    // convert rgb color space -> ycrcb space
    cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // determine if YCrCb thresold is skin
            bool b = R2(Y, Cr, Cb);

            if (!b) // if not skin color
                dst.ptr<Vec3b>(i)[j] = cblack;
            else // if skin color
                dst.ptr<Vec3b>(i)[j] = cwhite;
        }
    }
    return dst;
}

int main()
{
	Mat src, ing1, ing2,dst;
	Mat rgb;
	Mat hsv, ycrcb;

	//Load image
    src = imread("private/view_point.jpg");

    //Noise reduction
    GaussianBlur(src, dst, Size(3,3), 0, 0);

    //skin color at each color-space 
    rgb = GetSkinRGB(dst);
    hsv = GetSkinHSV(dst);
    ycrcb = GetSkinYCbCr(dst);

    imshow("original", src);
    imshow("filter", dst);
    imshow("rgb", rgb);
    imshow("hsv", hsv);
    imshow("ycrcb", ycrcb);


    waitKey(0);

	return 0;
}