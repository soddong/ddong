#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <vector>
#include<math.h>

using namespace std;
using namespace cv;

vector<unsigned char> end_angle;
vector<unsigned char> bif_angle;
vector<unsigned char> type;
vector<pair<int, int>> bif;
vector<pair<int, int>> end_;

void segmentation(Mat& src, Mat& mask, int blk_size, int variance_threshold, int mean_threshold) {

    int row = src.rows;
    int col = src.cols;
    Mat mean_mask; //mask by mean in each block(fore_ground= low, background=high)
    Mat variance_mask;//mask by variance in each block(fore_ground= high, background=low)
    copyMakeBorder(src, mean_mask, 0, blk_size, 0, blk_size, BORDER_CONSTANT, 0);
    copyMakeBorder(src, variance_mask, 0, blk_size, 0, blk_size, BORDER_CONSTANT, 0);

    for (int i = 0; i < row; i += blk_size) { //shift by block size at y-axis
        for (int j = 0; j < col; j += blk_size) {//shift by block size at x-aixs
            int sum = 0;
            int mean = 0;
            int variance = 0;
            for (int y = i; (y < i + blk_size) && (y < row); y++) {
                for (int x = j; (x < j + blk_size) && (x < col); x++) {
                    sum += mean_mask.at<uchar>(y, x); //sum of all pixels in (i,j)th block
                }
            }
            mean = sum / (blk_size * blk_size); //Calculate the pixel mean within a block by dividing by the square of the block size

            for (int y = i; (y < i + blk_size) && (y < row); y++) {
                for (int x = j; (x < j + blk_size) && (x < col); x++) {
                    int value = variance_mask.at<uchar>(y, x) - mean;
                    variance += pow((value - mean), 2);

                }
            }
            variance = variance / (blk_size * blk_size);    //Calculate pixel value deviation within one block

            if (mean > mean_threshold) {
                for (int y = i; (y < i + blk_size) && (y < src.rows); y++) {
                    for (int x = j; (x < j + blk_size) && (x < src.cols); x++) {
                        mean_mask.at<uchar>(y, x) = 0;  //If the mean pixel value within a block is higher than the mean threshold value, set it as backgroud 
                    }
                }
            }
            else {
                for (int y = i; (y < i + blk_size) && (y < src.rows); y++) {
                    for (int x = j; (x < j + blk_size) && (x < src.cols); x++) {
                        mean_mask.at<uchar>(y, x) = 255; //If the mean pixel value within a block is lower than the mean threshold value, set it as foreground 
                    }
                }
                if (variance < variance_threshold) {
                    for (int y = i; (y < i + blk_size) && (y < src.rows); y++) {
                        for (int x = j; (x < j + blk_size) && (x < src.cols); x++) {
                            variance_mask.at<uchar>(y, x) = 0; //If the variance pixel value within a block is lower than the mean threshold value, set it as backgroud   
                        }
                    }
                }
                else {
                    for (int y = i; (y < i + blk_size) && (y < src.rows); y++) {
                        for (int x = j; (x < j + blk_size) && (x < src.cols); x++) {
                            variance_mask.at<uchar>(y, x) = 255; //If the variance pixel value within a block is higher than the mean threshold value, set it as foreground
                        }
                    }
                }
            }
        }
        mean_mask = mean_mask(Rect(0, 0, src.cols, src.rows));
        variance_mask = variance_mask(Rect(0, 0, src.cols, src.rows)); //Fit both masks to original image size
        mask = mean_mask & variance_mask; // the mask of the final segmentation by two masks and operation.


    }
}


void block_orientation(Mat& src, Mat& dst, int blockSize) {
    Mat draw_image = src.clone();
    int col = src.cols;
    int row = src.rows;
    int type = src.type();
    int blockH, blockW;

    Mat Gx(src.size(), type), Gy(src.size(), type);
    Mat Gsx(src.size(), type), Gsy(src.size(), type);
    Mat Fx(src.size(), type), Fy(src.size(), type);
    Mat orientation(src.size(), type);
    Mat grad_theta(src.size(), type);

    // Get a gradient through a sobel filter (Gx, Gy)
    Sobel(src, Gx, type, 1, 0, 3);
    Sobel(src, Gy, type, 0, 1, 3);

    //  Get a local orientation gradient vector for each block centered at (m, n)
    for (int n = 0; n < row; n += blockSize)
    {
        for (int m = 0; m < col; m += blockSize)
        {
            float Gsx = 0.0;
            float Gsy = 0.0;

            // Handling when remaining block length is less than blocksize
            blockH = ((row - n) < blockSize) ? (row - n) : blockSize;
            blockW = ((col - m) < blockSize) ? (col - m) : blockSize;

            // Get squared gradient vector (Gsx, Gsy)
            for (int v = n; v < n + blockH; v++)
            {
                for (int u = m; u < m + blockW; u++)
                {
                    Gsx += (Gx.at<float>(v, u) * Gx.at<float>(v, u)) - (Gy.at<float>(v, u) * Gy.at<float>(v, u));
                    Gsy += 2 * (Gx.at<float>(v, u) * Gy.at<float>(v, u));
                }
            }

            // exception handling for tan(0) and tan(inf)
            if (Gsx != 0.0 && Gsy != 0.0) {
                // Get angle Gradient Orientation by given Gsx, Gsy
                grad_theta.at<float>(n, m) = 0.5 * fastAtan2(Gsy, Gsx) * (CV_PI / 180);
            }

            // because it square all complex numbers. grad_theta => 2*grad_theta
            // Get the x-axis, y-axis vector of Gradient Orientation
            Fx.at<float>(n, m) = cos(2 * grad_theta.at<float>(n, m));
            Fy.at<float>(n, m) = sin(2 * grad_theta.at<float>(n, m));

            // Fill each block with one vector
            for (int v = n; v < n + blockH; v++)
            {
                for (int u = m; u < m + blockW; u++)
                {
                    Fx.at<float>(v, u) = Fx.at<float>(n, m);
                    Fy.at<float>(v, u) = Fy.at<float>(n, m);
                }
            }
        }
    }

    // apply Gaussian filter 
    cv::GaussianBlur(Fx, Fx, Size(5, 5), 0);
    cv::GaussianBlur(Fy, Fy, Size(5, 5), 0);

    // Finally, Get the dominant direction of each block.
    for (int n = 0; n < row; n++)
    {
        for (int m = 0; m < col; m++)
        {
            orientation.at<float>(n, m) = 0.5 * fastAtan2(Fy.at<float>(n, m), Fx.at<float>(n, m)) * CV_PI / 180;

            // Line only when it is divided into block size
            if ((m % blockSize) == 0 && (n % blockSize) == 0) {
                int x = m;
                int y = n;
                // R is length of line
                int R = blockSize / sqrt(2);
                // ridge orient direction (dx, dy) 
                float dx = R * cos(orientation.at<float>(n, m) - CV_PI / 2);
                float dy = R * sin(orientation.at<float>(n, m) - CV_PI / 2);

                //draw line direction
                line(draw_image, Point(x, y), Point(x + dx, y + dy), Scalar::all(255), 2, LINE_AA, 0);

            }

        }
    }
    cv::imshow("orientation field", orientation);
    cv::imshow("orientation_dirction", draw_image);

    dst = orientation;
}

void gabor_filter(Mat& src, Mat& ori, Mat& dst) {
   
    dst = src.clone();
    Mat kernel; //kernel mat saving after applying gabor filter
    int kernel_size = 5; // gabor filter kernel size
    int lsize = (kernel_size - 1) / 2; // size of padding border for operation

    // Padding input image border for operation and saving to temp
    Mat temp = src.clone(); 
    copyMakeBorder(temp, temp, lsize, lsize, lsize, lsize, BORDER_REFLECT); 
    Mat gab(temp.size(), temp.type());

    // Padding Orientaion Imge border for operation and saving to temp2
    Mat temp2 = ori.clone();
    copyMakeBorder(temp2, temp2, lsize, lsize, lsize, lsize, BORDER_REFLECT);

    // parameter of gabor filter
    double sig = 3, lm = 7, gm = 1, ps = 0;
    double theta;
    float val;

    // Gabor filtering
    for (int m = lsize; m < temp.rows - lsize; m++) {
        for (int n = lsize; n < temp.cols - lsize; n++) {
            theta = temp2.at<float>(m, n); // direction of each block
            // Get 2d kernel through getGaborKernel
            kernel = getGaborKernel(Size(kernel_size, kernel_size), sig, theta, lm, gm, ps, CV_32F);
            val = 0;
            // colvolution window and kernel
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    val += temp.at<float>(m - lsize + k, n - lsize + l) * kernel.at<float>(kernel_size - 1 - k, kernel_size - 1 - l);
                }
            }
            // store at gab mat
            gab.at<float>(m, n) = val / (kernel_size * kernel_size);
        }
    }

    // Roi padding image
    Rect roi(lsize, lsize, dst.cols, dst.rows);

    // final result 
    Mat gab_roi = gab(roi);
    dst = gab_roi;

    // convert dst type
    dst.convertTo(dst, CV_8U, 255, 0);
    cv::imshow("roi", dst);
}

//zhang Suen skeletionization algorithm
void thinning(Mat& img, Mat& dst) {
    img = img / 255; // Convert 0/255 binary image to 0/1 binary image
    Mat prev = Mat::zeros(img.size(), CV_8UC1);
    Mat diff;

    do {
        for (int step = 0; step < 2; step++) {
            Mat marker = Mat::zeros(img.size(), CV_8UC1); //Mark the marker with pixels that meet the conditions
            uchar* ptrUp, * ptrCurr, * ptrDown;
            uchar* leftUp, * up, * rightUp;
            uchar* left, * curr, * right;
            uchar* leftDown, * down, * rightDown;  // 3x3 block pixel pointer based on current pixel

            int a, b, c, d;

            ptrCurr = img.ptr<uchar>(0); //Pointer in the current pixel row (starting from the 0th row)
            ptrDown = img.ptr<uchar>(1); //Pointer in the down pixel row (starting from the 1th row)
            for (int i = 1; i < img.rows - 1; i++) {

                ptrUp = ptrCurr;
                ptrCurr = ptrDown;
                ptrDown = img.ptr<uchar>(i + 1); //Shift the pixel's row pointer one by one

                up = &(ptrUp[0]);
                rightUp = &(ptrUp[1]);
                curr = &(ptrCurr[0]);
                right = &(ptrCurr[1]);
                down = &(ptrDown[0]);
                rightDown = &(ptrDown[1]);

                for (int j = 1; j < img.cols - 1; j++) {
                    leftUp = up;
                    up = rightUp;
                    rightUp = &(ptrUp[j + 1]);
                    left = curr;
                    curr = right;
                    right = &(ptrCurr[j + 1]);
                    leftDown = down;
                    down = rightDown;
                    rightDown = &(ptrDown[j + 1]); //Shift each pixel pointer one column

                    a = (*up == 0 && *rightUp == 1) + (*rightUp == 0 && *right == 1) +
                        (*right == 0 && *rightDown == 1) + (*rightDown == 0 && *down == 1) +
                        (*down == 0 && *leftDown == 1) + (*leftDown == 0 && *left == 1) +
                        (*left == 0 && *leftUp == 1) + (*leftUp == 0 && *up == 1);
                    b = *leftUp + *up + *rightUp + *left + *right + *leftDown + *down + *rightDown;
                    c = (step == 0) ? (*up * *right * *down) : (*up * *right * *left);
                    d = (step == 0) ? (*right * *down * *left) : (*up * *down * *left); // computation of four condition

                    if (a == 1 && (b >= 2 && b <= 6) && c == 0 && d == 0) marker.ptr<uchar>(i)[j] = 1; //Mark 1 on marker if condition is satisfied
                }
            }
            img = img & (~marker); //Delete all pixels marked 1 in the marker from the source
        }
        absdiff(img, prev, diff);
        img.copyTo(prev);
    } while (countNonZero(diff) > 0);

    dst = img * 255;
}

void detectMinutiae(Mat& src, Mat& seg, Mat& ori) {
    Mat inputImage = src.clone(); 
    Mat Mask = seg.clone();
    Mat minutiae;
 
    // To draw red, yellow circle, convert color of inputimage to BGR
    cvtColor(inputImage, minutiae, COLOR_GRAY2BGR);

    int ending = 0, cnt_end = 0;
    int bifurcation = 0, cnt_bif = 0;
    int Cn;
    int max_i = 0;

    // 
    for (int i = 1; i < inputImage.rows - 1; i++) {
        int min_j = 100;
        for (int j = 1; j < inputImage.cols - 1; j++) {
            Cn = 0;
            ending = 0;
            bifurcation = 0;

            // Check if pixel is background portion or not.
            if (Mask.at<uchar>(i, j) == 255 && Mask.at<uchar>(i - 1, j) == 255 && Mask.at<uchar>(i, j - 1) == 255
                && Mask.at<uchar>(i + 1, j) == 255 && Mask.at<uchar>(i, j + 1) == 255) {
                // Check if pixel is border of fingerprint or not.
                if (min_j > j) min_j = j;
                
                // Count changing point (0-->255 or 255->0) to check if pixel is ending or bifurcation.
                if (inputImage.at<uchar>(i, j) == 255 && Mask.at<uchar>(i + 3, j) == 255 && j > min_j) {
                    if ((inputImage.at<uchar>(i - 1, j - 1)) != (inputImage.at<uchar>(i, j - 1))) Cn++;
                    if ((inputImage.at<uchar>(i, j - 1)) != (inputImage.at<uchar>(i + 1, j - 1))) Cn++;
                    if ((inputImage.at<uchar>(i + 1, j - 1)) != (inputImage.at<uchar>(i + 1, j))) Cn++;
                    if ((inputImage.at<uchar>(i + 1, j)) != (inputImage.at<uchar>(i + 1, j + 1))) Cn++;
                    if ((inputImage.at<uchar>(i + 1, j + 1)) != (inputImage.at<uchar>(i, j + 1))) Cn++;
                    if ((inputImage.at<uchar>(i, j + 1)) != (inputImage.at<uchar>(i - 1, j + 1))) Cn++;
                    if ((inputImage.at<uchar>(i - 1, j + 1)) != (inputImage.at<uchar>(i - 1, j))) Cn++;
                    if ((inputImage.at<uchar>(i - 1, j)) != (inputImage.at<uchar>(i - 1, j - 1))) Cn++;
                }

                // If pixel is ending point,
                if (Cn == 2) {
                    ending = 1;
                    cnt_end++; 
                    end_.push_back(make_pair(j, i)); // store x, y of ending point
                    float angle = ((int)(ori.at<float>(i, j) * (-1)) + (CV_PI / 2)) * 180 * 255 / CV_PI / 360; // convert radian to degree
                    end_angle.push_back((unsigned char)angle); // store angle of ending point
                }

                // If pixel is bifurcation point,
                else if (Cn == 6) {
                    bifurcation = 1;
                    cnt_bif++;
                    bif.push_back(make_pair(j, i)); // store x, y of bifurcation point
                    float angle = ((int)(ori.at<float>(i, j) * (-1)) + (CV_PI / 2)) * 180 * 255 / CV_PI / 360; // convert radian to degree
                    bif_angle.push_back((unsigned char)angle); // store angle of bifurcation point
                }

            }
        }
    }
    src = minutiae;
    cout << minutiae.size() << endl;
    int cnt_minutiae = cnt_end + cnt_bif;
    cout << "end point Num : " << cnt_end << endl;
    cout << "bif point Num : " << cnt_bif << endl;
}


int main() {
    Mat input = imread("db/[38]_2019_5_1_L_P#1.bmp", IMREAD_GRAYSCALE);
    Mat visualization = input.clone();
    Mat dst = input.clone();
    cout << input.cols << endl;
    cout << input.rows << endl;
    cout << input.type() << endl;
    cv::imshow("input image", input);

    //Segmentation//
    Mat mask;
    segmentation(input, mask, 5, 5, 230);
    dilate(mask, mask, Mat(), Point(-1, -1), 3, 0, BORDER_CONSTANT);
    erode(mask, mask, Mat(), Point(-1, -1), 3, 0, BORDER_CONSTANT);
    cv::imshow("mask", mask);

    //Normalization
    equalizeHist(input, dst);
    cv::imshow("normailize", dst);

    //Enhancement
    Mat ori;
    dst.convertTo(dst, CV_32F, 1.0 / 255, 0);
    block_orientation(dst, ori, 10);
    gabor_filter(dst, ori, dst);

    //binarization
    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 0);
    //masked//
    dst |= ~mask;
    cv::imshow("binary", dst);

    //thinnging
    dst = ~dst;
    thinning(dst, dst);
    cv::imshow("thinning", dst);
    erode(mask, mask, Mat(), Point(-1, -1), 5, 0, BORDER_CONSTANT);

    //minutiae detection
    detectMinutiae(dst, mask, ori);
    cv::resize(dst, dst, Size(400, 600));
    cv::imshow("minutiae", dst);

    //--------------------------------------------------------------------------------------------//
    //------------------- To draw circle ending point & bifurcation point-------------------------//
    cvtColor(visualization, visualization, COLOR_GRAY2BGR);
    for (int i = 0; i < end_.size(); i++) {
        int x = end_[i].first;
        int y = end_[i].second;
        circle(visualization, Point(x, y), 2, Scalar(255, 0, 0), 1, 8, 0);
    }
    for (int i = 0; i < bif.size(); i++) {
        int x = bif[i].first;
        int y = bif[i].second;
        circle(visualization, Point(x, y), 2, Scalar(0, 0, 255), 1, 8, 0);
    }

    // for convenience's sake
    cv::resize(visualization, visualization, Size(400, 600));
    cv::imshow("result", visualization);

    // Print minutiea numbers
    int N_end = end_.size();
    int N_bif = bif.size();
    int N_minutiae = N_end + N_bif;
    cout << "minutiae of Number: " << N_minutiae << endl;

    //--------------------------------------------------------------------------------------------//
    //--------------------------- To output values as a binfile-----------------------------------//
    ofstream output("[38]_2019_5_1_L_P#1.bin", ios::out | ios::binary);
    output.write((char*)&dst.cols, sizeof(int)); // 4byte
    output.write((char*)&dst.rows, sizeof(int)); // 4byte
    output.write((char*)&N_minutiae, sizeof(int)); // 4byte

    int cnt = 0; // value for limit count
    int type_e = 1; // type 1 is ending point
    int type_b = 3; // type 3 is bifurcation point

    for (int k = 0; k < N_end -1; k++) {
        cnt++; // To count the limit
        if (cnt >= 50) break; // 50 is limit
        
        output.write((char*)&end_[k].first, sizeof(int)); // 4byte
        output.write((char*)&end_[k].second, sizeof(int)); // 4byte
        output.write((char*)&end_angle[k], sizeof(uchar)); // 1byte
        output.write((char*)&type_e, sizeof(uchar)); // 1byte
        printf("X[%d]: %d Y[%d]: %d O[%d]: %d T[%d]: 1\n", k, end_[k].first, k, end_[k].second, k, end_angle[k], k);
    }

    for (int k = 0; k < N_bif; k++) {
        cnt++; // To count the limit
        if (cnt >= 50) break; // 50 is limit
        //int angle = (int)(ori.at <float> (bif[k].first, bif[k].second) * (-1) + (CV_PI / 2)) * 180 * 255 / CV_PI / 360;
        output.write((char*)&bif[k].first, sizeof(int)); // 4byte
        output.write((char*)&bif[k].second, sizeof(int)); // 4byte
        output.write((char*)&bif_angle[k], sizeof(uchar)); // 1byte
        output.write((char*)&type_b, sizeof(uchar)); // 1byte
        printf("X[%d]: %d Y[%d]: %d O[%d]: %d T[%d]: 3\n", k, bif[k].first, k, bif[k].second, k, bif_angle[k], k);
    }

    output.close(); // close bin file

    waitKey(0);
    return 0;
}
