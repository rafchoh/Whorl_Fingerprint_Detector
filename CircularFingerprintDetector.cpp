#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    // Set picture filename after it is placed in the same folder as the project
    Mat originalImage = imread("fingerprints2022_41a_3.jpg");

    if (originalImage.empty()) {
        cout << "Failed to load the image!" << endl;
        return -1;
    }

    // Set new values to make the code detect the right fingerprints
    normalize(originalImage, originalImage, 0, 255, NORM_MINMAX);

    Mat grayImage;
    cvtColor(originalImage, grayImage, COLOR_BGR2GRAY);

    double scalingFactor = 0.5;
    grayImage *= scalingFactor;

    int height = grayImage.rows;
    int width = grayImage.cols;

    Mat gaussImage;
    GaussianBlur(grayImage, gaussImage, Size(3, 3), 1.5);

    Mat complexImage;
    Mat layered[] = { 
        Mat_<float>(gaussImage), 
        Mat::zeros(
            gaussImage.size(), 
            CV_32F
        )
    };
    merge(layered, 2, complexImage);
    dft(complexImage, complexImage);

    Mat filteredImage;
    idft(complexImage, filteredImage, DFT_REAL_OUTPUT | DFT_SCALE);

    normalize(filteredImage, filteredImage, 0, 255, NORM_MINMAX);
    filteredImage.convertTo(filteredImage, CV_8U);

    Mat histImage;
    equalizeHist(filteredImage, histImage);

    vector<Vec3f> circles;
    HoughCircles(
        histImage,                                              
        circles,                                                
        HOUGH_GRADIENT,                                         
        1.5,                                                    
        height / 8,                                             
        70,                                                     
        320,                                                     
        0,                                                      
        width / 10                                              
    );

    Mat circlesImage = originalImage.clone();
    for (size_t i = 0; i < circles.size(); i++) {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        int radius = c[2];
        circle(circlesImage, center, radius, Scalar(0, 0, 0), 3, LINE_AA);
    }

    imwrite("grayscale_with_circles.jpg", circlesImage);

    namedWindow("Detected Circles", WINDOW_NORMAL);
    imshow("Detected Circles", circlesImage);

    namedWindow("Fourier", WINDOW_NORMAL);
    imshow("Fourier", filteredImage);

    waitKey(0);

    return 0;
}