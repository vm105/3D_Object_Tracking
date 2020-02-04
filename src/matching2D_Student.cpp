#include <numeric>
#include "matching2D.hpp"

using namespace std;

float avg_kp = 0.0f;
float avg_kp_time = 0.0f;

float avg_desc_time = 0.0f;

float avg_match = 0.0f;
float avg_match_time = 0.0f;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    float knn_threshold = 0.8f;
    double t;
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        cv::Mat desc_src_temp, desc_ref_temp;
        descSource.convertTo(desc_src_temp, CV_32F);
        descRef.convertTo(desc_ref_temp, CV_32F);
        descSource = desc_src_temp;
        descRef = desc_ref_temp;
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        t = static_cast<double>(cv::getTickCount());
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = (static_cast<double>(cv::getTickCount()) - t)/ cv::getTickFrequency();
    }

    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> match_vector;
        t = static_cast<double>(cv::getTickCount());
        matcher->knnMatch(descSource, descRef, match_vector, 2);

        for (auto it = match_vector.begin(); it != match_vector.end(); ++it)
        {
            if (((*it)[0].distance / (*it)[1].distance) < knn_threshold)
            {
                matches.push_back((*it)[0]);
            }
        }
        t = (static_cast<double>(cv::getTickCount()) - t)/ cv::getTickFrequency();
    }
    std::cout << matcherType << " with " << selectorType << " found " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
    avg_match += matches.size();
    avg_match_time += (1000 * t / 1.0);
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create((img.rows* img.cols), 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    avg_desc_time +=  (1000 * t / 1.0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    avg_kp_time += (1000 * t / 1.0);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, true, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    avg_kp_time += (1000 * t / 1.0);
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // create generic detector pointer variable
    cv::Ptr<cv::Feature2D> detector_ptr;

    //assign the detector pointer the correct object
    if (detectorType.compare("FAST") == 0)
    {
        detector_ptr = cv::FastFeatureDetector::create();
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector_ptr = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector_ptr = cv::ORB::create((img.rows* img.cols), 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector_ptr = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector_ptr = cv::xfeatures2d::SIFT::create();
    }

    double t = (double)cv::getTickCount();
    detector_ptr->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    avg_kp_time += (1000 * t / 1.0);
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void apply_box_filter(std::vector<cv::KeyPoint> &keypoints, cv::Rect& rectangle)
{
    std::vector<cv::KeyPoint> temp;
    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
    {
        if (rectangle.contains((*it).pt))
        {
            temp.push_back(*it);
        }
    }
    avg_kp += temp.size();
    keypoints.swap(temp);
}

float get_avg_kp()
{
    return avg_kp / 10;
}

float get_avg_kp_time()
{
    return avg_kp_time / 10;
}

float get_avg_desc_time()
{
    return avg_desc_time / 10;
}

float get_avg_match()
{
    return avg_match / 10;
}

float get_avg_match_time()
{
    return avg_match_time / 10;
}

