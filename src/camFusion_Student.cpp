
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double running_avg = 0.0;
    double running_sum = 0.0;
    double num_points = 0.0;

    // loop over all matches
    for (auto match : kptMatches)
    {
        int curr_kpt_index = match.trainIdx;
        cv::Point2f curr_point = kptsCurr.at(curr_kpt_index).pt;

        if (boundingBox.roi.contains(curr_point))
        {
            running_sum += match.distance;
            running_avg = running_sum/++num_points;
            double err = (match.distance - running_avg);
            // std::cout << "err " << err << " avg dist " << running_avg << " curr dist " << match.distance << std::endl;
            // if the distance is less than equal to moving avg distance add it to the vector
            if (err < 0)
            {
                boundingBox.kptMatches.push_back(match);
            }
        }
    }
}

double get_median(vector<double>& point_vector)
{
    double median = 0.0;
    if (point_vector.size() % 2)
    {
       auto median_it = point_vector.begin();
       std::advance(median_it, point_vector.size()/2);
       nth_element(point_vector.begin(), median_it , point_vector.end());
       median = *median_it;
    }
    else
    {
       auto median_it_1 = point_vector.begin();
       auto median_it_2 = point_vector.begin();

       std::advance(median_it_1,  point_vector.size()/2);
       std::advance(median_it_2, point_vector.size()/2 -1);

       nth_element(point_vector.begin(), median_it_1, point_vector.end());
       nth_element(point_vector.begin(), median_it_2, point_vector.end());

       median = (*median_it_1 + *median_it_2)/2.0;
    }

    return median;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{

    vector<double> dist_ratios;
    double dt = 1/frameRate;
    for (auto match_outer : kptMatches)
    {
        int prev_kpt_index =  match_outer.queryIdx;
        int curr_kpt_index =  match_outer.trainIdx;

        //get outer prev and curr kpt location
        cv::Point2f prev_point_outer = kptsPrev.at(prev_kpt_index).pt;
        cv::Point2f curr_point_outer = kptsCurr.at(curr_kpt_index).pt;

        for (auto match_inner : kptMatches)
        {
            int prev_kpt_index =  match_inner.queryIdx;
            int curr_kpt_index =  match_inner.trainIdx;

            //get outer prev and curr kpt location
            cv::Point2f prev_point_inner = kptsPrev.at(prev_kpt_index).pt;
            cv::Point2f curr_point_inner = kptsCurr.at(curr_kpt_index).pt;

            double dist_prev = cv::norm(prev_point_outer - prev_point_inner);
            double dist_curr = cv::norm(curr_point_outer - curr_point_inner);

            if (dist_prev > std::numeric_limits<double>::epsilon() && dist_curr >= 100.0)
            {
                double dist_ratio = dist_curr/ dist_prev;
                dist_ratios.push_back(dist_ratio);
                // std::cout << "dist ratio " << dist_ratio << std::endl;
            }
        }
    }

    double median_dist_ratio = get_median(dist_ratios);
    std::cout << "median ratio " << median_dist_ratio << std::endl;
    TTC = -dt/(1- median_dist_ratio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
   // equation to compute TTC:
   // TTC = min_x_curr * dt/ (min_x_prev - min_x_curr)

    double dt = 1/frameRate;
    double min_x_prev = std::numeric_limits<double>::max();
    double min_x_curr = std::numeric_limits<double>::max();

    std::vector<double> prev_points;
    std::vector<double> curr_points;

    //push points to vectot
    for (auto point : lidarPointsPrev)
    {
        prev_points.push_back(point.x);
    }

    for (auto point: lidarPointsCurr)
    {
        curr_points.push_back(point.x);
    }

    min_x_prev = get_median(prev_points);
    min_x_curr = get_median(curr_points);

    std::cout << "min_x_curr " << min_x_curr << std::endl;
    std::cout << "min_x_prev " << min_x_prev << std::endl;

    //compute TTC
    TTC = (min_x_curr * dt)/(min_x_prev - min_x_curr);

}
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    multimap<int, int> matched_boxes_accumulated;

    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        int prev_kpt_index =  it->queryIdx;
        int curr_kpt_index =   it->trainIdx;

        //get prev and curr kpt location
        cv::Point2f prev_point = prevFrame.keypoints.at(prev_kpt_index).pt;
        cv::Point2f curr_point = currFrame.keypoints.at(curr_kpt_index).pt;

        std::vector<int> prev_box_ids;
        std::vector<int> curr_box_ids;

        // loop over pev bounding box
        for (auto prev_box : prevFrame.boundingBoxes)
        {
            // if prev box contains point, store it
            if (prev_box.roi.contains(prev_point))
            {
                prev_box_ids.push_back(prev_box.boxID);
            }
        }

        //loop over curr bounding boxes
        for (auto curr_box : currFrame.boundingBoxes)
        {
            // if curr box contains point, store it
            if (curr_box.roi.contains(curr_point))
            {
                curr_box_ids.push_back(curr_box.boxID);
            }
        }

        // generate pairs between all stored prev and curr boxes
        for (auto prev_box_id : prev_box_ids)
        {
            for (auto curr_box_id : curr_box_ids)
            {
                matched_boxes_accumulated.insert({prev_box_id, curr_box_id});
            }
        }
    }

    // loop over all boxes of the prev frame
    for (int i = 0; i < prevFrame.boundingBoxes.size(); ++i)
    {
        // get a range of all curr boxes matched based on kpts above for this prev box
        auto it_pair = matched_boxes_accumulated.equal_range(i);

        vector<int> count;
        count.resize(currFrame.boundingBoxes.size(), 0);

        // count the number of matched curr boxes for each curr box
        for (auto it = it_pair.first; it != it_pair.second; ++it)
        {
            count.at((*it).second) = count.at((*it).second) + 1;
        }

        auto max_elem_it = std::max_element(count.begin(), count.end());
        int max_elem_box_id = std::distance(count.begin(), max_elem_it);

        //insert the best match candidate
        bbBestMatches.insert({i, max_elem_box_id});
    }
}
