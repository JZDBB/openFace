#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2\opencv.hpp>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// Local includes
#include "LandmarkCoreIncludes.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
//	vector<string> arguments;

	LandmarkDetector::FaceModelParameters det_parameters;

	bool detection_success;
	Mat captured_image;
	Mat_<uchar> grayscale_image;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);
	string au_loc = "AU_predictors/AU_all_best.txt";
	string tri_loc = "model/tris_68_full.txt";
	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

	ifstream in_file("D:\\zzss\\database\\jaffe\\path.txt", ifstream::in);
	string file_name;
	string line;

	if (in_file.is_open())
	{
		while (getline(in_file,line))
		{
			file_name = line;
			captured_image = imread(file_name);

			if (captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
			}
			else
			{
				grayscale_image = captured_image.clone();
			}


			detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_model, det_parameters);

			Mat_<double> hog_descriptor;
			Mat sim_warped_img;
			
			face_analyser.AddNextFrame(captured_image, face_model, int(), false, !det_parameters.quiet_mode);
			face_analyser.GetLatestAlignedFace(sim_warped_img);

			int num_hog_rows;
			int num_hog_cols;
			FaceAnalysis::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);
			
			Mat_<double> hog_descriptor_vis;
			FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
			
			imshow("hog", hog_descriptor_vis);
			waitKey(1);

			ofstream out_file;
			string outfilename = file_name + ".csv";
			int k = hog_descriptor.cols;
			out_file.open(outfilename);
			for (int i = 0; i <= k; i++)
			{
				out_file << hog_descriptor(i) << ",";
			}
			
			out_file.close();

		}
	}
	
}