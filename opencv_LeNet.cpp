#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;


int main(int argc, char* argv[])
{
	try {
		dnn::Net net = dnn::readNet("lenet5.onnx");

		Mat img = imread("test/0.png", 0);
		Mat blob = dnn::blobFromImage(img, 1.0 / 255);
		net.setInput(blob);
		Mat prob = net.forward();
		Point classIdPoint;
		double confidence;
		minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;

		// Put efficiency information.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		std::cout << "class id: " << classId << ", time: " << label.c_str() << std::endl;
		std::cout << prob;
	}
	catch (const std::exception& e) {
		std::cout << e.what();
	}
	return 0;
}