#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

uint16_t faceCounter = 1;
uint16_t eyeCounter = 1;

void saveCroppedImage(const cv::Mat& image, const std::string& prefix, uint16_t counter);

int main()
{
    // Load the pre-trained eye cascade classifier
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load("haarcascade_eye.xml");

    // Load the pre-trained frontalface classifier
    cv::CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_default.xml");

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Failed" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (cap.read(frame)) {
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayFrame, grayFrame);

        // Detect faces
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // Create a separate image for saving without drawing
        cv::Mat saveFrame = frame.clone();

        // Draw rectangles around the detected faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 0, 255), 2);
        }

        // Detect eyes
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(grayFrame, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // Draw circles around the detected eyes
        for (const auto& eye : eyes) {
            cv::Point center(eye.x + eye.width / 2, eye.y + eye.height / 2);
            cv::ellipse(frame, center, cv::Size(eye.width / 2, eye.height / 2), 0, 0, 360, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Face and Eye Detection", frame);

        
        uint8_t key = cv::waitKey(1);
        if (key == 27)
            break;
        else if (key == 'f') {
            
            cv::Mat faceROI = saveFrame(faces[0]).clone();
            saveCroppedImage(faceROI, "face", faceCounter);
            faceCounter++;
        }
        else if (key == 'e') {
            
            for (const auto& eye : eyes) {
                cv::Mat eyeROI = saveFrame(eye).clone();
                saveCroppedImage(eyeROI, "eye", eyeCounter);
                eyeCounter++;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void saveCroppedImage(const cv::Mat& image, const std::string& prefix, uint16_t counter)
{
    std::string filename = prefix + "_" + std::to_string(counter) + ".jpg";
    cv::imwrite(filename, image);
    std::cout << "Saved image: " << filename << std::endl;
}

