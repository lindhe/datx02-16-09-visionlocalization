/*********************************************************************
 * This file is distributed as part of the C++ port of the APRIL tags
 * library. The code is licensed under GPLv2.
 *
 * Original author: Edwin Olson <ebolson@umich.edu>
 * C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>
 * ----------------------- Modified ---------------------------------e
 * Code modified for project in Vision Based Localization for
 * Autonomous Vehicles at Chalmers University, Goteborg, Sweden
 * Modification Authors:
 * Copyright (c) 2013-2014 Andrew Soderberg-Rivkin <sandrew@student.chalmers.se>
 * Copyright (c) 2013-2014 Sanjana Hangal <sanjana@student.chalmers.se>
 * Copyright (c) 2014 Thomas Petig <petig@chalmers.se>
 * Copyright (c) 2016 Robert Gustafsson <robg@student.chalmers.se>
 ********************************************************************/

#include "TagDetector.h"
#include "OpenCVHelper.h"

#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#include <getopt.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <signal.h>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/array.hpp>
#include <boost/asio.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"

#include <boost/chrono/chrono.hpp>
#include <boost/chrono/chrono_io.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>
#include <boost/chrono/ceil.hpp>
#include <boost/chrono/floor.hpp>
#include <boost/chrono/round.hpp>
#include "ueye.h"
#include "CameraUtil.h"

#define PI 3.14159265
#define DEFAULT_TAG_FAMILY "Tag16h5"
using namespace std;
using helper::ImageSource;
using boost::asio::ip::udp;
using boost::posix_time::ptime;
using boost::posix_time::time_duration;
//using namespace boost::asio;

sig_atomic_t sig_stop = 0;

void signal_handler (int param)
{
      sig_stop = 1;
}

double a1 = 0.0;
double a2 = 0.0;
double b1 = 0.0;
double b2 = 0.0;
double c1 = 0.0;
double c2 = 0.0;
double d1 = 0.0;
double d2 = 0.0;
double dir_x = 0;
double dir_y = 0;

at::Mat pts;

typedef struct GulliViewOptions {
  GulliViewOptions() :
      params(),
      family_str(DEFAULT_TAG_FAMILY),
      error_fraction(1),
      device_num(0),
      focal_length(906),
      tag_size(0.133),
      frame_width(0),
      frame_height(0),
      /* Changed to False so that text comes out correctly. */
      /* Issues with detection when set to False */
      mirror_display(false),
      no_gui(false),
      ueye(false),
      offset_x(0),
      offset_y(0),
      timestamp(false),
      period(false),
      debuginfo(false)
  {
  }
  TagDetectorParams params;
  std::string family_str;
  double error_fraction;
  int device_num;
  double focal_length;
  double tag_size;
  int frame_width;
  int frame_height;
  bool mirror_display;
  bool no_gui;
  bool ueye;
  int offset_x;
  int offset_y;
  bool timestamp;
  bool period;
  bool debuginfo;
} GulliViewOptions;


void print_usage(const char* tool_name, FILE* output=stderr) {

  TagDetectorParams p;
  GulliViewOptions o;

  fprintf(output, "\
Usage: %s [OPTIONS]\n\
GulliView Program used for tag detection on Autonomous Vehicles. Options:\n\
 -h              Show this help message.\n\
 -f FAMILY       Look for the given tag family (default \"%s\")\n\
 -d DEVICE       Set camera device number (default %d)\n\
 -z SIZE         Set the tag size in meters (default %f)\n\
 -W WIDTH        Set the camera image width in pixels\n\
 -H HEIGHT       Set the camera image height in pixels\n\
 -M              Toggle display mirroring\n\
 -n              No gui\n\
 -u              Ueye camera\n\
 -x LENGTH       Offset x-axis in millimeters\n\
 -y LENGTH       Offset y-axis in millimeters\n\
 -t              Display timestamp\n\
 -p              Display period since last detection\n\
 -i              Display debug information when terminate\n",
          tool_name,
      /* Options removed that are not needed */
      /* Can be added later for further functionality */
          //p.sigma,
          //p.segSigma,
          //p.thetaThresh,
          //p.magThresh,
          //p.adaptiveThresholdValue,
          //p.adaptiveThresholdRadius,
          DEFAULT_TAG_FAMILY,
          //o.error_fraction,
          o.device_num,
          //o.focal_length,
          o.tag_size);


  fprintf(output, "Known tag families:");
  TagFamily::StringArray known = TagFamily::families();
  for (size_t i = 0; i < known.size(); ++i) {
    fprintf(output, " %s", known[i].c_str());
  }
  fprintf(output, "\n");
  /* Old Options removed can be re-added if they are needed. Default values set for now:
   * -D              Use decimation for segmentation stage.\n\
   * -S SIGMA        Set the original image sigma value (default %.2f).\n\
   * -s SEGSIGMA     Set the segmentation sigma value (default %.2f).\n\
   * -a THETATHRESH  Set the theta threshold for clustering (default %.1f).\n\
   * -m MAGTHRESH    Set the magnitude threshold for clustering (default %.1f).\n\
   * -V VALUE        Set adaptive threshold value for new quad algo (default %f).\n\
   * -N RADIUS       Set adaptive threshold radius for new quad algo (default %d).\n\
   * -b              Refine bad quads using template tracker.\n\
   * -r              Refine all quads using template tracker.\n\
   * -n              Use the new quad detection algorithm.\n\
   * -e FRACTION     Set error detection fraction (default %f)\n\
   * -F FLENGTH      Set the camera's focal length in pixels (default %f)\n\
   */
}

GulliViewOptions parse_options(int argc, char** argv) {
  GulliViewOptions opts;
  const char* options_str = "hDS:s:a:m:V:N:brnf:e:d:F:z:W:H:M:utpix:y:";
  int c;
  while ((c = getopt(argc, argv, options_str)) != -1) {
    switch (c) {
      // Reminder: add new options to 'options_str' above and print_usage()!
      case 'h': print_usage(argv[0], stdout); exit(0); break;
      //case 'D': opts.params.segDecimate = true; break;
      //case 'S': opts.params.sigma = atof(optarg); break;
      //case 's': opts.params.segSigma = atof(optarg); break;
      //case 'a': opts.params.thetaThresh = atof(optarg); break;
      //case 'm': opts.params.magThresh = atof(optarg); break;
      //case 'V': opts.params.adaptiveThresholdValue = atof(optarg); break;
      //case 'N': opts.params.adaptiveThresholdRadius = atoi(optarg); break;
      //case 'b': opts.params.refineBad = true; break;
      //case 'r': opts.params.refineQuads = true; break;
      //case 'n': opts.params.newQuadAlgorithm = true; break;
      case 'f': opts.family_str = optarg; break;
      //case 'e': opts.error_fraction = atof(optarg); break;
      case 'd': opts.device_num = atoi(optarg); break;
      //case 'F': opts.focal_length = atof(optarg); break;
      case 'z': opts.tag_size = atof(optarg); break;
      case 'W': opts.frame_width = atoi(optarg); break;
      case 'H': opts.frame_height = atoi(optarg); break;
      case 'M': opts.mirror_display = !opts.mirror_display; break;
      case 'n': opts.no_gui = 1; break;
      case 'u': opts.ueye = true; break;
      case 'x': opts.offset_x = atoi(optarg); break;
      case 'y': opts.offset_y = atoi(optarg); break;
      case 't': opts.timestamp = true; break;
      case 'p': opts.period = true; break;
      case 'i': opts.debuginfo = true; break;
      default:
        fprintf(stderr, "\n");
        print_usage(argv[0], stderr);
        exit(1);
    }
  }
  opts.params.adaptiveThresholdRadius += (opts.params.adaptiveThresholdRadius+1) % 2;
  return opts;
}

cv::Mat OpenWarpPerspective(const cv::Mat& _image
      , const cv::Point& _lu
      , const cv::Point& _ru
      , const cv::Point& _rd
      , const cv::Point& _ld
      , const cv::Point& _lu_result
      , const cv::Point& _ru_result
      , const cv::Point& _rd_result
      , const cv::Point& _ld_result
      , cv::Mat& _transform_matrix)
    {
      // todo do some checks on input.

      cv::Point2f source_points[4];
      cv::Point2f dest_points[4];

#pragma omp parallel sections
{
    { source_points[0] = _lu; }
#pragma omp section
    { source_points[1] = _ru; }
#pragma omp section
    { source_points[2] = _rd; }
#pragma omp section
    { source_points[3] = _ld; }
#pragma omp section

    { dest_points[0] = _lu_result; }
#pragma omp section
    { dest_points[1] = _ru_result; }
#pragma omp section
    { dest_points[2] = _rd_result; }
#pragma omp section
    { dest_points[3] = _ld_result; }
}

      cv::Mat dst;
      _transform_matrix = cv::getPerspectiveTransform(source_points, dest_points);
      cv::warpPerspective(_image, dst, _transform_matrix, cv::Size(CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT));

      return dst;
}

time_duration totalPeriod = boost::posix_time::microseconds(0);

int main(int argc, char** argv) {
    //doing gracefull shutdown, prevents Linux USB system to crash
    signal (SIGINT, signal_handler);

   cv::VideoCapture vc;
   GulliViewOptions opts = parse_options(argc, argv);
   TagFamily family(opts.family_str);

   //init variables for ueye camera
   char* ppcImgMem;
   INT id;
   HIDS hCam = 0;
   HIDS *hCamPtr = &hCam;
   INT nRet;
   VOID * pMem;
   double newFPS;

   //Buffer to hold tags and coordinates
   //char* buffer = new char[100];
   if(opts.ueye){
       nRet = is_InitCamera(hCamPtr, NULL);
       if(nRet == IS_SUCCESS){
           cout << "init camera success" << endl;
       }
       nRet = is_SetColorMode(*hCamPtr, IS_CM_MONO8);

//       nRet = is_SetExternalTrigger(*hCamPtr, IS_SET_TRIGGER_SOFTWARE);
       if(nRet == IS_SUCCESS){
           cout << "init colormode success" << endl;
       }
       }else{

           vc.open(opts.device_num);

           if (opts.frame_width && opts.frame_height) {

      // Use uvcdynctrl to figure this out dynamically at some point?
              vc.set(CV_CAP_PROP_FRAME_WIDTH, opts.frame_width);
              vc.set(CV_CAP_PROP_FRAME_HEIGHT, opts.frame_height);


           }
     }

   if (opts.error_fraction >= 1 && opts.error_fraction <= 1) {
      family.setErrorRecoveryFraction(opts.error_fraction);
   }

   //std::cout << "family.minimumHammingDistance = " << family.minimumHammingDistance << "\n";
   //std::cout << "family.errorRecoveryBits = " << family.errorRecoveryBits << "\n";



  /* std::cout << "Set camera to resolution: "
      << vc.get(CV_CAP_PROP_FRAME_WIDTH) << "x"
      << vc.get(CV_CAP_PROP_FRAME_HEIGHT) << "\n";
  */
   //init ueye camera
   nRet = is_AllocImageMem (*hCamPtr, 1280, 1024, 8, &ppcImgMem, &id);
   nRet = is_SetImageMem(hCam, ppcImgMem, id);
   nRet = is_SetFrameRate(*hCamPtr, 60.0, &newFPS);
   nRet = is_CaptureVideo(*hCamPtr, IS_DONT_WAIT);
   nRet = is_GetImageMem(*hCamPtr, &pMem);
   UINT nPixelClock;

   // Get current pixel clock
   nRet = is_PixelClock(hCam, IS_PIXELCLOCK_CMD_GET, (void*)&nPixelClock, sizeof(nPixelClock));
   //cout << "This is the currernt pixel clock: " << nPixelClock << endl;
   cv::Mat frame(1024, 1280, CV_8UC1, (uchar *) pMem);
   cv::Point2d opticalCenter;
   /*
   vc >> frame;
   if (frame.empty()) {
      std::cerr << "no frames!\n";
      exit(1);
   }
   */
   /* Optical Center of video capturing frame with X and Y coordinates */
   opticalCenter.x = frame.cols * 0.5;
   opticalCenter.y = frame.rows * 0.5;

   std::string win = "GulliViewer";

   TagDetectorParams& params = opts.params;
   TagDetector detector(family, params);

   TagDetectionArray detections;
   TagDetectionArray source_points;
   TagDetectionArray dest_points;
   TagDetectionArray dst;
   TagDetectionArray newDetections;

   int cvPose = 0;

   boost::asio::io_service io_service;
   udp::resolver resolver(io_service);
   udp::resolver::query query(udp::v4(), "10.0.0.100", "daytime");
   udp::endpoint receiver_endpoint = *resolver.resolve(query);

   udp::socket socket(io_service);
   socket.open(udp::v4());
   uint32_t seq = 0;

   ptime start = boost::posix_time::microsec_clock::local_time();
   ptime periodStart = boost::posix_time::microsec_clock::local_time();

   int periodCount = 0;
   bool firstDetection = true;

   while (1) {

      if(opts.ueye){
         nRet = is_GetImageMem(*hCamPtr, &pMem);
       /*
      if(nRet == IS_BAD_STRUCTURE_SIZE){
          cout << "Bad structure" << endl;
      }else if(nRet == IS_CANT_COMMUNICATE_WITH_DRIVER){
          cout << "cant communciat with driver" << endl;
      }else if(nRet == IS_CANT_OPEN_DEVICE){
          cout << "cant open device" << endl;
      }else if(nRet == IS_CAPTURE_RUNNING){
          cout << "capture running" << endl;
      }else if(nRet == IS_INVALID_BUFFER_SIZE){
          cout << "invalid bufer size" << endl;
      }else if(nRet == IS_INVALID_CAMERA_TYPE){
          cout << "invalid camera type" << endl;
      }else if(nRet == IS_INVALID_CAMERA_HANDLE){
          cout << "invalid camera handel" << endl;
      }else if(nRet == IS_INVALID_MEMORY_POINTER){
          cout << "invalid mem ptr" << endl;
      }else if(nRet == IS_INVALID_PARAMETER){
          cout << "invalid parms" << endl;
      }else if(nRet == IS_IO_REQUEST_FAILED){
          cout << "io req failed" << endl;
      }else if(nRet == IS_NO_ACTIVE_IMG_MEM){
          cout << "no active img mem" << endl;
      }else if(nRet == IS_NOT_CALIBRATED){
          cout << "not calibrated" << endl;
      }else if(nRet == IS_NOT_SUPPORTED){
          cout << "not supported" << endl;
      }else if(nRet == IS_OUT_OF_MEMORY){
          cout << "out of mem" << endl;
      }

     double frameRate;
          if(nRet == IS_SUCCESS){
              cout << "Camera getImage success" << endl;
          }
          is_GetFramesPerSecond (hCam, &frameRate);
          cout << "The framerate is: " << frameRate << endl;
          */
          memcpy(frame.ptr(), pMem, frame.cols * frame.rows);
      } else {
          vc >> frame;
      }

      //std::cout << "Start Time " << start << "\n";
      //std::string startProcStr = helper::num2str(boost::posix_time::microsec_clock::local_time());
      if (frame.empty()) {
         break;
      }

      detector.process(frame, opticalCenter, detections);


      cv::Mat show;
      if (detections.empty()) {

         show = frame;
         string idToText = "---Nothing Detected---";
         putText(frame, idToText,
               cvPoint(30,30),
               cv::FONT_HERSHEY_PLAIN,
               1.5, cvScalar(180,250,0), 1, CV_AA);

      } else  {

         // Get time of frame/detection----------------
         //show = family.superimposeDetections(frame, detections); //-- Used to actually
         //superimpose tag image in video
         show = frame;

         double s = opts.tag_size;
         double ss = 0.5*s;
         //The height of the object
         double sz = -0.23;
         enum { npoints = 6, nedges = 5 };
         cv::Point3d src[npoints] = {
            cv::Point3d(-ss, -ss, 0),
            cv::Point3d(ss,  -ss, 0),
            cv::Point3d(ss,   ss, 0),
            cv::Point3d(-ss,  ss, 0),
            cv::Point3d(0,     0, 0),
            cv::Point3d(0,     0, sz),
         };

         /* Possible edges of the box created. Come back to THIS*/
         int edges[nedges][2] = {

            { 0, 1 },
            { 1, 2 },
            { 2, 3 },
            { 3, 0 },
            { 4, 5 }
         };

         cv::Point2d dst[npoints];

         double f = opts.focal_length;
         /* Optical centers, possible 2D config*/
         double K[9] = {
            f, 0, opticalCenter.x,
            0, f, opticalCenter.y,
            0, 0, 1
         };

         cv::Mat_<cv::Point3d> srcmat(npoints, 1, src);
         cv::Mat_<cv::Point2d> dstmat(npoints, 1, dst);

         cv::Mat_<double>      Kmat(3, 3, K);

         cv::Mat_<double>      distCoeffs = cv::Mat_<double>::zeros(4,1);

         boost::array<uint8_t, 256> recv_buf;
         size_t index = 0;

         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 1; //type
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 2; //subtype
         recv_buf[index++] = seq >> 24;
         recv_buf[index++] = seq >> 16;
         recv_buf[index++] = seq >> 8;
         recv_buf[index++] = seq;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0; // todo: age [s]
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0;
         recv_buf[index++] = 0; // todo: age [us]
         size_t len = 0;

         size_t len_index = index;
         index += 4;

         for (size_t i=0; i<detections.size(); ++i) {
            //Add code in order to copy and send array
            //Static buffer
            TagDetection &dd = detections[i];
            // Origin of axis detected
            if (dd.id == 0) {
               putText(frame, "0,0",
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,0,250), 2, CV_AA);
               a1 = dd.cxy.x;
               a2 = dd.cxy.y;
               // New X-Axis detected
            } else if (dd.id == 1) {
               putText(frame, "X Axis",
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,0,250), 2, CV_AA);
               b1 = dd.cxy.x;
               b2 = dd.cxy.y;
               //            b1 = b1-a1;
               //            b2 = b2-a2;
               // New Y-Axis detected
            } else if (dd.id == 2) {
               putText(frame, "Y Axis",
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,0,250), 2, CV_AA);
               c1 = dd.cxy.x;
               c2 = dd.cxy.y;
               //            c1 = c1-a1;
               //            c2 = c2-a2;
               // Quad Angle used for perspective transform
            } else if (dd.id == 3) {
               putText(frame, "Quad Axis",
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,0,250), 2, CV_AA);
               d1 = dd.cxy.x;
               d2 = dd.cxy.y;
               //            d1 = d1-a1;
               //            d2 = d2-a2;
            }
         }

         //cv::Mat edited;
         //cv::Mat dist;
         //cv::namedWindow( "Display window", CV_WINDOW_AUTOSIZE );

         //cv::imshow( "Display window" , edited );
         // Other ID's and coordinates detected
         //double det = 1.0/(b1*c2-c1*b2);
         //std::cout<<"1/det "<<det<<"\n";
         //double f1 = det*c2;
         //double f2 = det*(-c1);
         //double f3 = det*(-b2);
         //double f4 = det*b1;

         at::Point source_points[4];
         at::Point dest_points[4];


         source_points[0] = at::Point(a1, a2);
         source_points[1] = at::Point(b1, b2);
         source_points[2] = at::Point(d1, d2);
         source_points[3] = at::Point(c1, c2);

         //std::cout << "One: " << one << "\n";

         dest_points[0] =  at::Point(0.0, 0.0);
         dest_points[1] =  at::Point(1.0, 0.0);
         dest_points[2] =  at::Point(1.0, 1.0);
         dest_points[3] =  at::Point(0.0, 1.0);

         pts = getPerspectiveTransform(source_points, dest_points);
         //std::cout<<"PTS: " << pts << "\n";

         std::vector<at::Point>  prevDetections(detections.size());
#pragma omp parallel for
         for (size_t i=0; i<detections.size(); ++i) {
            TagDetection &dd = detections[i];
            prevDetections[i] = at::Point(dd.cxy.x, dd.cxy.y);
         }

         std::vector<at::Point>  newDetections(detections.size());
         perspectiveTransform(prevDetections, newDetections, pts);

#pragma omp parallel for
         for (size_t i=0; i<detections.size(); ++i) {
            TagDetection &dd = detections[i];

               /* Used to draw lines on video image */
               int lines = dd.id==5? nedges:(nedges-1);
               for (int j=0; j<lines; ++j) {
                  cv::line(show,
                        dstmat(edges[j][0],0),
                        dstmat(edges[j][1],0),
                        cvPose ? CV_RGB(0,0,255) : CV_RGB(255,0,0),
                        1, CV_AA);

               }
            if(dd.id == 4){
                dir_x = newDetections[i].x;
                dir_y = newDetections[i].y;
               // Print out Tag ID in center of Tag
               putText(frame, helper::num2str(dd.id),
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,250,0), 2, CV_AA);
            }
            if (dd.id == 5) {
               //boost::chrono::nanoseconds start;

               cv::Mat r, t;
               //Get rotation and translation vector
               if (cvPose) {


                  CameraUtil::homographyToPoseCV(f, f, s,
                        detections[i].homography,
                        r, t);

               } else {

                  cv::Mat_<double> M =
                     CameraUtil::homographyToPose(f, f, s,
                           detections[i].homography,
                           false);

                  cv::Mat_<double> R = M.rowRange(0,3).colRange(0, 3);

                  t = M.rowRange(0,3).col(3);

                  cv::Rodrigues(R, r);

               }

        double direction;

#pragma omp parallel sections
    {
        {
                //Get projection vector
                cv::projectPoints(srcmat, r, t, Kmat, distCoeffs, dstmat);
        }
#pragma omp section
        {
                //Calculate angle
                std::vector<at::Point>  frontDirection(1);
                std::vector<double>  refDirection(2);
                refDirection[0] = 1;
                refDirection[1] = 0;
                frontDirection[0] = at::Point(newDetections[i].x-dir_x, newDetections[i].y-dir_y);
                double length = sqrt(pow(frontDirection[0].x,2)+pow(frontDirection[0].y,2));
                std::vector<double> point(2);
                point[0] = frontDirection[0].x/length;
                point[1] = frontDirection[0].y/length;
                double dotproduct = std::inner_product(point.begin(), point.end(),refDirection.begin(), 0.0);
                direction = acos(dotproduct) * 180.0 / PI;

                if(newDetections[i].y < dir_y){
                    direction = 360.0-direction;
                }
        }
    }

                //Get coordinates from projection vector and project on the plane
                std::vector<at::Point>  prevPointDetections(1);
                prevPointDetections[0] = at::Point(dstmat[npoints-1]->x, dstmat[npoints-1]->y);
                std::vector<at::Point>  newPointDetections(1);
                perspectiveTransform(prevPointDetections, newPointDetections, pts);

               double x_new = newPointDetections[0].x;//newDetections[i].x;
               double y_new = newPointDetections[0].y;//newDetections[i].y;

#if 0
               double x_new = f1*(dd.cxy.x-a1) + f2*(dd.cxy.y-a2);
               double y_new = f3*(dd.cxy.x-a1) + f4*(dd.cxy.y-a2);
#endif

               //-------- Start of Perspcetive Transform TODO-------//

               //cv::Mat dst;
               //cv::warpPerspective(frame, dst, pts, cv::Size(640, 480));
               //---perspectiveTransform(detections, newDetections, pts);
               //dist = OpenWarpPerspective(frame,one,two,three,four,five,six,seven,eight,edited);
               //std::cout << "EDITED: " << edited << "\n";
               //TagDetection &ddn = newDetections[i];
               //std::cout << "New Coordinates X?: " << ddn.cxy.x  << "\n";
               //-----------------End Perspective Transform TODO----------//

               // Print out Tag ID in center of Tag
               putText(frame, helper::num2str(dd.id),
                     cv::Point(dd.cxy.x,dd.cxy.y),
                     CV_FONT_NORMAL,
                     1.0, cvScalar(0,250,0), 2, CV_AA);

               //TODO:Processing time
               //boost::chrono::nanoseconds end;
               //boost::chrono::nanoseconds count;
               //count = end - start;

               // d now holds the number of milliseconds from start to end.

               //std::cout<< count.count()<< "\n";
               //End timestamp (Processing)
               //std::string endProcStr = helper::num2str(boost::posix_time::microsec_clock::local_time());
               //std::string totProcStr = endProcStr-startProcStr;
               //std::cout <<"Processing time: " << totProcStr << "\n";
               ptime end;
               end = boost::posix_time::microsec_clock::local_time();
               //std::cout << "End Time " << end << "\n";
               time_duration processTime = end-start;
               time_duration periodTime = end-periodStart;
               //difftime(end,start);
               std::string procTim = helper::num2str(processTime);
               std::string perTim = helper::num2str(periodTime);
               /* std::cout << "Elapsed Time: " << procTim << "\n"; */

               uint32_t id = dd.id - 3;
               recv_buf[index++] = id >> 24;
               recv_buf[index++] = id >> 16;
               recv_buf[index++] = id >> 8;
               recv_buf[index++] = id;
               int32_t x_coord = (int32_t)((x_new * 1000.0) + opts.offset_x);
               recv_buf[index++] = x_coord >> 24;
               recv_buf[index++] = x_coord >> 16;
               recv_buf[index++] = x_coord >> 8;
               recv_buf[index++] = x_coord;
               int32_t y_coord = (int32_t)((y_new * 1000.0) + opts.offset_y);
               recv_buf[index++] = y_coord >> 24;
               recv_buf[index++] = y_coord >> 16;
               recv_buf[index++] = y_coord >> 8;
               recv_buf[index++] = y_coord;
               int32_t heading   = (int32_t) direction;
               recv_buf[index++] = heading >> 24;
               recv_buf[index++] = heading >> 16;
               recv_buf[index++] = heading >> 8;
               recv_buf[index++] = heading;
               ++len;
               string timestamp, period = "";
               opts.timestamp ? (timestamp = "\t" + procTim):"";
               opts.period ? (period = "\t" + perTim):"";
               std::cout << dd.id << "\t" << x_coord << "\t" << y_coord << "\t"
                  << heading << timestamp << period << std::endl;

               //       std::string outPut = "Tag ID: " + helper::num2str(dd.id) + " Coordinates: "
               //       + helper::num2str(x_new) + ", " + helper::num2str(y_new) + " Time: " + helper::num2str(boost::posix_time::microsec_clock::local_time()) + " [" +  helper::num2str(start) + "]";

               //std::cout<<"Output: " << outPut <<"\n";
               //std::string startTime =;
               //std::cout<<receiver_endpoint<<"\n";
               //socket.send_to(boost::asio::buffer(startTime),
               //receiver_endpoint);
               //Print out detections and full packet to be sent to server
               //std::cout << outPut << "\n";

               //Change coordinates to int, lose the extra decimal places
               //std::cout << "---Coordinates X---: " << x_new << "\n";
               //std::cout << "---Coordinates Y---: " << y_new << "\n";
               //Get the time of full processing/timestamp for packet

               if (!firstDetection) {
                  totalPeriod += periodTime;
               }
               /* This period reset has an accuracy of about 1 ms, compared to
                * the timestamp */
               periodStart = boost::posix_time::microsec_clock::local_time();
               periodCount++;
               firstDetection = false;
            }

            //std::cout << newOrgX << "\n";


         index = len_index;
         recv_buf[index++] = len << 24;
         recv_buf[index++] = len << 16;
         recv_buf[index++] = len << 8;
         recv_buf[index++] = len;
         try {
           socket.send_to(boost::asio::buffer(recv_buf), receiver_endpoint);
           ++seq;
         } catch (boost::system::system_error const& e) {
           if (opts.debuginfo) {
              std::cout << "Warning: " << e.what() << std::endl;
           }
         }

         }

      }

      if (opts.mirror_display) {
         cv::flip(show, show, 1);
      }

      if (not opts.no_gui) {
          cv::imshow(win, show);
      }
      int k = cv::waitKey(5);
      if (k % 256 == 's') {
         cv::imwrite("frame.png", frame);
         std::cout << "wrote frame.png\n";
      } else if (k % 256 == 'p') {
         cvPose = !cvPose;
      } else if ((k % 256 == 27) or sig_stop/* ESC */) {
         break;
      }

   }

   /* averagePeriod skips the first value (through use of firstDetection), since
    * it will always throw off the actual mean*/
   std::string totPeriod = helper::num2str(totalPeriod);
   long averagePeriod = totalPeriod.total_microseconds()/periodCount;
   time_duration avgPerTmp = boost::posix_time::microseconds(averagePeriod);
   std::string avgPer = helper::num2str(avgPerTmp);


   /* Report times of position? */
   if(opts.debuginfo){
        detector.reportTimers();
        std::cout << "Avg. period: " << avgPer << std::endl;
   }

   if(opts.ueye){
       nRet = is_FreeImageMem (*hCamPtr, ppcImgMem, id);
       nRet = is_ExitCamera(*hCamPtr);
   }

   return 0;

}
