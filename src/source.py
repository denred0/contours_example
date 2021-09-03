# #define CL_BLU      cv::Scalar(255, 0,      0   )
# #define CL_GREEN    cv::Scalar(0,   255,    0   )
# #define CL_RED      cv::Scalar(0,   0,      255 )
# #define CL_WHITE    cv::Scalar(255, 255,    255 )
# const string winName="Stones";
# Mat src,dst;
# int minSizeMM,thStone,thShadow,removeShadow;
# float pix2mm;
#
# //helper function
# void MorphClose(const Mat &imgIn,Mat &imgOut,int minThickess=2);
# //get the source [here](http://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/)
# void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0);
#
# // ON TRACK BAR FUNCTION
# void onStonesTb(int, void*)
# {
#   Mat blur,bwStones,bwShadow;
#   vector<vector<Point> > contours;
#   char buf[80];
#
#   cv::GaussianBlur(src,blur,Size(),2,2);
#
#   // convert to HSV
#   Mat src_hsv,brightness,saturation;
#   vector<Mat> hsv_planes;
#   cvtColor(blur, src_hsv, COLOR_BGR2HSV);
#   split(src_hsv, hsv_planes);
#   saturation = hsv_planes[1];
#   brightness = hsv_planes[2];
#
#   int minSizePx = cvRound(minSizeMM/pix2mm);
#   int closerSize = minSizePx /2.0;
#
#   //SELECT STONES (INCLUDING SHADOW)
#   MorphClose(saturation,saturation,closerSize);
#   BrightnessAndContrastAuto(saturation,saturation,1);
#   threshold(saturation,bwStones,thStone,255,THRESH_BINARY_INV); //Get 0..thStone
#   //show the selection
#   findContours(bwStones, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
#   for (int i = 0; i < contours.size(); i++)
#     cv::drawContours(saturation,contours,i,CL_WHITE,2);
#   imshow("Threshold Stones+Shadow on Saturation",saturation);
#
#   if(removeShadow)
#   {
#     //SELECT DARK AREAS (MAYBE SHADOWS)
#     MorphClose(brightness,brightness,closerSize);
#     BrightnessAndContrastAuto(brightness,brightness,1);
#     threshold(brightness,bwShadow,thShadow,255,THRESH_BINARY); //Get thShadow..255
#     //show the selection
#     contours.clear();
#     findContours(bwShadow, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
#     for (int i = 0; i < contours.size(); i++)
#       cv::drawContours(brightness,contours,i,CL_WHITE,2);
#     imshow("Threshold Shadow on Brightness",brightness);
#
#     //remove shadows from stones
#     cv::bitwise_and(bwStones,bwShadow,bwStones);
#   }
#
#   //show the result
#   src.copyTo(dst);
#   findContours(bwStones, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
#   Point2f centroid;
#   for (int i = 0; i < contours.size(); i++)
#   {
#
#     //draw the contour
#     polylines(dst, contours[i], true, CL_GREEN,2);
#     //get the bounding rect
#     cv::Rect rect = cv::boundingRect(contours[i]);
#     //ignore small objects
#     if ( max(rect.width,rect.width) < minSizePx)
#       continue;
#
#     //calculate moments
#     cv::Moments M = cv::moments(contours[i], false);
#     //reject if area is 0
#     double area = M.m00;
#     if (area <= 0)
#       continue;
#
#     // OK! THE STONE HAS BEEN SELECTED
#     cv::rectangle(dst,rect,CL_RED,2);
#     //get and draw the center
#     centroid.x = cvRound(M.m10 / M.m00);
#     centroid.y = cvRound(M.m01 / M.m00);
#     cv::circle(dst,centroid,3,CL_RED,1);
#     sprintf(buf,"%.0fx%.0fmm",rect.width*pix2mm,rect.height*pix2mm);
#     cv::putText(dst,buf,rect.tl(),CV_FONT_HERSHEY_PLAIN,1.2,CL_BLU,1,CV_AA);
#   }
#   imshow(winName,dst);
# }
#
# void Main_Stones()
# {
#   Mat orig = imread(_IMG_PATH_"stones.jpg");
#   int knowDistancePX = 170;
#   int knowDistanceMM = 150;
#   pix2mm = (float)knowDistancePX / knowDistanceMM;
#
#   //Scale down your because it's big... let's say to 50%
#   float scale = 0.5;
#   resize(orig,src,Size(),scale,scale);
#   pix2mm = pix2mm / scale;
#
#   //set defaults
#   minSizeMM=40;  // width in millimetre
#   thStone=100;    // max saturation for stones
#   thShadow=128;   // max brightness for shadow
#   removeShadow=1; // try to remove shadows (1=Yes 0=No)
#
#   imshow(winName,src);
#   createTrackbar("Size(mm)", winName, &minSizeMM,100, onStonesTb, 0);
#   createTrackbar("thStone", winName, &thStone, 255, onStonesTb, 0);
#   createTrackbar("thShadow", winName, &thShadow, 255, onStonesTb, 0);
#   createTrackbar("remove Shadow", winName, &removeShadow, 1, onStonesTb, 0);
#   onStonesTb(0,0);
#   waitKey(0);
# }
#
# void MorphClose(const Mat &imgIn,Mat &imgOut,int minThickess)
# {
#   int size = minThickess / 2;
#   Point anchor = Point(size, size);
#   Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * size + 1, 2 * size + 1), anchor);
#   morphologyEx(imgIn, imgOut, MORPH_CLOSE, element, anchor);
# }
#
# /**
#  *  \brief Automatic brightness and contrast optimization with optional histogram clipping
#  *  \param [in]src Input image GRAY or BGR or BGRA
#  *  \param [out]dst Destination image
#  *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
#  *  \note In case of BGRA image, we won't touch the transparency
# */
# void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
# {
#
#     CV_Assert(clipHistPercent >= 0);
#     CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));
#
#     int histSize = 256;
#     float alpha, beta;
#     double minGray = 0, maxGray = 0;
#
#     //to calculate grayscale histogram
#     cv::Mat gray;
#     if (src.type() == CV_8UC1) gray = src;
#     else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
#     else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
#     if (clipHistPercent == 0)
#     {
#         // keep full available range
#         cv::minMaxLoc(gray, &minGray, &maxGray);
#     }
#     else
#     {
#         cv::Mat hist; //the grayscale histogram
#
#         float range[] = { 0, 256 };
#         const float* histRange = { range };
#         bool uniform = true;
#         bool accumulate = false;
#         calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);
#
#         // calculate cumulative distribution from the histogram
#         std::vector<float> accumulator(histSize);
#         accumulator[0] = hist.at<float>(0);
#         for (int i = 1; i < histSize; i++)
#         {
#             accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
#         }
#
#         // locate points that cuts at required value
#         float max = accumulator.back();
#         clipHistPercent *= (max / 100.0); //make percent as absolute
#         clipHistPercent /= 2.0; // left and right wings
#         // locate left cut
#         minGray = 0;
#         while (accumulator[minGray] < clipHistPercent)
#             minGray++;
#
#         // locate right cut
#         maxGray = histSize - 1;
#         while (accumulator[maxGray] >= (max - clipHistPercent))
#             maxGray--;
#     }
#
#     // current range
#     float inputRange = maxGray - minGray;
#
#     alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
#     beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0
#
#     // Apply brightness and contrast normalization
#     // convertTo operates with saurate_cast
#     src.convertTo(dst, -1, alpha, beta);
#
#     // restore alpha channel from source
#     if (dst.type() == CV_8UC4)
#     {
#         int from_to[] = { 3, 3};
#         cv::mixChannels(&src, 4, &dst,1, from_to, 1);
#     }
#     return;
# }