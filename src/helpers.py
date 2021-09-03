import cv2
import numpy as np


def BrightnessAndContrastAuto(im_source, clipHistPercent=0):
    histSize = 256
    minGray = 0
    maxGray = 0

    imgray = im_source.copy()
    if len(im_source.shape) == 3:
        imgray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)

    if clipHistPercent == 0:
        # keep full available range
        minGray, maxGray, min_pt, max_pt = cv2.minMaxLoc(imgray)
    else:
        histRange = [0, 256]
        uniform = True
        accumulate = False
        hist = cv2.calcHist([imgray], [0], None, [histSize], histRange, uniform, accumulate)
        print()
        accumulator = [0] * len(hist)
        accumulator[0] = hist[0]
        for i in range(1, histSize):
            accumulator[i] = accumulator[i - 1] + hist[i]

        # locate points that cuts at required value
        max = accumulator[-1]
        clipHistPercent *= (max / 100.0)
        clipHistPercent /= 2.0
        minGray = 0

        #  locate left cut
        while accumulator[minGray] < clipHistPercent:
            minGray += 1

        # locate right cut
        maxGray = histSize - 1
        while accumulator[maxGray] >= (max - clipHistPercent):
            maxGray -= 1

    # current range
    inputRange = maxGray - minGray
    alpha = (histSize - 1) / inputRange
    beta = -minGray * alpha

    dst = imgray.copy()
    cv2.convertScaleAbs(imgray, dst, alpha, beta)

    return dst


def MorphClose(im_source, size):
    # size = int(minThickess / 2)
    # size = 10
    anchor = (size, size)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1), anchor)

    dst = im_source.copy()
    cv2.morphologyEx(im_source, cv2.MORPH_CLOSE, element, dst, anchor)

    return dst


if __name__ == '__main__':
    image = cv2.imread('data/stone1.jpg', cv2.IMREAD_COLOR)
    dst = BrightnessAndContrastAuto(image, 1)
    # dst = MorphClose(image, 20)
    cv2.imshow("Result", dst)
    cv2.waitKey(0)

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

# void MorphClose(const Mat &imgIn,Mat &imgOut,int minThickess)
# {
#   int size = minThickess / 2;
#   Point anchor = Point(size, size);
#   Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * size + 1, 2 * size + 1), anchor);
#   morphologyEx(imgIn, imgOut, MORPH_CLOSE, element, anchor);
# }
