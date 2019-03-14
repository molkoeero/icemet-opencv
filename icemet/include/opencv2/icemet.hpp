#ifndef OPENCV_ICEMET_HPP
#define OPENCV_ICEMET_HPP

#include "opencv2/core.hpp"

#include <vector>

namespace cv { namespace icemet {

typedef enum _focus_method {
	FOCUS_MIN = 0,
	FOCUS_STD
} FocusMethod;

class CV_EXPORTS_W Hologram : public Algorithm {
public:
	CV_WRAP virtual void setImg(const UMat& img) = 0;
	CV_WRAP virtual void recon(UMat& dst, float z) = 0;
	CV_WRAP virtual void recon(std::vector<UMat>& dst, UMat& dstMin, float z0, float z1, float dz) = 0;
	
	CV_WRAP virtual void applyFilter(const UMat& H) = 0;
	CV_WRAP virtual cv::UMat createLPF(float f) const = 0;
	
	CV_WRAP static void focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method=FOCUS_STD, int n=-1);
	CV_WRAP static Ptr<Hologram> create(Size2i size, float psz, float dist, float lambda);
};

class CV_EXPORTS_W BGSubStack : public Algorithm {
public:
	CV_WRAP virtual bool push(const UMat& img) = 0;
	CV_WRAP virtual void meddiv(UMat& dst) = 0;
	
	CV_WRAP static Ptr<BGSubStack> create(Size2i imgSize, int len);
};

CV_EXPORTS_W void adjust(const cv::UMat& src, cv::UMat& dst, uchar a0, uchar a1, uchar b0, uchar b1);
CV_EXPORTS_W void hist(const cv::UMat& src, cv::Mat& counts, cv::Mat& bins, float min, float max, float stepp);
CV_EXPORTS_W void imghist(const cv::UMat& src, cv::Mat& dst);

}}

#endif
