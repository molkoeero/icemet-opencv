#ifndef OPENCV_ICEMET_HPP
#define OPENCV_ICEMET_HPP

#include "opencv2/core.hpp"

#include <vector>

namespace cv { namespace icemet {

typedef enum _recon_output {
	RECON_OUTPUT_AMPLITUDE = 0,
	RECON_OUTPUT_PHASE,
	RECON_OUTPUT_COMPLEX
} ReconOutput;

typedef enum _focus_method {
	FOCUS_MIN = 0,
	FOCUS_MAX,
	FOCUS_RANGE,
	FOCUS_STD,
	FOCUS_TOG
} FocusMethod;

class CV_EXPORTS_W Hologram : public Algorithm {
public:
	CV_WRAP virtual void setImg(const UMat& img) = 0;
	CV_WRAP virtual void recon(UMat& dst, float z, ReconOutput output=RECON_OUTPUT_AMPLITUDE) = 0;
	CV_WRAP virtual void reconMin(std::vector<UMat>& dst, UMat& dstMin, float z0, float z1, float dz) = 0;
	
	CV_WRAP virtual float focus(float z0, float z1, float dz, FocusMethod method=FOCUS_STD, float K=3.0) = 0;
	
	CV_WRAP virtual void applyFilter(const UMat& H) = 0;
	CV_WRAP virtual UMat createLPF(float f) const = 0;
	
	CV_WRAP static float magnf(float dist, float z);
	
	CV_WRAP static void focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method=FOCUS_STD, int begin=0, int end=-1, float K=3.0);
	
	CV_WRAP static Ptr<Hologram> create(Size2i size, float psz, float lambda, float dist=0.0);
};

class CV_EXPORTS_W BGSubStack : public Algorithm {
public:
	CV_WRAP virtual bool push(const UMat& img) = 0;
	CV_WRAP virtual void meddiv(UMat& dst) = 0;
	
	CV_WRAP static Ptr<BGSubStack> create(Size2i imgSize, int len);
};

CV_EXPORTS_W void adjust(const UMat& src, UMat& dst, uchar a0, uchar a1, uchar b0, uchar b1);
CV_EXPORTS_W void hist(const UMat& src, Mat& counts, Mat& bins, float min, float max, float step);
CV_EXPORTS_W void imghist(const UMat& src, Mat& dst);

}}

#endif
