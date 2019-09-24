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

typedef enum _filter_type {
	FILTER_LOWPASS = 0,
	FILTER_HIGHPASS
} FilterType;

class CV_EXPORTS_W ZRange {
public:
	ZRange() : start(0), stop(0), step(0) {}
	ZRange(float start_, float stop_, float step_) : start(start_), stop(stop_), step(step_) {}
	
	int n();
	float z(int i);
	int i(float z);
	
	float start;
	float stop;
	float step;
};

class CV_EXPORTS_W Hologram : public Algorithm {
public:
	CV_WRAP virtual void setImg(const UMat& img) = 0;
	CV_WRAP virtual void recon(UMat& dst, float z, ReconOutput output=RECON_OUTPUT_AMPLITUDE) = 0;
	CV_WRAP virtual void reconMin(std::vector<UMat>& dst, UMat& dstMin, ZRange z) = 0;
	
	CV_WRAP virtual float focus(ZRange z, FocusMethod method=FOCUS_STD, float K=3.0) = 0;
	CV_WRAP virtual float focus(ZRange z, std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method=FOCUS_STD, float K=3.0) = 0;
	
	CV_WRAP virtual void applyFilter(const UMat& H) = 0;
	CV_WRAP virtual UMat createFilter(float f, FilterType type) const = 0;
	CV_WRAP virtual UMat createLPF(float f) const = 0;
	CV_WRAP virtual UMat createHPF(float f) const = 0;
	
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

CV_EXPORTS_W void adjust(const Mat& src, Mat& dst, uchar a0, uchar a1, uchar b0, uchar b1);
CV_EXPORTS_W void adjust(const UMat& src, UMat& dst, uchar a0, uchar a1, uchar b0, uchar b1);
CV_EXPORTS_W void hist(const UMat& src, Mat& counts, Mat& bins, float min, float max, float step);
CV_EXPORTS_W void imghist(const UMat& src, Mat& dst);

}}

#endif
