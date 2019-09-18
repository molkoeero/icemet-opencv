#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

namespace cv { namespace icemet {

void adjust(const Mat& src, Mat& dst, uchar a0, uchar a1, uchar b0, uchar b1)
{
	CV_Assert(src.type() == CV_8UC1);
	if (dst.empty())
		dst = cv::Mat(src.size(), CV_8UC1);
	
	unsigned char *pSrc = (unsigned char*)src.data;
	unsigned char *pDst = (unsigned char*)dst.data;
	
	for(int y = 0; y < src.rows; y++) {
		for(int x = 0; x < src.cols; x++) {
			int i = src.step * y + src.channels() * x;
			float val = pSrc[i];
			val = val < a0 ? a0 : (val > a1 ? a1 : val);
			val -= a0;
			val /= a1 - a0;
			val *= b1 - b0;
			val += b0;
			pDst[i] = val;
		}
	}
}

void adjust(const UMat& src, UMat& dst, uchar a0, uchar a1, uchar b0, uchar b1)
{
	CV_Assert(src.type() == CV_8UC1);
	if (dst.empty())
		dst = cv::UMat(src.size(), CV_8UC1);
	
	size_t gsize[1] = {(size_t)(src.cols * src.rows)};
	ocl::Kernel("adjust", ocl::icemet::misc_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(src),
		ocl::KernelArg::PtrWriteOnly(dst),
		a0, a1, b0, b1
	).run(1, gsize, NULL, true);
}

void hist(const UMat& src, Mat& counts, Mat& bins, float min, float max, float step)
{
	int n = roundf((max-min) / step);
	UMat tmp = UMat::zeros(1, n, CV_32SC1);
	size_t gsize[1] = {(size_t)(src.cols * src.rows)};
	ocl::Kernel("hist", ocl::icemet::misc_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(src),
		ocl::KernelArg::PtrReadWrite(tmp),
		min, max, step
	).run(1, gsize, NULL, true);
	tmp.copyTo(counts);
	
	bins = Mat(1, n, CV_32FC1);
	for (int i = 0; i < n; i++)
		bins.at<float>(0, i) = min + i*step;
}

void imghist(const UMat& src, Mat& dst)
{
	UMat tmp = UMat::zeros(1, 256, CV_32SC1);
	size_t gsize[1] = {(size_t)(src.cols * src.rows)};
	ocl::Kernel("imghist", ocl::icemet::misc_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(src),
		ocl::KernelArg::PtrReadWrite(tmp)
	).run(1, gsize, NULL, true);
	tmp.copyTo(dst);
}

}}
