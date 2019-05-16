#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

#define STACK_LEN_MAX 101

namespace cv { namespace icemet {

class BGSubStackImpl : public BGSubStack {
private:
	Size2i m_size;
	int m_len;
	int m_idx;
	bool m_full;
	UMat m_stack;
	Mat m_means;

public:
	BGSubStackImpl(Size2i imgSize, int len) :
		m_size(imgSize),
		m_len(len),
		m_idx(0),
		m_full(false)
	{
		CV_Assert(len <= STACK_LEN_MAX);
		CV_Assert(len%2);
		m_stack = UMat(1, len * m_size.width * m_size.height, CV_8UC1);
		m_means = Mat(1, len, CV_32FC1);
	}
	
	bool push(const UMat& img)
	{
		size_t gsize[1] = {(size_t)(m_size.width * m_size.height)};
		ocl::Kernel("push", ocl::icemet::bgsub_oclsrc).args(
			ocl::KernelArg::PtrReadOnly(img),
			ocl::KernelArg::PtrWriteOnly(m_stack),
			m_size.width * m_size.height,
			m_idx
		).run(1, gsize, NULL, true);
		
		Scalar imgMean = mean(img);
		m_means.at<float>(0, m_idx) = imgMean[0];
		
		// Increment index
		m_idx = (m_idx+1) % m_len;
		if (m_idx == 0)
			m_full = true;
		return m_full;
	}
	
	void meddiv(UMat& dst)
	{
		size_t gsize[1] = {(size_t)(m_size.width * m_size.height)};
		ocl::Kernel("meddiv", ocl::icemet::bgsub_oclsrc).args(
			ocl::KernelArg::PtrReadOnly(m_stack),
			ocl::KernelArg::PtrReadOnly(m_means.getUMat(ACCESS_READ)),
			ocl::KernelArg::PtrWriteOnly(dst),
			m_size.width * m_size.height,
			m_len,
			(m_idx + m_len/2) % m_len
		).run(1, gsize, NULL, true);
	}
};

Ptr<BGSubStack> BGSubStack::create(Size2i imgSize, int len)
{
	return makePtr<BGSubStackImpl>(imgSize, len);
}

}}
