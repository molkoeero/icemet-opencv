#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

namespace cv { namespace icemet {

class HologramImpl : public Hologram {
private:
	Size2i m_sizeOrig;
	Size2i m_sizePad;
	
	float m_psz;
	float m_dist;
	float m_lambda;
	
	UMat m_prop;
	UMat m_dft;
	UMat m_complex;

private:
	inline float Mz(float z)
	{
		float M = m_dist / (m_dist - z);
		return M * z;
	}

public:
	HologramImpl(Size2i size, float psz, float dist, float lambda) :
		m_sizeOrig(size),
		m_psz(psz),
		m_dist(dist),
		m_lambda(lambda)
	{
		
		// TODO: Fix padding
		//m_sizePad = Size2i(getOptimalDFTSize(size.width), getOptimalDFTSize(size.height));
		m_sizePad = size;
		
		// Allocate UMats
		m_prop = UMat(m_sizePad, CV_32FC2);
		m_dft = UMat(m_sizePad, CV_32FC2);
		m_complex = UMat(m_sizePad, CV_32FC2);
		
		// Fill propagator
		size_t gsize[2] = {m_sizePad.width, m_sizePad.height};
		ocl::Kernel("angularspectrum", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::WriteOnly(m_prop),
			Vec2f(psz*m_sizePad.width, psz*m_sizePad.height),
			m_lambda
		).run(2, gsize, NULL, true);
	}
	
	void set(const UMat& img) CV_OVERRIDE
	{
		size_t gsize[2] = {m_sizePad.width, m_sizePad.height};
		
		// Convert to complex
		ocl::Kernel("r2c", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::ReadOnly(img),
			ocl::KernelArg::WriteOnly(m_dft)
		).run(2, gsize, NULL, true);
		
		// FFT
		dft(m_dft, m_dft, DFT_COMPLEX_INPUT|DFT_COMPLEX_OUTPUT|DFT_SCALE);
	}
	
	void recon(UMat& dst, float z) CV_OVERRIDE
	{
		size_t gsizeProp[1] = {m_sizePad.width * m_sizePad.height};
		size_t gsizeC2R[2] = {m_sizeOrig.width, m_sizeOrig.height};
		
		// Propagate
		ocl::Kernel("propagate", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::PtrReadOnly(m_dft),
			ocl::KernelArg::PtrReadOnly(m_prop),
			ocl::KernelArg::PtrWriteOnly(m_complex),
			Mz(z)
		).run(1, gsizeProp, NULL, true);
		
		// IFFT
		idft(m_complex, m_complex, DFT_COMPLEX_INPUT|DFT_COMPLEX_OUTPUT);
		
		// Convert to real
		ocl::Kernel("c2r", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::ReadOnly(m_complex),
			ocl::KernelArg::WriteOnly(dst)
		).run(2, gsizeC2R, NULL, true);
	}
	
	void recon(std::vector<UMat>& dst, UMat& dstMin, float z0, float z1, float dz) CV_OVERRIDE
	{
		size_t gsizeProp[1] = {m_sizePad.width * m_sizePad.height};
		size_t gsizeC2R[2] = {m_sizeOrig.width, m_sizeOrig.height};
		
		int dstIdx = 0;
		int ndst = dst.size();
		for (float z = z0; z < z1 && dstIdx < ndst; z += dz) {
			// Propagate
			ocl::Kernel("propagate", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::PtrReadOnly(m_dft),
				ocl::KernelArg::PtrReadOnly(m_prop),
				ocl::KernelArg::PtrWriteOnly(m_complex),
				Mz(z)
			).run(1, gsizeProp, NULL, true);
			
			// IFFT
			idft(m_complex, m_complex, DFT_COMPLEX_INPUT|DFT_COMPLEX_OUTPUT);
			
			// Convert to real
			ocl::Kernel("c2r_min", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst[dstIdx++]),
				ocl::KernelArg::PtrReadWrite(dstMin)
			).run(2, gsizeC2R, NULL, true);
		}
	}
};

static void focusMin(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, int n)
{
	std::vector<UMat> slice(n);
	std::vector<double> min(n);
	
	for (int i = 0; i < n; i++)
		UMat(src[i], rect).copyTo(slice[i]);
	
	for (int i = 0; i < n; i++)
		minMaxLoc(slice[i], &min[i]);
	
	for (int i = 0; i < n; i++) {
		double val = 255.0 - min[i];
		if (val > score) {
			score = val;
			idx = i;
		}
	}
}

static void focusSTD(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, int n)
{
	ocl::Queue q = ocl::Queue::getDefault();
	size_t gsize[2] = {rect.width, rect.height};
	
	std::vector<UMat> slice(n);
	std::vector<UMat> filt(n);
	std::vector<Vec<double,1>> mean(n);
	std::vector<Vec<double,1>> stddev(n);
	
	for (int i = 0; i < n; i++) {
		slice[i] = UMat(rect.height, rect.width, CV_8UC1);
		filt[i] = UMat(rect.height, rect.width, CV_8UC1);
	}
	
	for (int i = 0; i < n; i++)
		UMat(src[i], rect).copyTo(slice[i]);
	
	for (int i = 0; i < n; i++) {
		ocl::Kernel("stdfilt_3x3", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::PtrReadOnly(slice[i]),
			ocl::KernelArg::WriteOnly(filt[i])
		).run(2, gsize, NULL, false, q);
	}
	q.finish();
	
	for (int i = 0; i < n; i++)
		meanStdDev(filt[i], mean[i], stddev[i]);
	
	for (int i = 0; i < n; i++) {
		double val = stddev[i][0];
		if (val > score) {
			score = val;
			idx = i;
		}
	}
}

void Hologram::focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method, int n)
{
	int sz = src.size();
	n = n < 0 || n > sz ? sz : n;
	
	switch (method) {
	case FOCUS_MIN:
		focusMin(src, rect, idx, score, n);
		break;
	case FOCUS_STD:
		focusSTD(src, rect, idx, score, n);
		break;
	default:
		focusSTD(src, rect, idx, score, n);
	}
}

Ptr<Hologram> Hologram::create(Size2i size, float psz, float dist, float lambda)
{
	return makePtr<HologramImpl>(size, psz, dist, lambda);
}

}}
