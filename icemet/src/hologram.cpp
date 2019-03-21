#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

#define LPF_N 6
#define LPF_F 0.5

namespace cv { namespace icemet {

typedef struct _focus_result {
	int imax;     // Index of the maximum value
	int il;       // Left of the maximum
	int ir;       // Right of the maximum
	double score; // Maximum score found
} FocusResult;

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
	inline float Mz(float z) const
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
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		ocl::Kernel("angularspectrum", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::WriteOnly(m_prop),
			Vec2f(psz*m_sizePad.width, psz*m_sizePad.height),
			m_lambda
		).run(2, gsize, NULL, true);
	}
	
	void setImg(const UMat& img) CV_OVERRIDE
	{
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		
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
		size_t gsizeProp[1] = {(size_t)(m_sizePad.width * m_sizePad.height)};
		size_t gsizeC2R[2] = {(size_t)m_sizeOrig.width, (size_t)m_sizeOrig.height};
		
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
		size_t gsizeProp[1] = {(size_t)(m_sizePad.width * m_sizePad.height)};
		size_t gsizeC2R[2] = {(size_t)m_sizeOrig.width, (size_t)m_sizeOrig.height};
		
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
	
	void applyFilter(const UMat& H)
	{
		mulSpectrums(m_dft, H, m_dft, 0);
	}
	
	cv::UMat createLPF(float f) const
	{
		float sigma = f * pow(log(1.0/pow(LPF_F, 2)), -1.0/(2.0*LPF_N));
		cv::UMat H(m_sizePad, CV_32FC2);
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		ocl::Kernel("lpf", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::WriteOnly(H),
			Vec2f(m_psz*m_sizePad.width, m_psz*m_sizePad.height),
			Vec2f(sigma, sigma),
			LPF_N
		).run(2, gsize, NULL, true);
		return H;
	}
};

static double scoreMin(const UMat& slice)
{
	double min;
	minMaxLoc(slice, &min);
	return 255.0 - min;
}

static double scoreSTD(const UMat& slice)
{
	size_t gsize[2] = {(size_t)slice.cols, (size_t)slice.rows};
	UMat filt(slice.size(), CV_8UC1);
	Vec<double,1> mean;
	Vec<double,1> stddev;
	ocl::Kernel("stdfilt_3x3", ocl::icemet::hologram_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(slice),
		ocl::KernelArg::WriteOnly(filt)
	).run(2, gsize, NULL, true);
	meanStdDev(filt, mean, stddev);
	return stddev[0];
}

static int generateIndices(std::vector<int>& dst, int first, int last, int n)
{
	dst.clear();
	float step = (float)(last - first) / n;
	int prev = -1;
	for (float idxf = first; roundf(idxf) < last; idxf += step) {
		int idx = roundf(idxf);
		if (idx != prev) {
			prev = idx;
			dst.push_back(idx);
		}
	}
	int size = dst.size();
	if (size)
		dst[size-1] = last;
	return size;
}

static void findBestIndices(FocusResult& res, const std::vector<double>& scores, const std::vector<int>& indices)
{
	res.imax = 0;
	res.il = 0;
	res.ir = 0;
	res.score = 0;
	int iimax = 0;
	int sz = indices.size();
	
	// Find the index of largest value
	for (int ii = 0; ii < sz; ii++) {
		int i = indices[ii];
		double score = scores[ii];
		if (score > scores[iimax]) {
			res.imax = i;
			res.score = score;
			iimax = ii;
		}
	}
	
	// Save left and right
	res.il = iimax > 0 ? indices[iimax-1] : res.imax;
	res.ir = iimax < sz-1 ? indices[iimax+1] : res.imax;
}

void Hologram::focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method, int first, int last, int points)
{
	int sz = src.size();
	first = first < 0 ? 0 : first;
	last = last < 0 || last > sz-1 ? sz-1 : last;
	
	// Select the scoring function
	double (*scoreFunc)(const UMat&);
	switch (method) {
	case FOCUS_MIN:
		scoreFunc = scoreMin;
		break;
	case FOCUS_STD:
		scoreFunc = scoreSTD;
		break;
	default:
		scoreFunc = scoreSTD;
	}
	
	// Allocate slices
	std::vector<UMat> slices(points);
	for (int i = 0; i < points; i++)
		slices[i] = UMat(rect.height, rect.width, CV_8UC1);
	
	// Start iterating
	FocusResult res = {0, first, last, 0.0};
	int n = points;
	while (n >= points) {
		std::vector<int> indices;
		n = generateIndices(indices, res.il, res.ir, points);
		
		std::vector<double> scores(n);
		for (int i = 0; i < n; i++)
			UMat(src[indices[i]], rect).copyTo(slices[i]);
		for (int i = 0; i < n; i++)
			scores[i] = scoreFunc(slices[i]);
		
		findBestIndices(res, scores, indices);
		idx = res.imax;
		score = res.score;
	}
}

Ptr<Hologram> Hologram::create(Size2i size, float psz, float dist, float lambda)
{
	return makePtr<HologramImpl>(size, psz, dist, lambda);
}

}}
