#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

#define LPF_N 6
#define LPF_F 0.5

namespace cv { namespace icemet {

typedef struct _focus_param {
	int type; // OpenCV type
	ReconOutput output;
	double (*scoreFunc)(const UMat&);
} FocusParam;

typedef struct _focus_result {
	int imax;     // Index of the maximum value
	int il;       // Left of the maximum
	int ir;       // Right of the maximum
	double score; // Maximum score found
} FocusResult;

static double scoreMin(const UMat& slice)
{
	double min, max;
	minMaxLoc(slice, &min, &max);
	return -min;
}

static double scoreMax(const UMat& slice)
{
	double min, max;
	minMaxLoc(slice, &min, &max);
	return max;
}

static double scoreRange(const UMat& slice)
{
	double min, max;
	minMaxLoc(slice, &min, &max);
	return max - min;
}

static double scoreSTD(const UMat& slice)
{
	size_t gsize[2] = {(size_t)slice.cols, (size_t)slice.rows};
	UMat filt(slice.size(), CV_32FC1);
	Vec<double,1> mean;
	Vec<double,1> stddev;
	ocl::Kernel("stdfilt_3x3", ocl::icemet::hologram_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(slice),
		ocl::KernelArg::WriteOnly(filt)
	).run(2, gsize, NULL, true);
	meanStdDev(filt, mean, stddev);
	return stddev[0];
}

static double scoreToG(const UMat& slice)
{
	size_t gsize[2] = {(size_t)slice.cols, (size_t)slice.rows};
	UMat grad(slice.size(), CV_32FC1);
	Vec<double,1> mean;
	Vec<double,1> stddev;
	ocl::Kernel("gradient", ocl::icemet::hologram_oclsrc).args(
		ocl::KernelArg::PtrReadOnly(slice),
		ocl::KernelArg::WriteOnly(grad)
	).run(2, gsize, NULL, true);
	meanStdDev(grad, mean, stddev);
	return sqrt(stddev[0] / mean[0]);
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
	for (int i = 0; i < sz; i++) {
		int idx = indices[i];
		double score = scores[i];
		if (score > scores[iimax]) {
			res.imax = idx;
			res.score = score;
			iimax = i;
		}
	}
	
	// Save left and right
	res.il = iimax > 0 ? indices[iimax-1] : res.imax;
	res.ir = iimax < sz-1 ? indices[iimax+1] : res.imax;
}

static const FocusParam focusParam[] {
	{CV_32FC1, RECON_OUTPUT_AMPLITUDE, scoreMin},
	{CV_32FC1, RECON_OUTPUT_AMPLITUDE, scoreMax},
	{CV_32FC1, RECON_OUTPUT_AMPLITUDE, scoreRange},
	{CV_32FC1, RECON_OUTPUT_AMPLITUDE, scoreSTD},
	{CV_32FC2, RECON_OUTPUT_COMPLEX, scoreToG}
};

static const FocusParam* getFocusParam(FocusMethod method)
{
	return &focusParam[method];
}

class HologramImpl : public Hologram {
private:
	Size2i m_sizeOrig;
	Size2i m_sizePad;
	
	float m_psz;
	float m_lambda;
	float m_dist;
	
	UMat m_prop;
	UMat m_dft;
	UMat m_complex;
	
	void propagate(float z)
	{
		size_t gsizeProp[1] = {(size_t)(m_sizePad.width * m_sizePad.height)};
		ocl::Kernel("propagate", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::PtrReadOnly(m_dft),
			ocl::KernelArg::PtrReadOnly(m_prop),
			ocl::KernelArg::PtrWriteOnly(m_complex),
			z * magnf(m_dist, z)
		).run(1, gsizeProp, NULL, true);
		idft(m_complex, m_complex, DFT_COMPLEX_INPUT|DFT_COMPLEX_OUTPUT);
	}

public:
	HologramImpl(Size2i size, float psz, float lambda, float dist) :
		m_sizeOrig(size),
		m_psz(psz),
		m_lambda(lambda),
		m_dist(dist)
	{
		m_sizePad = Size2i(getOptimalDFTSize(size.width), getOptimalDFTSize(size.height));
		
		// Allocate UMats
		m_prop = UMat::zeros(m_sizePad, CV_32FC2);
		m_dft = UMat::zeros(m_sizePad, CV_32FC2);
		m_complex = UMat::zeros(m_sizePad, CV_32FC2);
		
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
		CV_Assert(img.channels() == 1 && img.size() == m_sizeOrig);
		
		cv::UMat padded(m_sizePad, CV_32FC1, cv::mean(img));
		img.copyTo(cv::UMat(padded, cv::Rect(cv::Point(0, 0), m_sizeOrig)));
		
		// FFT
		dft(padded, m_dft, DFT_COMPLEX_INPUT|DFT_COMPLEX_OUTPUT|DFT_SCALE, m_sizeOrig.height);
	}
	
	void recon(UMat& dst, float z, ReconOutput output) CV_OVERRIDE
	{
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		propagate(z);
		switch (output) {
		case RECON_OUTPUT_COMPLEX:
			UMat(m_complex, Rect(Point(0, 0), m_sizeOrig)).copyTo(dst);
			return;
		case RECON_OUTPUT_AMPLITUDE:
			if (dst.empty())
				dst = UMat(m_sizeOrig, CV_32FC1);
			ocl::Kernel("amplitude", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst)
			).run(2, gsize, NULL, true);
			break;
		case RECON_OUTPUT_PHASE:
			if (dst.empty())
				dst = UMat(m_sizeOrig, CV_32FC1);
			ocl::Kernel("phase", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst),
				m_lambda,
				z * magnf(m_dist, z)
			).run(2, gsize, NULL, true);
			break;
		}
	}
	
	void reconMin(std::vector<UMat>& dst, UMat& dstMin, float z0, float z1, float dz) CV_OVERRIDE
	{
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		int n = roundf((z1 - z0) / dz);
		
		if (dstMin.empty())
			dstMin = UMat(m_sizeOrig, CV_8UC1, Scalar(255));
		int empty = n - dst.size();
		for (int i = 0; i < empty; i++)
			dst.emplace_back(m_sizeOrig, CV_8UC1);
		
		for (int i = 0; i < n; i++) {
			propagate(z0 + i*dz);
			ocl::Kernel("amplitude_min_8u", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst[i]),
				ocl::KernelArg::PtrReadWrite(dstMin)
			).run(2, gsize, NULL, true);
		}
	}
	
	float focus(float z0, float z1, float dz, FocusMethod method, int points) CV_OVERRIDE
	{
		std::map<int, double> scoreMap;
		const FocusParam* param = getFocusParam(method);
		FocusResult res = {0, 0, (int)roundf((z1 - z0) / dz)-1, 0.0};
		UMat slice(m_sizeOrig, param->type);
		
		// Start iterating
		int n = points;
		float z = z0;
		while (n >= points) {
			std::vector<int> indices;
			n = generateIndices(indices, res.il, res.ir, points);
			
			// Fill our score vector
			std::vector<double> scoreVec(n);
			for (int i = 0; i < n; i++) {
				int scoreIdx = indices[i];
				
				// Check if the score is already in our score map
				auto it = scoreMap.find(scoreIdx);
				if (it != scoreMap.end()) {
					scoreVec[i] = it->second;
				}
				else {
					// Calculate the score
					recon(slice, z0 + scoreIdx*dz, param->output);
					double newScore = param->scoreFunc(slice);
					scoreVec[i] = newScore;
					scoreMap[scoreIdx] = newScore;
				}
			}
			
			// Find the maximum value and surrounding indices
			findBestIndices(res, scoreVec, indices);
			z = z0 + res.imax*dz;
		}
		return z;
	}
	
	void applyFilter(const UMat& H) CV_OVERRIDE
	{
		mulSpectrums(m_dft, H, m_dft, 0);
	}
	
	UMat createLPF(float f) const CV_OVERRIDE
	{
		float sigma = f * pow(log(1.0/pow(LPF_F, 2)), -1.0/(2.0*LPF_N));
		UMat H(m_sizePad, CV_32FC2);
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

float Hologram::magnf(float dist, float z)
{
	return dist == 0.0 ? 1.0 : dist / (dist - z);
}

void Hologram::focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method, int first, int last, int points)
{
	int sz = src.size();
	last = last < 0 || last > sz-1 ? sz-1 : last;
	
	// We store every score in this map to make sure we don't calculate same
	// scores multiple times
	std::map<int,double> scoreMap;
	const FocusParam* param = getFocusParam(method);
	FocusResult res = {0, first, last, 0.0};
	UMat slice(rect.height, rect.width, param->type);
	
	// Start iterating
	int n = points;
	while (n >= points) {
		std::vector<int> indices;
		n = generateIndices(indices, res.il, res.ir, points);
		
		// Fill our score vector
		std::vector<double> scoreVec(n);
		for (int i = 0; i < n; i++) {
			int scoreIdx = indices[i];
			
			// Check if the score is already in our score map
			auto it = scoreMap.find(scoreIdx);
			if (it != scoreMap.end()) {
				scoreVec[i] = it->second;
			}
			else {
				// Calculate the score
				UMat(src[scoreIdx], rect).convertTo(slice, param->type);
				double newScore = param->scoreFunc(slice);
				scoreVec[i] = newScore;
				scoreMap[scoreIdx] = newScore;
			}
		}
		
		// Find the maximum value and surrounding indices
		findBestIndices(res, scoreVec, indices);
		idx = res.imax;
		score = res.score;
	}
}

Ptr<Hologram> Hologram::create(Size2i size, float psz, float lambda, float dist)
{
	return makePtr<HologramImpl>(size, psz, lambda, dist);
}

}}
