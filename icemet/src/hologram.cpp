#include "precomp.hpp"

#include "opencl_kernels_icemet.hpp"

#define FILTER_N 6
#define FILTER_F 0.5

namespace cv { namespace icemet {

typedef struct _focus_param {
	int type; // OpenCV type
	ReconOutput output;
	std::function<double(const UMat&)> scoreFunc;
} FocusParam;

int ZRange::n()
{
	return roundf((stop - start) / step);
}

float ZRange::z(int i)
{
	return start + i*step;
}

int ZRange::i(float z)
{
	return roundf((z - start) / step);
}

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

static double KSearch(std::function<double(double)> f, double begin, double end, double K, TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 2.0))
{
	int count = 0;
	while (fabs(end - begin) > termcrit.epsilon && count++ < termcrit.maxCount) {
		double low = begin + (end - begin) / K;
		double high = end - (end - begin) / K;
		
		if (f(low) < f(high))
			begin = low;
		else
			end = high;
	}
	return (end + begin) / 2.0;
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
		img.convertTo(cv::UMat(padded, cv::Rect(cv::Point(0, 0), m_sizeOrig)), CV_32FC1);
		
		// FFT
		dft(padded, m_dft, DFT_COMPLEX_OUTPUT|DFT_SCALE, m_sizeOrig.height);
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
	
	void min(UMat& dst, ZRange z) CV_OVERRIDE
	{
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		int n = z.n();
		
		if (dst.empty())
			dst = UMat(m_sizeOrig, CV_8UC1, Scalar(255));
		for (int i = 0; i < n; i++) {
			propagate(z.z(i));
			ocl::Kernel("min_8u", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst)
			).run(2, gsize, NULL, true);
		}
	}
	
	void reconMin(std::vector<UMat>& dst, UMat& dstMin, ZRange z) CV_OVERRIDE
	{
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		int n = z.n();
		
		if (dstMin.empty())
			dstMin = UMat(m_sizeOrig, CV_8UC1, Scalar(255));
		int empty = n - dst.size();
		for (int i = 0; i < empty; i++)
			dst.emplace_back(m_sizeOrig, CV_8UC1);
		
		for (int i = 0; i < n; i++) {
			propagate(z.z(i));
			ocl::Kernel("amplitude_min_8u", ocl::icemet::hologram_oclsrc).args(
				ocl::KernelArg::ReadOnly(m_complex),
				ocl::KernelArg::WriteOnly(dst[i]),
				ocl::KernelArg::PtrReadWrite(dstMin)
			).run(2, gsize, NULL, true);
		}
	}
	
	float focus(ZRange z, FocusMethod method, float K) CV_OVERRIDE
	{
		const FocusParam* param = getFocusParam(method);
		UMat slice(m_sizeOrig, param->type);
		std::map<int,double> scores;
		auto f = [&](double x) {
			int i = round(x);
			auto it = scores.find(i);
			if (it != scores.end()) {
				return it->second;
			}
			else {
				recon(slice, z.z(i), param->output);
				double newScore = param->scoreFunc(slice);
				scores[i] = newScore;
				return newScore;
			}
		};
		return z.z(KSearch(f, 0, z.n()-1, K));
	}
	
	float focus(ZRange z, std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method, float K) CV_OVERRIDE
	{
		const FocusParam* param = getFocusParam(method);
		if (src.empty())
			src = std::vector<UMat>(z.n());
		UMat slice(rect.size(), param->type);
		std::map<int,double> scores;
		auto f = [&](double x) {
			int i = round(x);
			auto it = scores.find(i);
			if (it != scores.end()) {
				return it->second;
			}
			else {
				if (src[i].empty())
					recon(src[i], z.z(i), param->output);
				UMat(src[i], rect).convertTo(slice, param->type);
				double newScore = param->scoreFunc(slice);
				scores[i] = newScore;
				return newScore;
			}
		};
		idx = KSearch(f, 0, z.n()-1, K);
		score = f(idx);
		return z.z(idx);
	}
	
	void applyFilter(const UMat& H) CV_OVERRIDE
	{
		mulSpectrums(m_dft, H, m_dft, 0);
	}
	
	UMat createFilter(float f, FilterType type) const CV_OVERRIDE
	{
		float sigma = f * pow(log(1.0/pow(FILTER_F, 2)), -1.0/(2.0*FILTER_N));
		UMat H(m_sizePad, CV_32FC2);
		size_t gsize[2] = {(size_t)m_sizePad.width, (size_t)m_sizePad.height};
		ocl::Kernel("supergaussian", ocl::icemet::hologram_oclsrc).args(
			ocl::KernelArg::WriteOnly(H),
			Vec2f(m_psz*m_sizePad.width, m_psz*m_sizePad.height),
			type,
			Vec2f(sigma, sigma),
			FILTER_N
		).run(2, gsize, NULL, true);
		return H;
	}
	
	UMat createLPF(float f) const CV_OVERRIDE
	{
		return createFilter(f, FILTER_LOWPASS);
	}
	
	UMat createHPF(float f) const CV_OVERRIDE
	{
		return createFilter(f, FILTER_HIGHPASS);
	}
};

float Hologram::magnf(float dist, float z)
{
	return dist == 0.0 ? 1.0 : dist / (dist - z);
}

void Hologram::focus(std::vector<UMat>& src, const Rect& rect, int &idx, double &score, FocusMethod method, int begin, int end, float K)
{
	int sz = src.size();
	end = end < 0 || end > sz-1 ? sz-1 : end;
	
	const FocusParam* param = getFocusParam(method);
	UMat slice(rect.height, rect.width, param->type);
	std::map<int,double> scores;
	auto f = [&](double x) {
		int i = round(x);
		auto it = scores.find(i);
		if (it != scores.end()) {
			return it->second;
		}
		else {
			UMat(src[i], rect).convertTo(slice, param->type);
			double newScore = param->scoreFunc(slice);
			scores[i] = newScore;
			return newScore;
		}
	};
	idx = KSearch(f, begin, end, K);
	score = f(idx);
}

Ptr<Hologram> Hologram::create(Size2i size, float psz, float lambda, float dist)
{
	return makePtr<HologramImpl>(size, psz, lambda, dist);
}

}}
