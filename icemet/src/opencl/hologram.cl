typedef float2 cfloat;

/* Real part. */
__attribute__((always_inline))
float creal(cfloat z)
{
	return z.x;
}

/* Imaginiary part. */
__attribute__((always_inline))
float cimag(cfloat z)
{
	return z.y;
}

/* Create complex number. */
__attribute__((always_inline))
cfloat cnum(float r, float i)
{
	return (cfloat)(r, i);
}

/* Addition. */
/* UNUSED
__attribute__((always_inline))
cfloat cadd(cfloat z1, cfloat z2)
{
	return z1 + z2;
}
*/

/* Subtraction. */
/* UNUSED
__attribute__((always_inline))
cfloat csub(cfloat z1, cfloat z2)
{
	return z1 - z2;
}
*/

/* Multiplication. */
__attribute__((always_inline))
cfloat cmul(cfloat z1, cfloat z2)
{
	return (cfloat)(
		z1.x*z2.x - z1.y*z2.y,
		z1.x*z2.y + z1.y*z2.x
	);
}

/* Division. */
/* UNUSED
__attribute__((always_inline))
cfloat cdiv(cfloat z1, cfloat z2)
{
	return (cfloat)(
		(z1.x*z2.x + z1.y*z2.y) / (z2.x*z2.x + z2.y*z2.y),
		(z1.y*z2.x - z1.x*z2.y) / (z2.x*z2.x + z2.y*z2.y)
	);
}
*/

/* Absolute. */
__attribute__((always_inline))
float cabs(cfloat z)
{
	return length(z);
}

/* Exponential function. */
__attribute__((always_inline))
cfloat cexp(cfloat z)
{
	float expx = exp(z.x);
	return (cfloat)(expx * cos(z.y), expx * sin(z.y));
}

/* Sine. */
__attribute__((always_inline))
cfloat csin(cfloat z)
{
	float x = z.x;
	float y = z.y;
	return (cfloat)(sin(x) * cosh(y), cos(x) * sinh(y));
}

/* Cosine. */
__attribute__((always_inline))
cfloat ccos(cfloat z)
{
	float x = z.x;
	float y = z.y;
	return (cfloat)(cos(x) * cosh(y), -sin(x) * sinh(y));
}

/* Tangent. */
/* UNUSED
__attribute__((always_inline))
cfloat ctan(cfloat z)
{
	return cdiv(csin(z), ccos(z));
}
*/

/* FFT shift e^(dir * i * pi * (x+y)). */
__attribute__((always_inline))
cfloat shift(int x, int y, int dir)
{
	return cexp(cmul(cnum(dir * M_PI * (x+y), 0.0), cnum(0.0, 1.0)));
}

/* Convert real numbers to complex. */
__kernel void r2c(
	__global uchar* src,
	int src_step, int src_offset,
	int src_h, int src_w,
	__global cfloat* dst,
	int dst_step, int dst_offset,
	int dst_h, int dst_w
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	if (x >= src_w || y >= src_h)
		dst[y*dst_w + x] = cnum(0.0, 0.0);
	else
		dst[y*dst_w + x] = cmul(cnum(src[y*src_w + x], 0.0), shift(x, y, 1));
}

/* Converts complex numbers to real. */
__kernel void c2r(
	__global cfloat* src,
	int src_step, int src_offset,
	int src_h, int src_w,
	__global uchar* dst,
	int dst_step, int dst_offset,
	int dst_h, int dst_w
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	dst[y*dst_w + x] = cabs(cmul(src[y*src_w + x], shift(x, y, -1)));
}

/* Converts complex numbers to real and updates minimum image. */
__kernel void c2r_min(
	__global cfloat* src,
	int src_step, int src_offset,
	int src_h, int src_w,
	__global uchar* dst,
	int dst_step, int dst_offset,
	int dst_h, int dst_w,
	__global uchar* img_min
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	uchar val = cabs(cmul(src[y*src_w + x], shift(x, y, -1)));
	img_min[y*dst_w + x] = min(val, img_min[y*dst_w + x]);
	dst[y*dst_w + x] = val;
}

/* Generates angular spectrum propagator. */
__kernel void angularspectrum(
	__global cfloat* prop,
	int step, int offset,
	int h, int w,
	float2 size,
	float lambda
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int i = y*w + x;
	
	float u = ((float)x - w/2.0 - 1.0) / size.x;
	float v = ((float)y - h/2.0 - 1.0) / size.y;
	
	// (2*pi*i)/lambda * sqrt(1 - (lambda*u)^2 - (lambda*v)^2)
	float root = native_sqrt(1 - pow(lambda*u, 2) - pow(lambda*v, 2));
	prop[i] = cmul(cnum(2 * M_PI * root / lambda, 0.0), cnum(0.0, 1.0));
}

/* Performs propagation. */
__kernel void propagate(
	__global cfloat* src,
	__global cfloat* prop,
	__global cfloat* dst,
	float z
)
{
	const int i = get_global_id(0);
	
	// x * e^(z * prop)
	dst[i] = cmul(src[i], cexp(cmul(prop[i], cnum(z, 0.0))));
}

/* Generates a super-Gaussian low-pass filter. */
__kernel void lpf(
	__global cfloat* H,
	int step, int offset,
	int h, int w,
	float2 size,
	float2 sigma,
	int n
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int i = y*w + x;
	
	float u = ((float)x - w/2.0 - 1.0) / size.x;
	float v = ((float)y - h/2.0 - 1.0) / size.y;
	H[i] = cnum(exp(-1.0/2.0 * pow(pow(u / sigma.x, 2) + pow(v / sigma.y, 2), n)), 0.0);
}

/* Applies 3x3 standard deviation filter to image. */
__kernel void stdfilt_3x3(
	__global uchar* src,
	__global uchar* dst,
	int step, int offset,
	int h, int w
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	
	float sum = 0.0;
	float sqsum = 0.0;
	int n = 0;
	for (int xx = max(x-1, 0); xx <= min(x+1, w-1); xx++) {
		for (int yy = max(y-1, 0); yy <= min(y+1, h-1); yy++) {
			float val = src[yy*w + xx];
			sum += val;
			sqsum += val*val;
			n++;
		}
	}
	float mean = sum / n;
	float var = sqsum / n - mean*mean;
	dst[y*w + x] = native_sqrt(var);
}
