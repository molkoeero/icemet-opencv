/* Adjusts image intensity values. */
__kernel void adjust(
	__global uchar* src,
	__global uchar* dst,
	uchar a0, uchar a1,
	uchar b0, uchar b1
)
{
	int i = get_global_id(0);
	float val = src[i];
	
	if (val < a0)
		val = a0;
	else if (val > a1)
		val = a1;
	
	val -= a0;
	val /= a1 - a0;
	val *= b1 - b0;
	val += b0;
	dst[i] = val;
}

/* Generic histogram. */
__kernel void hist(
	__global float* src,
	__global int* dst,
	float min, float max,
	float step
)
{
	float val = src[get_global_id(0)];
	if (val < min || val > max)
		return;
	int i = round((val-min) / step);
	atomic_inc(&dst[i]);
}

/* Image histogram. */
__kernel void imghist(__global uchar* src, __global int* dst)
{
	atomic_inc(&dst[src[get_global_id(0)]]);
}
