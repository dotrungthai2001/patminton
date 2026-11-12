#ifndef __PAT_TYPES__
#define __PAT_TYPES__

enum class EllipticMode {Lut, Exact, Trapezoidal} ;

// Map of the time step (in upsampled bins) as a function of the position of the voxel in the
// local frame of the transducer (x along e2, y along e3 = e1 x e2, z along the cylinder axis e1,
// origin at the transducer center). The step of a voxel is the value of the nearest node
// (positions outside the map are clamped to its border). Passed by value to the kernels.
struct StepsMap {
	const int* steps ;          // (nx, ny, nz) row-major on the device, nullptr = no map (scalar steps)
	const double* x ;           // increasing node coordinates (device), sizes nx, ny, nz
	const double* y ;
	const double* z ;
	int nx, ny, nz ;
} ;

#endif
