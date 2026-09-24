#ifndef __PAT_FUNCTIONS__
#define __PAT_FUNCTIONS__

#include "pat_types.h"
#include <cufft.h>

// ======================================
// === Functions manipulating arrays ===
// ======================================

void compute_inv_increments_array(double *a, double *delta_a, int n) ; // not CUDA function

__global__ void zeroingArray(double *f, int nx, int ny, int nz) ;
__global__ void zeroingArray(cufftDoubleComplex *f, int nx, int ny, int nz) ;
__global__ void batchComplexMultiplication(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) ;
__global__ void batchComplexMultiplicationConj(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) ;
__global__ void complexMultiplication1d(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) ;
__global__ void complexMultiplicationConj1d(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) ;
__global__ void subSampling(double su[], double s[], int nx, int ny, int upsampling) ;
__global__ void upSampling(double su[], double s[], int nx, int ny, int upsampling) ;

__global__ void multConstant(double f[], double a, int nx, int ny) ;

// ======================================
// === Convolution kernel ===
// ======================================
__device__ double dI(double d) ;
__global__ void initConvKernel(double g[], int n, double dx, double dt, double c) ;
__global__ void initIndicator(double g[], int n, int upsampling) ;
__global__ void initGaussian(double g[], int n, double dt, double sigma) ;

// ======================================
// === Point-based transducers ===
// ======================================

__global__ void linkingPointsToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, int nPoints, int nTrans,
					double tStart, int nT, double dt, double c, double area[],
					double locPoints[], bool use_sparse_optimization, double p[], double s[]) ;

__global__ void linkingPointsToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, int nPoints, int nTrans,
					double tStart, int nT, double dt, double c, double area[],
					double locPoints[], double p[], double s[]) ;

// ======================================
// === Cylindrical transducers ===
// ======================================


// __global__ void linkingCylinderToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
// 					int nTrans,
// 					double tStart, int nT, double dt, double c,
// 					int upsample, int steps_border,
// 					double infos_transducers[], bool use_sparse_optimization, double p[], double s[]) ;

// __global__ void linkingCylinderToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
// 					int nTrans,
// 					double tStart, int nT, double dt, double c,
// 					int upsample, int steps_border,
// 					double infos_transducers[], double p[], double s[]) ;

__device__ void gridToCylinderCoords(double x, double y, double z,
										double e1x, double e1y, double e1z,
										double e2x, double e2y, double e2z,
										double &x_local, double &y_local, double &z_local) ;

__device__ void splitThetaDomain(double cmin, double cmax, double Theta[2][2], int &n_xi) ;
__device__ void splitZDomain(double zl, double h, double Zsq[2][2], int &n_upsilon) ;

// =============================================
// === Cylindrical transducers with Elliptic ===
// =============================================

// Largest thread block the kernels below accept: blockSize = 8 (the default)
// gives 8^3 = 512 threads. With one block per SM, this caps registers at 128
// per thread; without it the Exact kernels use up to 154 and fail to launch.
#define ELLIPTIC_MAX_THREADS 512

/// Forward operator using elliptic integrals (with bilinear-interpolated LUTs)
template <EllipticMode mode> __global__ void __launch_bounds__(ELLIPTIC_MAX_THREADS, 1) linkingCylinderToGrid_Elliptic(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
    				int nTrans,
    				double tStart, int nT, double dt, double c,
					const double* const infos_transducers,
                    const double* __restrict__ lut,
					const double* __restrict__ sinphi0, const double* __restrict__  k0,
     				const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
    				int Nphi, int Nk, double eps,
                    int upsample, int steps_border, int steps,
					const bool use_sparse_optimization,
					double d_p[], double d_s []);

/// Adjoint operator using elliptic integrals (with LUTs)
template <EllipticMode mode> __global__ void __launch_bounds__(ELLIPTIC_MAX_THREADS, 1) linkingCylinderToGridT_Elliptic(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
				int nTrans,
				double tStart, int nT, double dt, double c,
				const double* const infos_transducers,
                const double* __restrict__ lut,
				const double* __restrict__ sinphi0, const double* __restrict__  k0,
    			const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
				int Nphi, int Nk, double eps,
                int upsample, int steps_border, int steps,
				double d_p[], double d_s []);

__device__ double bilinear_interp_E_lut(double sinphi, double k,
					const double* __restrict__ lut, // size (Nphi,Nk)
					const double* __restrict__ sinphi0, const double* __restrict__ k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
					int Nphi, int Nk,
					double eps) ;

__device__ void bilinear_interp_EF_lut(double sinphi, double k,
					const double* __restrict__ lut, // size (Nphi,Nk)
					const double* __restrict__ sinphi0, const double* __restrict__ k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
					int Nphi, int Nk,
					double eps,
					double &valueE, double &valueF) ;

__device__ double compute_I(double cos_alpha, double cos_beta, double a, double b,
						const double* __restrict__ lut,
						const double* __restrict__ sinphi0, const double* __restrict__ k0,
						const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
                        double eps,
                        int Nphi, int Nk) ;

__device__ double compute_I_exact(double alpha, double beta, double a, double b) ;

__device__ inline int computeStep(int itime, int it_min, int it_max, int it_alpha, int it_beta, int steps_border, int steps, int upsample) ;


// ======================================
// === Planar transducers ===
// ======================================

__global__ void linkingPlanesToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					double infos_transducers[],
					int upsample, int steps_border, int steps,
					bool use_sparse_optimization,
					double d_p[], double d_s[]) ;

__global__ void linkingPlanesToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					double infos_transducers[],
					int upsample, int steps_border, int steps,
					double d_p[], double d_s[]) ;

__device__ double compute_I_plane(double a, double x) ;

__device__ void gridToPlaneCoords(double x, double y, double z,
								double e1x, double e1y, double e1z,
								double e2x, double e2y, double e2z,
								double &x_local, double &y_local, double &z_local) ;
// ======================================
// === Utilities ===
// ======================================

__device__ double wrap_angle(double angle) ;
__device__ double fourth_power(double x) ;
__device__ inline double third_power(double x) ;
__device__ inline double squared(double x) ;
__device__ inline double fast_inversion(double x) ;


#endif
