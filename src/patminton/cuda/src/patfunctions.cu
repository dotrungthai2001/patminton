#include "patfunctions.cuh"
#include "pat_types.h"
#include "cudaUtility.h"
#include <cstdio>
#include <cufft.h>
#include "cufftUtility.h"
#include "math.h"
#include "elliptic_integrals.cuh"
// #include <boost/math/special_functions/ellint_2.hpp>
// #include <boost/math/special_functions/ellint_1.hpp>

#define PERIODIZE(x,n) ( (x < 0) ? (x+n) : ( (x>=n) ? (x-n) : x) )  // It ensures that the index x wraps around within the bounds [0, n-1].

// ======================================
// === Functions manipulating arrays ===
// ======================================

void compute_inv_increments_array(double *a, double *delta_a, int n) {
    for (int i = 0 ; i < (n-1) ; i++){
        delta_a[i] = 1/(a[i+1]-a[i]) ;
    }
}



__global__ void zeroingArray(double *f, int nx, int ny, int nz) {    // Sets all elements of a 3D array f to zero. The array is represented in a flattened 1D format.
	int i, j, k ;  // The function uses the CUDA thread and block indices to determine the 3D coordinates (i, j, k) of each element in the array.

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < nx) && (j < ny) && (k < nz)) {
		f[nz*ny*i + nz*j + k] = 0 ;
	}
}

__global__ void zeroingArray(cufftDoubleComplex *f, int nx, int ny, int nz) {    // Sets all elements of a 3D array f to zero. The array is represented in a flattened 1D format.
	int i, j, k ;  // The function uses the CUDA thread and block indices to determine the 3D coordinates (i, j, k) of each element in the array.

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < nx) && (j < ny) && (k < nz)) {
		f[nz*ny*i + nz*j + k].x = 0 ;
		f[nz*ny*i + nz*j + k].y = 0 ;
	}
}

// a of size [nx,ny]
// b of size [1,ny]
// compute a[i,:] <- a[i,:] * b for all i
__global__ void batchComplexMultiplication(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) {
	int i, j ;
	int Inda, Indb ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		double real, imag ;
		Inda = ny*i + j;
		Indb = j ;

		real = (a[Inda].x * b[Indb].x - a[Inda].y * b[Indb].y) ;
		imag = (a[Inda].x * b[Indb].y + a[Inda].y * b[Indb].x) ;
		a[Inda].x = real ;
		a[Inda].y = imag ;
	}
}

// a of size [nx,ny]
// b of size [1,ny]
// compute a[i,:] <- a[i,:] * conj(b) for all i
__global__ void batchComplexMultiplicationConj(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) {
	int i, j ;
	int Inda, Indb ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		double real, imag ;
		Inda = ny*i + j ;
		Indb = j ;

		real = (a[Inda].x * b[Indb].x + a[Inda].y * b[Indb].y) ;
		imag = (-a[Inda].x * b[Indb].y + a[Inda].y * b[Indb].x) ;
		a[Inda].x = real ;
		a[Inda].y = imag ;
	}
}

// a of size [nx,ny]
// b of size [nx,ny]
// compute a <- a * b
__global__ void complexMultiplication1d(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) {
	int i, j ;
	int Ind ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		double real, imag ;
		Ind = ny*i + j;

		real = (a[Ind].x * b[Ind].x - a[Ind].y * b[Ind].y) ;
		imag = (a[Ind].x * b[Ind].y + a[Ind].y * b[Ind].x) ;
		a[Ind].x = real ;
		a[Ind].y = imag ;
	}
}

__global__ void complexMultiplicationConj1d(cufftDoubleComplex a[], cufftDoubleComplex b[], int nx, int ny) {

   	int i, j ;
	int Ind ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		double real, imag ;
		Ind = ny*i + j ;

		real = (a[Ind].x * b[Ind].x + a[Ind].y * b[Ind].y) ;
		imag = (-a[Ind].x * b[Ind].y + a[Ind].y * b[Ind].x) ;
		a[Ind].x = real ;
		a[Ind].y = imag ;
	}

}
// nx, ny are the sizes of the s
// the size of the upsampled signal is nx x (ny*upsample)
// __global__ void subSampling(double su[], double s[], int nx, int ny, int upsampling) {
// 	int i, j ;
// 	int Ind, Indu ;

// 	i = blockIdx.x * blockDim.x + threadIdx.x ;
// 	j = blockIdx.y * blockDim.y + threadIdx.y ;

// 	if ( (i < nx) && (j < ny) ) {
// 		Ind = (ny * i) + j ;
// 		Indu = i*(ny*upsampling) + (j * upsampling);
// 		s[Ind] = su[Indu] ;
// 	}

// }


// nx, ny are the sizes of the s
// the size of the upsampled signal is nx x (ny*upsample)
__global__ void subSampling(double su[], double s[], int nx, int ny, int upsampling) {
	int i, j, k ;
	int Ind, Indu ;
	double value ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		Ind = (ny * i) + j ;
		Indu = i*(ny*upsampling) + (j * upsampling);

		value = 0. ;
		for (k = 0 ; k < upsampling ; k++) {
            value += su[Indu+k] ;
		}

		s[Ind] = value / upsampling ;
	}

}

// nx, ny are the sizes of the s
// the size of the upsampled signal is nx x (ny*upsample)
__global__ void upSampling(double su[], double s[], int nx, int ny, int upsampling) {
	int i, j ;
	int Ind, Indu ;

	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < upsampling*ny) ) {
		Ind = ny * i + j / upsampling;
		Indu =  i*(ny*upsampling) + j;

		su[Indu] = s[Ind] / upsampling ;
	}

}

__global__ void multConstant(double f[], double a, int nx, int ny) {
	int i, j ;
	i = blockIdx.x * blockDim.x + threadIdx.x ;
	j = blockIdx.y * blockDim.y + threadIdx.y ;

	if ( (i < nx) && (j < ny) ) {
		f[(ny * i) + j] *= a ;
	}

}

// ======================================
// === Convolution kernel ===
// ======================================

__device__ int sign(double x) {
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

__device__ double dI(double r) {
	double value = 0 ;
	double ad = abs(r) ;
	if (ad <= 1) {
		value = r * (r * sign(r) - 1) ;
	}
	return value ;

}

__global__ void initConvKernel(double g[], int n, double dx, double dt, double c) {
	int i ;
	i = blockIdx.x * blockDim.x + threadIdx.x ; // time point
	if (i < n) {
		double ii = (double)i - (double)n / 2;
		g[PERIODIZE(i+n/2,n)] = 2 * M_PI * dx * dI(c * dt *(double)(ii)/dx) ;
	}
}

__global__ void initIndicator(double g[], int n, int upsampling) {
   	int i ;
	i = blockIdx.x * blockDim.x + threadIdx.x ; // time point
	if (i < upsampling) {
        // g[PERIODIZE(i-upsampling-upsampling/2-20,n)] = 1./ (double)upsampling ;
        g[i] = 1.0 / ((double) upsampling) ;
        // g[i+upsampling] = 1.0 / (double) upsampling ;
	}

}

__global__ void initGaussian(double g[], int n, double dt, double sigma) {
	int i ;
	i = blockIdx.x * blockDim.x + threadIdx.x ; // time point
	if (i < n) {
		double ii = ((double)i - (double)n / 2) * dt ;
		double gaussian = exp(-0.5 * (ii*ii) / (sigma*sigma)) / (sigma * sqrt(2 * M_PI)) ;

		g[PERIODIZE(i+n/2,n)] = gaussian * dt;
	}
}

// ======================================
// === Point-based transducers ===
// ======================================

// This function is designed to perform computations on a grid of points, where each point is influenced by a set of sensors.
// Implements the forward operator 𝐴⋅𝑝, where 𝐴 is the system matrix, p is the pressure field, and the result is the sensor data s.
// The function is executed in parallel on a GPU, with each thread handling a specific grid point.
// The function is declared as __global__, indicating that it is a CUDA kernel that will be executed on the GPU.
__global__ void linkingPointsToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, int nSensors, int nTrans,
					double tStart, int nT, double dt, double c, double area[],
					double locSensors[], bool use_sparse_optimization, double p[], double s[]) {
	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {
		//getting info of grid point
		double dx, dy, dz, x, y, z, xs, ys, zs, a ;


		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double cvp = p[i*Ny*Nz + j*Nz + k] ;

		if (use_sparse_optimization ? (cvp > 1e-16) : true) {

		double dist, t ;
		double xxs, yys, zzs ;
		int it ;

		// looping over all sensor points
		for (int trans = 0 ; trans < nTrans ; trans++) {
			for (int sensor = 0 ; sensor < nSensors ; sensor++) {
				xs = locSensors[3*nSensors*trans + 3*sensor] ;
				ys = locSensors[3*nSensors*trans + 3*sensor + 1] ;
				zs = locSensors[3*nSensors*trans + 3*sensor + 2] ;
				a = area[nSensors*trans + sensor] ;

				xxs = x-xs ;
				yys = y-ys ;
				zzs = z-zs ;

				dist = sqrt( xxs*xxs + yys*yys +zzs*zzs) ;
				t = dist / c ;
				it = (int)floor((t-tStart)/dt) ;

				if ((it >= 0) and (it < nT)) {
					//s[k*nT+it] += cvp/dist ;
					atomicAdd(&s[trans*nT+it], a*cvp/dist) ; // atomicAdd ensures that these updates happen without conflicts by serializing access to s[].
				}
			}
		}
	}
	}
}

// Implements the adjoint operator 𝐴𝑇⋅𝑠, which is used in iterative reconstruction algorithms.
__global__ void linkingPointsToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, int nSensors, int nTrans,
					double tStart, int nT, double dt, double c, double area[],
					double locSensors[], double p[], double s[]) {
// p[]: Output array where the computed values for each grid point are stored. s[]: Input array containing sensor data.
	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of grid pt
	k = blockIdx.z * blockDim.z + threadIdx.z ; // depth of grid pt

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {
		//getting info of grid point
		double dx, dy, dz, x, y, z, xs, ys, zs, a ;

		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double value = 0.0 ;

		double dist, t ;
		double xxs, yys, zzs ;
		int it ;

		// looping over all sensor points
		for (int trans = 0 ; trans < nTrans ; trans++) {
			for (int sensor = 0 ; sensor < nSensors ; sensor++) {
				// locSensor has size: nTrans x nSensors x d
				xs = locSensors[3*nSensors*trans + 3*sensor] ;
				ys = locSensors[3*nSensors*trans + 3*sensor+1] ;
				zs = locSensors[3*nSensors*trans + 3*sensor+2] ;
				a = area[nSensors*trans + sensor] ;
				xxs = x-xs ;
				yys = y-ys ;
				zzs = z-zs ;

				dist = sqrt( xxs*xxs + yys*yys + zzs*zzs) ;
				t = dist / c ;
				it = (int)floor((t-tStart)/dt) ;

				if ((it >= 0) and (it < nT)) {
					value += a * s[trans * nT + it] / dist ;
				}
			}
		}
		p[i*Ny*Nz + j*Nz + k] = value ;
	}
}

// ======================================
// === Cylindrical transducers ===
// ======================================

// __global__ void linkingCylinderToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
// 					int nTrans,
// 					double tStart, int nT, double dt, double c,
// 					int upsample, int steps_border,
// 					double infos_transducers[], bool use_sparse_optimization, double p[], double s[]) {

// 	int i, j, k ;

// 	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
// 	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
// 	k = blockIdx.z * blockDim.z + threadIdx.z ;

// 	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

// 		//getting info of grid point
// 		double dx, dy, dz ; // grid size
// 		double x, y, z ; // coordinates
// 		double xs, ys, zs ; // coordinates centered in transducer system
// 		double xl, yl, zl ; // coordinales of grid point in the transducer system
// 		dx = (2*Lx) / (Nx-1) ;
// 		dy = (2*Ly) / (Ny-1) ;
// 		dz = (2*Lz) / (Nz-1) ;
//         // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
// 		x = -Lx + i*dx ;
// 		y = -Ly + j*dy ;
// 		z = -Lz + k*dz ;

// 		double vp = p[i*Ny*Nz + j*Nz + k] ;

// 		if (use_sparse_optimization ? (vp > 1e-16) : true) {

// 		double *s_transducer ;
// 		double Rvp ;
// 		double D, Dsq ;
// 		double a ;
// 		double cmin,cmax ;


// 		double xc, yc, zc ; // coordinates of transducer center
// 		double e1x, e1y, e1z ; // coordinates of the e1 vector
// 		double e2x, e2y, e2z ; // coordinates of the e2 vector
// 		double R, Rsq ; //radius
// 		double theta_max_trans; // half angle of aperture of the transducer
// 		double h ; // length of the transducer in its z-axis

// 		int n_xi, n_upsilon ;
// 		int xi, upsilon ;
// 		double Theta[2][2] ;
// 		double Zsq[2][2] ;

// 		double rmin, rmax, ralpha, rbeta ;
// 		int it_min, it_max, it_alpha, it_beta, itime ;
// 		double cosine_beta_next, cosine_alpha_next ;
// 		double alpha, beta, alpha_next, beta_next ;
// 		double zalpha, zbeta, zalpha_next, zbeta_next ;
// 		double time, time_next, r_next, r_next_sq ;
// 		double area, I, I_next ;

// 		// variable to store computations
// 		double theta_min, theta_max, cos_theta_min, cos_theta_max ;
// 		double Zsq_min, Zsq_max, Zmin, Zmax ;
// 		double constant_cosine_alpha, constant_cosine_beta ;
// 		double coeff_cosine ;
// 		double twoRD ;
// 		double value ;
// 		int it_min_upsample, it_max_upsample, step, istep ;

// 		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {

// 			// gathering infos of the transducer
// 			xc = infos_transducers[12*id_trans] ;
// 			yc = infos_transducers[12*id_trans+1] ;
// 			zc = infos_transducers[12*id_trans+2] ;

// 			e1x = infos_transducers[12*id_trans+3] ;
// 			e1y = infos_transducers[12*id_trans+4] ;
// 			e1z = infos_transducers[12*id_trans+5] ;

// 			e2x = infos_transducers[12*id_trans+6] ;
// 			e2y = infos_transducers[12*id_trans+7] ;
// 			e2z = infos_transducers[12*id_trans+8] ;

// 			R = infos_transducers[12*id_trans+9] ;
// 			Rsq = R*R ;

// 			theta_max_trans = infos_transducers[12*id_trans+10] ;
// 			h = infos_transducers[12*id_trans+11] ; // must be 1/2 ?

// 			Rvp = R * vp / c ;
// 			s_transducer = &s[id_trans*nT] ;

// 			// converting grid coordinates to transducers system of coords
// 			xs = x - xc ;
// 			ys = y - yc ;
// 			zs = z - zc ;

// 			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, xl, yl, zl) ;
// 			a = atan2(yl,xl) ;
// 			Dsq = xl*xl + yl*yl ;
// 			D = sqrt(Dsq) ;

// 			cmin = wrap_angle(-a - theta_max_trans) ;
// 			cmax = wrap_angle(-a + theta_max_trans) ;

// 			splitThetaDomain(cmin, cmax, Theta, n_xi) ; //writes in Theta and n_xi
// 			splitZDomain(zl,h, Zsq, n_upsilon) ; //write in Zsq and in n_upsilon

// 			// check and improve next loops

// 			twoRD = 2*R*D ;

// 			for (xi = 0 ; xi < n_xi ; xi++) {
// 				theta_min = Theta[xi][0] ;
// 				theta_max = Theta[xi][1] ;
// 				cos_theta_min = cos(theta_min) ;
// 				cos_theta_max = cos(theta_max) ;

// 				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
// 					Zsq_min = Zsq[upsilon][0] ;
// 					Zsq_max = Zsq[upsilon][1] ;
// 					Zmin = sqrt(Zsq_min) ;
// 					Zmax = sqrt(Zsq_max) ;

// 					constant_cosine_alpha = (Dsq + Rsq + Zsq_max) / twoRD  ;
// 					constant_cosine_beta = (Dsq + Rsq + Zsq_min) / twoRD  ;
// 					coeff_cosine = 1 / twoRD ;

// 					// if ((xi < n_xi) & (upsilon < n_upsilon) ) {
// 					rmin = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_min) ;
// 					ralpha = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_min) ;
// 					rbeta = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_max) ;
// 					rmax = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_max) ;

// 					it_min = max(min((int)floor((rmin / c - tStart) /dt ), nT-1), 0);
// 					it_alpha = max(min((int)floor((ralpha / c - tStart) / dt), nT-1), 0);
// 					it_beta = max(min((int)floor((rbeta / c - tStart) / dt), nT-1), 0);
// 					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT-1), 0) ;

// 					it_min_upsample = upsample*(it_min/upsample+1) ;
// 					it_max_upsample = upsample*(it_max/upsample) ;

// 					alpha = theta_min ;
// 					beta = theta_min ;
// 					zalpha = Zmin ;
// 					zbeta = Zmin ;
// 					I = 0. ;


//     				for (itime = it_min ; itime < (it_max_upsample-1) ; ) {

//                         // first uspample iteration, compute for each time step
//                         // then every upsample iteration compute area
//                         // last upsample iteration, compute for each time step

//                         // step = ((itime >= it_min + 2*upsample) && (itime <= it_max - 2*upsample)) ? min(upsample, it_max - 2*upsample - itime + 1) : 1;
//                         // step = (itime < it_min_upsample)
// 						// 	? min(steps_border, it_min_upsample - itime + 1)
// 						// 	: (itime >= it_max_upsample)
// 						// 		? min(steps_border, it_max-1 - itime + 1)
// 						// 		: upsample;

// 						step = computeStep(itime,it_min, it_max, it_alpha, it_beta, steps_border, steps, upsample) ;

// 						time = tStart + itime*dt ;
// 						time_next = time+step*dt ;
// 						r_next = c*time_next ;
// 						r_next_sq = r_next * r_next ;

// 							if (itime+1 < it_alpha) {
// 							 	alpha_next = theta_min ;
// 							 	zalpha_next = sqrtf(twoRD*cos_theta_min + r_next_sq - Dsq - Rsq) ;
// 							} else {
// 								cosine_alpha_next = constant_cosine_alpha - coeff_cosine * r_next_sq ;
// 								cosine_alpha_next = max(min(cosine_alpha_next, 1.),-1.) ; //with the choice of time, cosine should be in [-1,1] up to rounding errors
// 								alpha_next = acosf(cosine_alpha_next) ;
// 								zalpha_next = Zmax ;
// 							}

// 							if (itime+1 > it_beta) {
// 							 	beta_next = theta_max ;
// 							 	zbeta_next = sqrtf(twoRD*cos_theta_max + r_next_sq - Dsq - Rsq) ;
// 							} else {
// 								cosine_beta_next = constant_cosine_beta - coeff_cosine * r_next_sq ;
// 								cosine_beta_next = max(min(cosine_beta_next, 1.),-1.) ; //with the choice of time, cosine should be in [-1,1] up to rounding errors
// 								beta_next = acosf(cosine_beta_next) ;
// 								zbeta_next = Zmin ;
// 							}


// 						I_next = 0.5*(beta_next - alpha_next)*(zbeta_next + zalpha_next) ;
// 						area = (alpha_next - alpha)*Zmax + I_next - I - (beta_next-beta)*Zmin ;

// 						value = area*Rvp / time /step ;

// 						for (istep = 0 ; istep < step ; istep++) {
// 						    atomicAdd(&s_transducer[itime+istep], value) ;
// 						}

// 						// atomicAdd(&s_transducer[itime], c*area*Rvp/(rmin + c*(itime-it_min)*dt)) ;
// 						I = I_next ;
// 						alpha = alpha_next ;
// 						beta = beta_next ;
// 						zalpha = zalpha_next ;
// 						zbeta = zbeta_next ;
// 						itime += step ;
//             		}

// 					itime = it_max - 1 ;
// 					time = tStart + itime*dt ;
// 					alpha_next = theta_max ;
// 					beta_next = theta_max ;
// 					zalpha_next = Zmax ;
// 					zbeta_next = Zmax ;
// 					I_next = 0. ;
// 					area = (alpha_next - alpha)*Zmax - I - (beta_next - beta) * Zmin ;

// 					atomicAdd(&s_transducer[itime], area*Rvp/time) ;
// 					// }
// 				}
// 			}
// 		}
// 	}
// 	}
// }

// __global__ void linkingCylinderToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
// 					int nTrans,
// 					double tStart, int nT, double dt, double c,
// 					int upsample, int steps_border,
// 					double infos_transducers[], double p[], double s[]) {

// 	int i, j, k ;

// 	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
// 	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
// 	k = blockIdx.z * blockDim.z + threadIdx.z ;

// 	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

// 		//getting info of grid point
// 		double dx, dy, dz ; // grid size
// 		double x, y, z ; // coordinates
// 		double xs, ys, zs ; // coordinates centered in transducer system
// 		double xl, yl, zl ; // coordinates of grid point in the transducer system
// 		dx = (2*Lx) / (Nx-1) ;
// 		dy = (2*Ly) / (Ny-1) ;
// 		dz = (2*Lz) / (Nz-1) ;
//         // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
// 		x = -Lx + i*dx ;
// 		y = -Ly + j*dy ;
// 		z = -Lz + k*dz ;

// 		double value = 0.0 ;
// 		double value_transducer ;
// 		double *s_transducer ;
// 		double Rvp ;
// 		double D, Dsq ;
// 		double a ;
// 		double cmin,cmax ;


// 		double xc, yc, zc ; // coordinates of transducer center
// 		double e1x, e1y, e1z ; // coordinates of the e1 vector
// 		double e2x, e2y, e2z ; // coordinates of the e2 vector
// 		double R, Rsq ; //radius
// 		double theta_max_trans; // half angle of aperture of the transducer
// 		double h ; // length of the transducer in its z-axis

// 		int n_xi, n_upsilon ;
// 		int xi, upsilon ;
// 		double Theta[2][2] ;
// 		double Zsq[2][2] ;

// 		double rmin, rmax, ralpha, rbeta ;
// 		int it_min, it_max, it_alpha, it_beta, itime ;
// 		double cosine_beta_next, cosine_alpha_next ;
// 		double alpha, beta, alpha_next, beta_next ;
// 		double zalpha, zbeta, zalpha_next, zbeta_next ;
// 		double time, time_next, r_next, r_next_sq ;
// 		double area, I, I_next ;

// 		// variable to store computations
// 		double theta_min, theta_max, cos_theta_min, cos_theta_max ;
// 		double Zsq_min, Zsq_max, Zmin, Zmax ;
// 		double constant_cosine_alpha, constant_cosine_beta ;
// 		double coeff_cosine ;
// 		double twoRD ;
// 		int it_min_upsample, it_max_upsample, step ;


// 		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {
// 			// gathering infos of the transducer
// 			xc = infos_transducers[12*id_trans] ;
// 			yc = infos_transducers[12*id_trans+1] ;
// 			zc = infos_transducers[12*id_trans+2] ;

// 			e1x = infos_transducers[12*id_trans+3] ;
// 			e1y = infos_transducers[12*id_trans+4] ;
// 			e1z = infos_transducers[12*id_trans+5] ;

// 			e2x = infos_transducers[12*id_trans+6] ;
// 			e2y = infos_transducers[12*id_trans+7] ;
// 			e2z = infos_transducers[12*id_trans+8] ;

// 			R = infos_transducers[12*id_trans+9] ;
// 			Rsq = R*R ;

// 			theta_max_trans = infos_transducers[12*id_trans+10] ;
// 			h = infos_transducers[12*id_trans+11] ; // must be 1/2 ?

// 			Rvp = R / c ;
// 			s_transducer = &s[id_trans*nT] ;

// 			// converting grid coordinates to transducers system of coords
// 			xs = x - xc ;
// 			ys = y - yc ;
// 			zs = z - zc ;

// 			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, xl, yl, zl) ;
// 			a = atan2(yl,xl) ;
// 			Dsq = xl*xl + yl*yl ;
// 			D = sqrt(Dsq) ;

// 			cmin = wrap_angle(-a - theta_max_trans) ;
// 			cmax = wrap_angle(-a + theta_max_trans) ;

// 			splitThetaDomain(cmin, cmax, Theta, n_xi) ; //writes in Theta and n_xi
// 			splitZDomain(zl,h, Zsq, n_upsilon) ; //write in Zsq and in n_upsilon

// 			twoRD = 2*R*D ;
// 			value_transducer = 0.0 ;

// 			for (xi = 0 ; xi < n_xi ; xi++) {
// 				theta_min = Theta[xi][0] ;
// 				theta_max = Theta[xi][1] ;
// 				cos_theta_min = cos(theta_min) ;
// 				cos_theta_max = cos(theta_max) ;

// 				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
// 					Zsq_min = Zsq[upsilon][0] ;
// 					Zsq_max = Zsq[upsilon][1] ;
// 					Zmin = sqrt(Zsq_min) ;
// 					Zmax = sqrt(Zsq_max) ;

// 					constant_cosine_alpha = (Dsq + Rsq + Zsq_max) / twoRD  ;
// 					constant_cosine_beta = (Dsq + Rsq + Zsq_min) / twoRD  ;
// 					coeff_cosine = 1 / twoRD ;

// 					// if ((xi < n_xi) & (upsilon < n_upsilon) ) {
// 					rmin = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_min) ;
// 					ralpha = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_min) ;
// 					rbeta = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_max) ;
// 					rmax = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_max) ;

// 					it_min = max(min((int)floor((rmin / c - tStart) /dt ), nT-1), 0);
// 					it_alpha = max(min((int)floor((ralpha / c - tStart) / dt), nT-1), 0);
// 					it_beta = max(min((int)floor((rbeta / c - tStart) / dt), nT-1), 0);
// 					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT-1), 0) ;

// 					it_min_upsample = upsample*(it_min/upsample+1) ;
// 					it_max_upsample = upsample*(it_max/upsample) ;

// 					alpha = theta_min ;
// 					beta = theta_min ;
// 					zalpha = Zmin ;
// 					zbeta = Zmin ;
// 					I = 0. ;

// 					for (itime = it_min ; itime < (it_max_upsample-1) ; ) {

// 						// step = (itime < it_min_upsample)
// 							// ? min(steps_border, it_min_upsample - itime + 1)
// 							// : (itime >= it_max_upsample)
// 								// ? min(steps_border, it_max-1 - itime + 1)
// 								// : upsample;

// 						step = computeStep(itime,it_min, it_max, it_alpha, it_beta, steps_border, upsample) ;


// 						time = tStart + itime*dt ;
// 						time_next = time+dt ;
// 						r_next = c*time_next ;
// 						r_next_sq = r_next * r_next ;

// 						if (itime+1 < it_alpha) {
// 							alpha_next = theta_min ;
// 							zalpha_next = sqrtf(twoRD*cos_theta_min + r_next_sq - Dsq - Rsq) ;
// 						} else {
// 							cosine_alpha_next = constant_cosine_alpha - coeff_cosine * r_next_sq ;
// 							cosine_alpha_next = max(min(cosine_alpha_next, 1.),-1.) ; //with the choice of time, cosine should be in [-1,1] up to rounding errors
// 							alpha_next = acosf(cosine_alpha_next) ;
// 							zalpha_next = Zmax ;
// 						}

// 						if (itime+1 > it_beta) {
// 							beta_next = theta_max ;
// 							zbeta_next = sqrtf(twoRD*cos_theta_max + r_next_sq - Dsq - Rsq) ;
// 						} else {
// 							cosine_beta_next = constant_cosine_beta - coeff_cosine * r_next_sq ;
// 							cosine_beta_next = max(min(cosine_beta_next, 1.),-1.) ; //with the choice of time, cosine should be in [-1,1] up to rounding errors
// 							beta_next = acosf(cosine_beta_next) ;
// 							zbeta_next = Zmin ;
// 						}

// 						I_next = 0.5*(beta_next - alpha_next)*(zbeta_next + zalpha_next) ;
// 						area = (alpha_next - alpha)*Zmax + I_next - I - (beta_next-beta)*Zmin ;

// 						value_transducer += s_transducer[itime] * area / time ;

// 						I = I_next ;
// 						alpha = alpha_next ;
// 						beta = beta_next ;
// 						zalpha = zalpha_next ;
// 						zbeta = zbeta_next ;
// 						itime += step ;
// 					}

// 					itime = it_max - 1 ;
// 					time = tStart + itime*dt ;
// 					alpha_next = theta_max ;
// 					beta_next = theta_max ;
// 					zalpha_next = Zmax ;
// 					zbeta_next = Zmax ;
// 					I_next = 0. ;
// 					area = (alpha_next - alpha)*Zmax - I - (beta_next - beta) * Zmin ;

// 					value_transducer += s_transducer[itime] * area / time ;
// 					// }
// 				}
// 			}

// 			value += Rvp * value_transducer ;
// 		}
// 		p[(size_t)Ny * Nz * i + (size_t)Nz * j + k] = value;
// 	}
// }

// computing coordinates of grid point (xs,ys,zs) in the local coordinate system of the transducer
// Note (xs,ys,zs) are already shifted by the center of the transducer
__device__ void gridToCylinderCoords(double xs, double ys, double zs,
										double e1x, double e1y, double e1z,
										double e2x, double e2y, double e2z,
										double &x_local, double &y_local, double &z_local) {

	// Cylinder's local y-axis (e3 = e1 x e2)
    double e3x = e1y * e2z - e1z * e2y;
    double e3y = e1z * e2x - e1x * e2z;
    double e3z = e1x * e2y - e1y * e2x;

    // Project P_relative onto the cylinder's local axes
    x_local = xs * e2x + ys * e2y + zs * e2z;
    y_local = xs * e3x + ys * e3y + zs * e3z;
    z_local = xs * e1x + ys * e1y + zs * e1z;

}
// // computing coordinates of grid point (xs,ys,zs) in the local coordinate system of the transducer
// // Note (xs,ys,zs) are already shifted by the center of the transducer
// __device__ void gridToCylinderCoords(double xs, double ys, double zs,
// 										double e1x, double e1y, double e1z,
// 										double e2x, double e2y, double e2z,
// 										double &x_local, double &y_local, double &z_local) {

// 	// Cylinder's local y-axis (e3 = e1 x e2)
//     double e3x = e1y * e2z - e1z * e2y;
//     double e3y = e1z * e2x - e1x * e2z;
//     double e3z = e1x * e2y - e1y * e2x;

//     // Project P_relative onto the cylinder's local axes
//     x_local = xs * e2x + ys * e3x + zs * e1z;
//     y_local = xs * e2y + ys * e3y + zs * e1y;
//     z_local = xs * e2z + ys * e3z + zs * e1z;

// }

__device__ void splitThetaDomain(double cmin, double cmax, double Theta[2][2], int &n_xi) {
	n_xi = 0 ;

	if ( (cmin >= 0) && (cmax > 0) ) {
		Theta[0][0] = cmin ;
		Theta[0][1] = cmax ;
		n_xi = 1 ;
	} else if ((cmin < 0) && (cmax <=0)) {
		Theta[0][0] = -cmax ;
		Theta[0][1] = -cmin ;
		n_xi = 1 ;
	} else if ( (cmin < 0)  && (cmax > 0)) {
		Theta[0][0] = 0 ;
		Theta[0][1] = -cmin ;
		Theta[1][0] = 0 ;
		Theta[1][1] = cmax ;
		n_xi = 2 ;
	} else if ((cmin > 0) && (cmax < 0)) {
		Theta[0][0] = cmin ;
		Theta[0][1] = M_PI ;
		Theta[1][0] = -cmax ;
		Theta[1][1] = M_PI ;
		n_xi = 2 ;
	}

}

__device__ void splitZDomain(double zl, double h, double Zsq[2][2], int &n_upsilon) {

	n_upsilon = 0 ;

	double zmin, zmax ;
	zmin = -h - zl ;
	zmax = h - zl ;

	if (zl >= h) {
		Zsq[0][0] = zmax*zmax ;
		Zsq[0][1] = zmin*zmin ;
		n_upsilon = 1 ;
	} else if (zl <= -h) {
		Zsq[0][0] = zmin*zmin ;
		Zsq[0][1] = zmax*zmax ;
		n_upsilon = 1 ;
	} else {
		Zsq[0][0] = 0 ;
		Zsq[0][1] = zmin*zmin ;
		Zsq[1][0] = 0 ;
		Zsq[1][1] = zmax*zmax ;
		n_upsilon = 2 ;
	}
}

// True when the arrival [rmin, rmax] / c lies entirely outside the time
// window. The clamped time indices would then put the whole patch area on an
// edge sample, or write to index -1 when the arrival ends before tStart.
__device__ __forceinline__ bool outsideWindow(double rmin, double rmax, double c,
					double tStart, double dt, int nT) {
	return (ceil((rmax / c - tStart) / dt) <= 0) || (floor((rmin / c - tStart) / dt) >= nT) ;
}

// =============================================
// === Cylindrical transducers with Elliptic ===
// =============================================

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
					double p[], double s[]
) {

	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

		//getting info of grid point
		double dx, dy, dz ; // grid size
		double x, y, z ; // coordinates
		double xs, ys, zs ; // coordinates centered in transducer system
		double xl, yl, zl ; // coordinales of grid point in the transducer system

		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double vp = p[i*Ny*Nz + j*Nz + k] ;

		if (use_sparse_optimization ? (vp > 1e-16) : true) {

		double *s_transducer ;
		double Rvp ;
		double D, Dsq ;
		double a ;
		double cmin,cmax ;


		double xc, yc, zc ; // coordinates of transducer center
		double e1x, e1y, e1z ; // coordinates of the e1 vector
		double e2x, e2y, e2z ; // coordinates of the e2 vector
		double R, Rsq ; //radius
		double theta_max_trans; // half angle of aperture of the transducer
		double h ; // length of the transducer in its z-axis

		int n_xi, n_upsilon ;
		int xi, upsilon ;
		double Theta[2][2] ;
		double Zsq[2][2] ;

		double rmin, rmax ;
		double ralpha, rbeta ;
		int it_min, it_max, it_alpha, it_beta, itime ;
		double cosine ;
		double alpha, beta, alpha_next, beta_next ;
		double cos_alpha, cos_beta, cos_alpha_next, cos_beta_next ;
		double time, time_next, r_next, r_next_sq ;
		double area, I, I_next ;

		// variable to store computations
		double theta_min, theta_max, cos_theta_min, cos_theta_max ;
		double Zsq_min, Zsq_max, Zmin, Zmax ;
		double constant_cosine_alpha, constant_cosine_beta ;
		double coeff_cosine ;
		double twoRD ;

		// variable to track steps
		int it_min_upsample, it_max_upsample, step, step_prev, istep ;
		double value, value_prev, interp_value ;


		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {
			// gathering infos of the transducer
			xc = infos_transducers[12*id_trans] ;
			yc = infos_transducers[12*id_trans+1] ;
			zc = infos_transducers[12*id_trans+2] ;

			e1x = infos_transducers[12*id_trans+3] ;
			e1y = infos_transducers[12*id_trans+4] ;
			e1z = infos_transducers[12*id_trans+5] ;

			e2x = infos_transducers[12*id_trans+6] ;
			e2y = infos_transducers[12*id_trans+7] ;
			e2z = infos_transducers[12*id_trans+8] ;

			R = infos_transducers[12*id_trans+9] ;
			Rsq = R*R ;

			theta_max_trans = infos_transducers[12*id_trans+10] ;
			h = infos_transducers[12*id_trans+11] ;

			Rvp = R * vp / c ;
			s_transducer = &s[id_trans*nT] ;

			// converting grid coordinates to transducers system of coords
			xs = x - xc ;
			ys = y - yc ;
			zs = z - zc ;

			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, xl, yl, zl) ;
			a = atan2(yl,xl) ;
			Dsq = xl*xl + yl*yl ;
			D = sqrt(Dsq) ;

			cmin = wrap_angle(-a - theta_max_trans) ;
			cmax = wrap_angle(-a + theta_max_trans) ;

			splitThetaDomain(cmin, cmax, Theta, n_xi) ; //writes in Theta and n_xi
			splitZDomain(zl,h, Zsq, n_upsilon) ; //write in Zsq and in n_upsilon

			// check and improve next loops

			twoRD = 2*R*D ;

			for (xi = 0 ; xi < n_xi ; xi++) {
				theta_min = Theta[xi][0] ;
				theta_max = Theta[xi][1] ;
				cos_theta_min = cos(theta_min) ;
				cos_theta_max = cos(theta_max) ;

				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
					Zsq_min = Zsq[upsilon][0] ;
					Zsq_max = Zsq[upsilon][1] ;
					Zmin = sqrt(Zsq_min) ;
					Zmax = sqrt(Zsq_max) ;

					constant_cosine_alpha = (Dsq + Rsq + Zsq_max) / twoRD  ;
					constant_cosine_beta = (Dsq + Rsq + Zsq_min) / twoRD  ;
					coeff_cosine = 1 / twoRD ;

				// if ((xi < n_xi) & (upsilon < n_upsilon) ) {
					rmin = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_min) ;
					rmax = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_max) ;
					ralpha = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_min) ;
                    rbeta = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_max) ;

					if (outsideWindow(rmin, rmax, c, tStart, dt, nT)) continue ;
					it_min = max(min((int)floor((rmin / c - tStart) / dt), nT-1), 0);
					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT-1), 0)  ;
					it_alpha = max(min((int)floor((ralpha / c - tStart) / dt), nT-1), 0);
					it_beta = max(min((int)floor((rbeta / c - tStart) / dt), nT-1), 0);

					it_min_upsample = upsample*(it_min/upsample+1) ;
					it_max_upsample = upsample*(it_max/upsample) ;

					alpha = theta_min ;
					beta = theta_min ;
					cos_alpha = cos_theta_min ;
					cos_beta = cos_theta_min ;
					I = 0. ;
					// value_prev = 0.0 // for linear interp
					// step_prev = 0.0 // for linear interp

					for (itime = it_min ; itime < it_max ; ) {
					    // first iterations from it_min to the next time on the coarse grid, compute every steps_border steps
                        // then every upsample iteration compute area
                        // last iterations from previous time on the coarse grid to it_max, compute every steps_border steps

						// step = (itime < it_min_upsample)
						// 		? min(steps_border, it_min_upsample - itime + 1)
						// 		: (itime >= it_max_upsample)
						// 			? min(steps_border, it_max-1 - itime + 1)
						// 			: upsample;

						step = computeStep(itime,it_min, it_max, it_alpha, it_beta, steps_border, steps, upsample) ;

						time = tStart + itime * dt;
						time_next = time + step*dt;
						r_next = c * time_next;
						r_next_sq = r_next * r_next;

						cosine = constant_cosine_alpha - coeff_cosine * r_next_sq;
						cos_alpha_next = fmaxf(fminf(cosine, cos_theta_min), cos_theta_max) ;
						alpha_next = acosf(cos_alpha_next) ;

						cosine = constant_cosine_beta - coeff_cosine * r_next_sq;
						cos_beta_next = fmaxf(fminf(cosine, cos_theta_min), cos_theta_max) ;
						beta_next = acosf(cos_beta_next) ;

						if constexpr (mode == EllipticMode::Lut) {
						    I_next =  compute_I(cos_alpha_next, cos_beta_next, twoRD, r_next_sq - Dsq - Rsq,
                                    lut, sinphi0, k0, delta_sinphi0, delta_k0, eps, Nphi, Nk) ;
						} else if constexpr (mode == EllipticMode::Exact) {
						    I_next = compute_I_exact(alpha_next, beta_next,twoRD,r_next_sq - Dsq - Rsq) ; // boost::math (slow)
						} else if constexpr (mode == EllipticMode::Trapezoidal) {
				 	        double zalpha_next = fminf(fmaxf(sqrtf(twoRD*cos_theta_min + r_next_sq - Dsq - Rsq), Zmin), Zmax) ;
							double zbeta_next = fminf(fmaxf(sqrtf(twoRD*cos_theta_max + r_next_sq - Dsq - Rsq), Zmin), Zmax) ;
						    I_next = 0.5*(beta_next - alpha_next)*(zbeta_next + zalpha_next) ;
						}

						area = (alpha_next - alpha) * Zmax + I_next - I - (beta_next - beta) * Zmin;

						// replicate the computed value on the step next values in s_transducer
						double inv_time = fast_inversion(step*time) ;
						value = area*Rvp * inv_time ;
						for (istep = 0 ; istep < step ; istep++) {
						    atomicAdd(&s_transducer[itime+istep], value) ;
						}

						// write the value : only when upsample = 1
						// atomicAdd(&s_transducer[itime], area * Rvp / time);

						// linear interpolation
						// int half_step, half_step_prev ;
						// half_step = step/2 - (1 - (step) % 2) ;
						// half_step_prev = step_prev/2  ;
						// double coeff = (value - value_prev) / ((half_step + half_step_prev + 1)) ;
						// for (istep = -half_step_prev ; istep <= half_step ; istep++) {
						//                 interp_value = value_prev + coeff * (istep + half_step_prev) ;
						//                 atomicAdd(&s_transducer[itime+istep], interp_value) ;
						// }


						I = I_next ;
						alpha = alpha_next ;
						beta = beta_next ;
						cos_alpha = cos_alpha_next ;
						cos_beta = cos_beta_next ;
						itime += step ;
						// value_prev = value ; // for linear interp
						// step_prev = step ; // for linear interp
					}

					itime = it_max - 1 ;
					time = tStart + itime*dt ;
					alpha_next = theta_max ;
					beta_next = theta_max ;
					cos_alpha_next = cos_theta_max ;
					cos_beta_next = cos_theta_max ;
					I_next = 0. ;
					area = (alpha_next - alpha)*Zmax - I - (beta_next - beta) * Zmin ;

					atomicAdd(&s_transducer[itime], area*Rvp/time) ;
				// }
				}
			}
		}
		}
	}
}
template __global__ void linkingCylinderToGrid_Elliptic<EllipticMode::Lut>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					const double* const infos_transducers,
					const double* __restrict__ lut,
					const double* __restrict__ sinphi0, const double* __restrict__  k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
					int Nphi, int Nk, double eps,
					int upsample, int steps_border, int steps,
					const bool use_sparse_optimization,
					double p[], double s[]) ;

template __global__ void linkingCylinderToGrid_Elliptic<EllipticMode::Exact>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					const double* const infos_transducers,
					const double* __restrict__ lut,
					const double* __restrict__ sinphi0, const double* __restrict__  k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
					int Nphi, int Nk, double eps,
					int upsample, int steps_border, int steps,
					const bool use_sparse_optimization,
					double p[], double s[]) ;

template __global__ void linkingCylinderToGrid_Elliptic<EllipticMode::Trapezoidal>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                    int nTrans,
                    double tStart, int nT, double dt, double c,
                    const double* const infos_transducers,
                    const double* __restrict__ lut,
                    const double* __restrict__ sinphi0, const double* __restrict__  k0,
                    const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
                    int Nphi, int Nk, double eps,
                    int upsample, int steps_border, int steps,
                    const bool use_sparse_optimization,
                    double p[], double s[]) ;

template <EllipticMode mode> __global__ void __launch_bounds__(ELLIPTIC_MAX_THREADS, 1) linkingCylinderToGridT_Elliptic(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
				int nTrans,
				double tStart, int nT, double dt, double c,
				const double* const infos_transducers,
				const double* __restrict__ lut,
				const double* __restrict__ sinphi0, const double* __restrict__  k0,
				const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
				int Nphi, int Nk, double eps,
				int upsample, int steps_border, int steps,
				double p[], double s []
) {

	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

		//getting info of grid point
		double dx, dy, dz ; // grid size
		double x, y, z ; // coordinates
		double xs, ys, zs ; // coordinates centered in transducer system
		double xl, yl, zl ; // coordinales of grid point in the transducer system

		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double value = 0.0 ;
		double value_pixel = 0.0 ;
		double value_transducer = 0.0 ;
		double *s_transducer ;
		double Rvp ;
		double D, Dsq ;
		double a ;
		double cmin,cmax ;


		double xc, yc, zc ; // coordinates of transducer center
		double e1x, e1y, e1z ; // coordinates of the e1 vector
		double e2x, e2y, e2z ; // coordinates of the e2 vector
		double R, Rsq ; //radius
		double theta_max_trans; // half angle of aperture of the transducer
		double h ; // length of the transducer in its z-axis

		int n_xi, n_upsilon ;
		int xi, upsilon ;
		double Theta[2][2] ;
		double Zsq[2][2] ;

		double rmin, rmax, ralpha, rbeta ;
		int it_min, it_max, it_alpha, it_beta, itime ;
		double cosine ;
		double alpha, beta, alpha_next, beta_next ;
		double cos_alpha, cos_beta, cos_alpha_next, cos_beta_next ;
		double time, time_next, r_next, r_next_sq ;
		double area, I, I_next ;

		// variable to store computations
		double theta_min, theta_max, cos_theta_min, cos_theta_max ;
		double Zsq_min, Zsq_max, Zmin, Zmax ;
		double constant_cosine_alpha, constant_cosine_beta ;
		double coeff_cosine ;
		double twoRD ;

		// variable to track steps
		int it_min_upsample, it_max_upsample, step, step_prev, istep ;
		// double value_prev, interp_value ; // for linear interp

		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {
			// gathering infos of the transducer
			xc = infos_transducers[12*id_trans] ;
			yc = infos_transducers[12*id_trans+1] ;
			zc = infos_transducers[12*id_trans+2] ;

			e1x = infos_transducers[12*id_trans+3] ;
			e1y = infos_transducers[12*id_trans+4] ;
			e1z = infos_transducers[12*id_trans+5] ;

			e2x = infos_transducers[12*id_trans+6] ;
			e2y = infos_transducers[12*id_trans+7] ;
			e2z = infos_transducers[12*id_trans+8] ;

			R = infos_transducers[12*id_trans+9] ;
			Rsq = R*R ;

			theta_max_trans = infos_transducers[12*id_trans+10] ;
			h = infos_transducers[12*id_trans+11] ;

			Rvp = R / c ;
			s_transducer = &s[id_trans*nT] ;

			// converting grid coordinates to transducers system of coords
			xs = x - xc ;
			ys = y - yc ;
			zs = z - zc ;

			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, xl, yl, zl) ;
			a = atan2(yl,xl) ;
			Dsq = xl*xl + yl*yl ;
			D = sqrt(Dsq) ;

			cmin = wrap_angle(-a - theta_max_trans) ;
			cmax = wrap_angle(-a + theta_max_trans) ;

			splitThetaDomain(cmin, cmax, Theta, n_xi) ; //writes in Theta and n_xi
			splitZDomain(zl,h, Zsq, n_upsilon) ; //write in Zsq and in n_upsilon

			// check and improve next loops

			twoRD = 2*R*D ;
			value_transducer = 0.0 ;

			for (xi = 0 ; xi < n_xi ; xi++) {
				theta_min = Theta[xi][0] ;
				theta_max = Theta[xi][1] ;
				cos_theta_min = cos(theta_min) ;
				cos_theta_max = cos(theta_max) ;

				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
					Zsq_min = Zsq[upsilon][0] ;
					Zsq_max = Zsq[upsilon][1] ;
					Zmin = sqrt(Zsq_min) ;
					Zmax = sqrt(Zsq_max) ;

					constant_cosine_alpha = (Dsq + Rsq + Zsq_max) / twoRD  ;
					constant_cosine_beta = (Dsq + Rsq + Zsq_min) / twoRD  ;
					coeff_cosine = 1 / twoRD ;

				// if ((xi < n_xi) & (upsilon < n_upsilon) ) {
					rmin = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_min) ;
					rmax = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_max) ;
					ralpha = sqrt(Dsq + Rsq + Zsq_max - twoRD*cos_theta_min) ;
                    rbeta = sqrt(Dsq + Rsq + Zsq_min - twoRD*cos_theta_max) ;

					if (outsideWindow(rmin, rmax, c, tStart, dt, nT)) continue ;
					it_min = max(min((int)floor((rmin / c - tStart) / dt), nT-1), 0);
					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT-1), 0)  ;
					it_alpha = max(min((int)floor((ralpha / c - tStart) / dt), nT-1), 0);
					it_beta = max(min((int)floor((rbeta / c - tStart) / dt), nT-1), 0);

					it_min_upsample = upsample*(it_min/upsample+1) ;
					it_max_upsample = upsample*(it_max/upsample) ;


					alpha = theta_min ;
					beta = theta_min ;
					cos_alpha = cos_theta_min ;
					cos_beta = cos_theta_min ;
					I = 0. ;
					// value_prev = 0.0 // for linear interp
					// step_prev = 0.0 // for linear interp

					for (itime = it_min ; itime < it_max ; ) {
					    // first iterations from it_min to the next time on the coarse grid, compute every steps_border steps
                        // then every upsample iteration compute area
                        // last iterations from previous time on the coarse grid to it_max, compute every steps_border steps

						// step = (itime < it_min_upsample)
						// 	? min(steps_border, it_min_upsample - itime + 1)
						// 	: (itime >= it_max_upsample)
						// 		? min(steps_border, it_max-1 - itime + 1)
						// 		: upsample;

						step = computeStep(itime,it_min, it_max, it_alpha, it_beta, steps_border, steps, upsample) ;

						time = tStart + itime * dt;
						time_next = time + step*dt;
						r_next = c * time_next;
						r_next_sq = r_next * r_next;

						cosine = constant_cosine_alpha - coeff_cosine * r_next_sq;
						cos_alpha_next = fmaxf(fminf(cosine, cos_theta_min), cos_theta_max) ;
						alpha_next = acosf(cos_alpha_next) ;

						cosine = constant_cosine_beta - coeff_cosine * r_next_sq;
						cos_beta_next = fmaxf(fminf(cosine, cos_theta_min), cos_theta_max) ;
						beta_next = acosf(cos_beta_next) ;

						if constexpr (mode == EllipticMode::Lut) {
						    I_next =  compute_I(cos_alpha_next, cos_beta_next, twoRD, r_next_sq - Dsq - Rsq,
                                    lut, sinphi0, k0, delta_sinphi0, delta_k0, eps, Nphi, Nk) ;
						} else if constexpr (mode == EllipticMode::Exact) {
						    I_next = compute_I_exact(alpha_next, beta_next,twoRD,r_next_sq - Dsq - Rsq) ; // boost::math (slow)
						} else if constexpr (mode == EllipticMode::Trapezoidal) {
				 	        double zalpha_next = fminf(fmaxf(sqrtf(twoRD*cos_theta_min + r_next_sq - Dsq - Rsq), Zmin), Zmax) ;
							double zbeta_next = fminf(fmaxf(sqrtf(twoRD*cos_theta_max + r_next_sq - Dsq - Rsq), Zmin), Zmax) ;
						    I_next = 0.5*(beta_next - alpha_next)*(zbeta_next + zalpha_next) ;
						}
						area = (alpha_next - alpha) * Zmax + I_next - I - (beta_next - beta) * Zmin;

						// adjoint of replication : sum
						double inv_time = fast_inversion(step*time) ;
						value = area* inv_time ;
						for (istep = 0 ; istep < step ; istep++) {
							value_transducer += s_transducer[itime+istep] * value ;
						}

						// value_transducer += s_transducer[itime+istep] * area / time / step ;
						// TODO: adjoint of linear interpolation


						I = I_next ;
						alpha = alpha_next ;
						beta = beta_next ;
						cos_alpha = cos_alpha_next ;
						cos_beta = cos_beta_next ;
						itime += step ;

						// value_prev = value ; // for linear interp
						// step_prev = step ; // for linear interp
					}

					itime = it_max - 1 ;
					time = tStart + itime*dt ;
					alpha_next = theta_max ;
					beta_next = theta_max ;
					cos_alpha_next = cos_theta_max ;
					cos_beta_next = cos_theta_max ;
					I_next = 0. ;
					area = (alpha_next - alpha)*Zmax - I - (beta_next - beta) * Zmin ;

					value_transducer += s_transducer[itime] * area / time ;
				// }
				}
			}

			value_pixel += Rvp * value_transducer ;
		}
		p[Ny * Nz * i + Nz * j + k] = value_pixel;
	}
}

template __global__ void linkingCylinderToGridT_Elliptic<EllipticMode::Lut>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					const double* const infos_transducers,
					const double* __restrict__ lut,
					const double* __restrict__ sinphi0, const double* __restrict__  k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
					int Nphi, int Nk, double eps,
					int upsample, int steps_border, int steps,
					double p[], double s[]) ;

template __global__ void linkingCylinderToGridT_Elliptic<EllipticMode::Exact>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					const double* const infos_transducers,
					const double* __restrict__ lut,
					const double* __restrict__ sinphi0, const double* __restrict__  k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
					int Nphi, int Nk, double eps,
					int upsample, int steps_border, int steps,
					double p[], double s[]) ;

template __global__ void linkingCylinderToGridT_Elliptic<EllipticMode::Trapezoidal>(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                    int nTrans,
                    double tStart, int nT, double dt, double c,
                    const double* const infos_transducers,
                    const double* __restrict__ lut,
                    const double* __restrict__ sinphi0, const double* __restrict__  k0,
                    const double* __restrict__ delta_sinphi0, const double* __restrict__  delta_k0,
                    int Nphi, int Nk, double eps,
                    int upsample, int steps_border, int steps,
                    double p[], double s[]) ;


__device__ double compute_I_exact(double alpha, double beta, double a, double b) {

    double value = 0.0 ;
    double m,sqrtm,invm, invsqrtm ;
    double Ea,Eb,Fa,Fb ;
    double arg_alpha, arg_beta ;
    double phi_alpha, phi_beta ;

    if (a+b > 0) {
        m = 2 * a / (a+b) ;
        sqrtm = sqrt(m) ;

        if ( m <= 1) {
            // Eb = boost::math::ellint_2<double,double>(sqrtm, 0.5*beta) ;
            // Ea = boost::math::ellint_2<double,double>(sqrtm, 0.5*alpha) ;
            Eb = ellint_2(sqrtm, 0.5*beta) ;
            Ea = ellint_2(sqrtm, 0.5*alpha) ;
            value = 2.0 * sqrt(a + b) * (Eb-Ea);
        } else {
            invm = 1. / m ;
            invsqrtm = 1. / sqrtm ;
            arg_beta = sqrtm * sinf(0.5*beta);
            arg_alpha = sqrtm * sinf(0.5*alpha) ;

            arg_beta = fmin(fmax(arg_beta, -1.0), 1.0);
            arg_alpha = fmin(fmax(arg_alpha, -1.0), 1.0);

            phi_beta = asinf(arg_beta) ;
            phi_alpha = asinf(arg_alpha) ;

            // Eb = boost::math::ellint_2<double,double>(invsqrtm, phi_beta) ;
            // Ea = boost::math::ellint_2<double,double>(invsqrtm, phi_alpha) ;
            // Fb = boost::math::ellint_1<double,double>(invsqrtm, phi_beta) ;
            // Fa = boost::math::ellint_1<double,double>(invsqrtm, phi_alpha) ;
            Eb = ellint_2(invsqrtm, phi_beta) ;
            Ea = ellint_2(invsqrtm, phi_alpha) ;
            Fb = ellint_1(invsqrtm, phi_beta) ;
            Fa = ellint_1(invsqrtm, phi_alpha) ;

            value = sqrtm * ( (Eb-Ea) - (1-invm)*(Fb-Fa) ) ;
            value *= 2*sqrt(a+b) ;
        }
    }

    return value ;

}


__device__ inline int computeStep(int itime, int it_min, int it_max, int it_alpha, int it_beta, int steps_border, int steps, int upsample) {
    int half_upsample = upsample / 2 ;
    int step = steps ;
    int next_window = it_max ;
    int alpha_start = it_alpha - half_upsample ;
    int alpha_end = it_alpha + half_upsample ;
    int beta_start = it_beta - half_upsample ;
    int beta_end = it_beta + half_upsample ;

    // int alpha_start = it_alpha - upsample ;
    // int alpha_end = it_alpha + upsample ;
    // int beta_start = it_beta - upsample ;
    // int beta_end = it_beta + upsample ;
    // Borders
    if ((itime < it_min + upsample) || (itime > it_max - upsample) ) {
        step = steps_border ;
    }
    // in the alpha and beta regions
    else if (((alpha_start <= itime) && (itime < alpha_end) )||
        ((beta_start <= itime) && (itime < beta_end) )) {
            step = steps_border ;
        }
    // rule to reach the broder of alpha, beta intervals
    else
    {
        if (itime < alpha_start && alpha_start < next_window)
            next_window = alpha_start;
        if (itime < beta_start && beta_start < next_window)
            next_window = beta_start;
        step = min(steps,next_window-itime+1);
    }

    step = min(step,it_max-itime+1) ;
    return step ;
}



__device__ double compute_I(double cos_alpha, double cos_beta, double a, double b,
    const double* __restrict__ lut,
    const double* __restrict__ sinphi0, const double* __restrict__ k0,
    const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
    double eps,
    int Nphi, int Nk) {

        double value = 0.0 ;
        double m,sqrtm,invm ;
        double Ea,Eb,Fa,Fb ;
        double arg_alpha, arg_beta ;

        if (a+b > 0) {
            m = 2 * a / (a+b) ;

            if ( m <= 1) {
                arg_beta = sqrtf(0.5-0.5*cos_beta) ; //sin(beta/2)
                arg_alpha = sqrtf(0.5-0.5*cos_alpha) ;
                Eb = bilinear_interp_E_lut(arg_beta, m, lut, sinphi0, k0, delta_sinphi0, delta_k0, Nphi, Nk, eps);
                Ea = bilinear_interp_E_lut(arg_alpha, m, lut, sinphi0, k0, delta_sinphi0, delta_k0, Nphi, Nk,eps);
                value = 2.0 * sqrtf(a + b) * (Eb-Ea);
            } else {
                sqrtm = sqrtf(m) ;
                invm = fast_inversion(m) ;
                arg_beta = sqrtm * sqrtf(0.5-0.5*cos_beta) ;
                arg_alpha = sqrtm * sqrtf(0.5-0.5*cos_alpha) ;

                arg_beta = fmin(fmax(arg_beta, -1.0), 1.0);
                arg_alpha = fmin(fmax(arg_alpha, -1.0), 1.0);

                bilinear_interp_EF_lut(arg_beta, invm, lut, sinphi0, k0, delta_sinphi0, delta_k0, Nphi, Nk, eps, Eb, Fb) ;
                bilinear_interp_EF_lut(arg_alpha, invm, lut, sinphi0, k0, delta_sinphi0, delta_k0, Nphi, Nk, eps, Ea, Fa) ;

                value = sqrtm * ( (Eb-Ea) - (1-invm)*(Fb-Fa) ) ;
                value *= 2*sqrtf(a+b) ;
            }
        }

        return value ;

    }


__device__ double bilinear_interp_E_lut(double sinphi, double k,
    const double* __restrict__ lut, // size (Nphi,Nk)
    const double* __restrict__ sinphi0, const double* __restrict__ k0,
    const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
    int Nphi, int Nk,
    double eps) {


        int i = floor(  (1-sqrtf(sqrtf(1-sinphi))) * (Nphi-1) ) ;
        int j = floor(  (1-sqrtf(sqrtf(1-k))) * (Nk-1) ) ;
        i = max(0,min(i,Nphi-2)) ;
        j = max(0,min(j,Nk-2)) ;

        double sinphi0_i = __ldg(&sinphi0[i]) ;
        double k0_i = __ldg(&k0[j]) ;

	double dx = (sinphi - sinphi0_i) * __ldg(&delta_sinphi0[i]) ;
    double dy = (k - k0_i) * __ldg(&delta_k0[j]) ;

	// Compute linear indices into flattened LUT
	int i00 = i * 2*Nk + 2*j;
	int i01 = i * 2*Nk + 2*(j+1);
	int i10 = (i+1) * 2*Nk + 2*j;
	int i11 = (i+1) * 2*Nk + 2*(j+1);

	// Bilinear interpolation
	return  (1 - dx) * (1 - dy) * __ldg(&lut[i00]) +
	            (1 - dx) * (dy)     * __ldg(&lut[i01]) +
                (dx)     * (1 - dy) * __ldg(&lut[i10]) +
                (dx)     * (dy)     * __ldg(&lut[i11]);

}

__device__ void bilinear_interp_EF_lut(double sinphi, double k,
					const double* __restrict__ lut, // size (Nphi,Nk)
					const double* __restrict__ sinphi0, const double* __restrict__ k0,
					const double* __restrict__ delta_sinphi0, const double* __restrict__ delta_k0,
					int Nphi, int Nk,
					double eps,
					double &valueE, double &valueF) {
	// int i = floor(  (1-sqrtf(sqrtf(1-sinphi/(1-eps)))) * (Nphi-1) ) ;
	// int j = floor(  (1-sqrtf(sqrtf(1-k/(1-eps)))) * (Nk-1) ) ;
	int i = floor(  (1-sqrtf(sqrtf(1-sinphi))) * (Nphi-1) ) ;
	int j = floor(  (1-sqrtf(sqrtf(1-k))) * (Nk-1) ) ;
	i = max(0,min(i,Nphi-2)) ;
	j = max(0,min(j,Nk-2)) ;

	double sinphi0_i = __ldg(&sinphi0[i]) ;
    double k0_i = __ldg(&k0[j]) ;

    double dx = (sinphi - sinphi0_i) * __ldg(&delta_sinphi0[i]) ;
    double dy = (k - k0_i) * __ldg(&delta_k0[j]) ;

    // Compute linear indices into flattened LUT
	int i00 = i * 2*Nk + 2*j;
	int i01 = i * 2*Nk + 2*(j+1);
	int i10 = (i+1) * 2*Nk + 2*j;
	int i11 = (i+1) * 2*Nk + 2*(j+1);

	double w00 = (1-dx)*(1-dy) ;
	double w01 = (1-dx)*dy ;
	double w10 = dx*(1-dy) ;
	double w11 = dx*dy ;

	// double2 v00 = reinterpret_cast<const double2*>(lut)[i00];
	// double2 v01 = reinterpret_cast<const double2*>(lut)[i01];
	// double2 v10 = reinterpret_cast<const double2*>(lut)[i10];
	// double2 v11 = reinterpret_cast<const double2*>(lut)[i11];

	// valueE =   w00*v00.x +  w01*v01.x + w10*v10.x +  w11*v11.x ;
	// valueF =   w00*v00.y +  w01*v01.y + w10*v10.y +  w11*v11.y ;

	// Bilinear interpolation
	valueE =   w00*__ldg(&lut[i00]) +  w01*__ldg(&lut[i01]) + w10*__ldg(&lut[i10]) +  w11*__ldg(&lut[i11]);
	valueF =   w00*__ldg(&lut[i00+1]) +  w01*__ldg(&lut[i01+1]) + w10*__ldg(&lut[i10+1]) +  w11*__ldg(&lut[i11+1]);
  //

}

// ======================================
// === Planar transducers ===
// ======================================

__device__ double compute_I_plane(double a, double x) {
    // primitive: G(a,x) = 0.5 * ( x*sqrt(a^2 - x^2) + a^2 * asin( x / a ) )
    if (a <= 0.0) return 0.0;
    double asq = a * a;
    double arg = x / a;
    // clamp to [-1,1] for numerical stability
    if (arg > 1.0) arg = 1.0;
    if (arg < -1.0) arg = -1.0;
    double s = sqrt(fmax(0.0, asq - x * x));
    double as = asin(arg);
    // return 1.0 ;
    return 0.5 * (x * s + asq * as);
}
// Planar rectangular transducer kernel (forward). The wavefront of radius
// r = c t cuts the element plane, at distance z_pl, in a circle of radius
// rho = sqrt(r^2 - z_pl^2); each time step accumulates the area of that disk
// inside the rectangle gained since the previous step.
__global__ void linkingPlanesToGrid(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					double infos_transducers[],
					int upsample, int steps_border, int steps,
					bool use_sparse_optimization,
					double p[], double s[]
) {

	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

		//getting info of grid point
		double dx, dy, dz ; // grid size
		double x, y, z ; // coordinates
		double xs, ys, zs ; // coordinates centered in transducer system
		double x_cl, y_cl, z_cl ; // coordinales of grid point in the transducer system

		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double vp = p[i*Ny*Nz + j*Nz + k] ;

		if (use_sparse_optimization ? (vp > 1e-16) : true) {

		double *s_transducer ;
		double Rvp ;

		double xc, yc, zc ; // coordinates of the center of the element face
		double e1x, e1y, e1z ; // coordinates of the e1 vector (in the face)
		double e2x, e2y, e2z ; // coordinates of the e2 vector (normal to the face)
		double w, h ; // half-extents of the face across and along e1
		double z_pl ; // distance from the grid point to the element plane

		int n_xi, n_upsilon ;
		int xi, upsilon ;
		double Xsq[2][2] ;
		double Ysq[2][2] ;

		double rmin, rmax ;
		int it_min, it_max, itime ;
		double rho_next, rho_next_sq ;
		double xalpha, xbeta, xalpha_next, xbeta_next ;
		double time, time_next, r_next, r_next_sq ;
		double area, I, I_next ;

		// variable to store computations
		double Xsq_min, Xsq_max, Xmin, Xmax ;
		double Ysq_min, Ysq_max, Ymin, Ymax ;

		// variable to track steps
		int it_min_upsample, it_max_upsample, step, istep ;
		double value = 0.0;

		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {
			// gathering infos of the transducer
			xc = infos_transducers[12*id_trans] ;
			yc = infos_transducers[12*id_trans+1] ;
			zc = infos_transducers[12*id_trans+2] ;

			e1x = infos_transducers[12*id_trans+3] ;
			e1y = infos_transducers[12*id_trans+4] ;
			e1z = infos_transducers[12*id_trans+5] ;

			e2x = infos_transducers[12*id_trans+6] ;
			e2y = infos_transducers[12*id_trans+7] ;
			e2z = infos_transducers[12*id_trans+8] ;

			w = infos_transducers[12*id_trans+10] ;
			h = infos_transducers[12*id_trans+11] ;

			Rvp = vp / c ;
			s_transducer = &s[id_trans*nT] ;

			// converting grid coordinates to transducers system of coords:
			// x_cl along the normal e2, y_cl along e3 = e1 x e2, z_cl along e1
			xs = x - xc ;
			ys = y - yc ;
			zs = z - zc ;

			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, x_cl, y_cl, z_cl) ;

			z_pl = x_cl ;
			splitZDomain(z_cl, h, Xsq, n_xi) ; //writes in Xsq and n_xi
			splitZDomain(y_cl, w, Ysq, n_upsilon) ; //write in Ysq and in n_upsilon

			for (xi = 0 ; xi < n_xi ; xi++) {
				Xsq_min = Xsq[xi][0] ;
				Xsq_max = Xsq[xi][1] ;
				Xmin = sqrt(Xsq_min) ;
				Xmax = sqrt(Xsq_max) ;

				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
					Ysq_min = Ysq[upsilon][0] ;
					Ysq_max = Ysq[upsilon][1] ;
					Ymin = sqrt(Ysq_min) ;
					Ymax = sqrt(Ysq_max) ;

					// compute limits for integration
					rmin = sqrt( Xsq_min + Ysq_min + z_pl*z_pl);
					rmax = sqrt( Xsq_max + Ysq_max + z_pl*z_pl);

					if (outsideWindow(rmin, rmax, c, tStart, dt, nT)) continue ;
					it_min = max(min((int)floor((rmin / c - tStart) / dt), nT - 1), 0);
					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT - 1), 0);
					it_min_upsample = upsample * (it_min / upsample + 1);
					it_max_upsample = upsample * (it_max / upsample);

					I = 0.0;
					xalpha = Xmin;
					xbeta = Xmin;

					for (itime = it_min ; itime < it_max ; ) {
						// first iterations from it_min to the next time on the coarse grid, compute every steps_border steps
						// then every upsample iteration compute area
						// last iterations from previous time on the coarse grid to it_max, compute every steps_border steps
						step = (itime < it_min_upsample)
							? min(steps_border, it_min_upsample - itime + 1)
							: (itime >= it_max_upsample)
								? min(steps_border, it_max - 1 - itime + 1)
								: steps;
						step = min(step, it_max - itime + 1) ; // do not step past it_max

						time = tStart + itime * dt;
						time_next = time + step*dt;
						r_next = c * time_next;
						r_next_sq = r_next * r_next;

						// compute the next squared radius
						rho_next_sq = r_next_sq - z_pl * z_pl;
						rho_next = (rho_next_sq > 0.0) ? sqrt(rho_next_sq) : 0.0;

						// X bounds: intersection between circle and vertical sides of detection window
						xalpha_next = fmax(Xmin, sqrt(fmax(0.0, rho_next_sq - Ysq_max)));
						xbeta_next  = fmin(Xmax,  sqrt(fmax(0.0, rho_next_sq - Ysq_min)));

						I_next =  compute_I_plane(rho_next, xbeta_next) - compute_I_plane(rho_next, xalpha_next) ;

						area = (xalpha_next - xalpha) * Ymax + I_next - I - (xbeta_next - xbeta) * Ymin;

						value = area * Rvp / time / step;

						for (istep = 0 ; istep < step ; istep++) {
							atomicAdd(&s_transducer[itime+istep], value) ;
						}

						I = I_next ;
						xalpha = xalpha_next ;
						xbeta = xbeta_next ;
						itime += step ;
					}

					itime = it_max - 1 ;
					time = tStart + itime*dt ;
					xalpha_next = Xmax ;
					xbeta_next = Xmax ;
					I_next = 0. ;
					area = (xalpha_next - xalpha)*Ymax - I - (xbeta_next - xbeta) * Ymin ;

					atomicAdd(&s_transducer[itime], area*Rvp / time) ;
				}
			}
		}
		}
	}
}

// Planar rectangular transducer kernel (adjoint of linkingPlanesToGrid)
__global__ void linkingPlanesToGridT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
					int nTrans,
					double tStart, int nT, double dt, double c,
					double infos_transducers[],
					int upsample, int steps_border, int steps,
					double p[], double s[])
{
	int i, j, k ;

	i = blockIdx.x * blockDim.x + threadIdx.x ; // row of grid pt
	j = blockIdx.y * blockDim.y + threadIdx.y ; // col of gird pt
	k = blockIdx.z * blockDim.z + threadIdx.z ;

	if ( (i < Nx) && (j < Ny) && (k < Nz) ) {

		//getting info of grid point
		double dx, dy, dz ; // grid size
		double x, y, z ; // coordinates
		double xs, ys, zs ; // coordinates centered in transducer system
		double x_cl, y_cl, z_cl ; // coordinales of grid point in the transducer system

		dx = (2*Lx) / (Nx) ;
		dy = (2*Ly) / (Ny) ;
		dz = (2*Lz) / (Nz) ;
        // the physical coordinates x and y of the grid point using the grid dimensions and physical dimensions.
		x = -Lx + (i+0.5)*dx ;
		y = -Ly + (j+0.5)*dy ;
		z = -Lz + (k+0.5)*dz ;

		double *s_transducer ;
		double Rvp ;

		double xc, yc, zc ; // coordinates of the center of the element face
		double e1x, e1y, e1z ; // coordinates of the e1 vector (in the face)
		double e2x, e2y, e2z ; // coordinates of the e2 vector (normal to the face)
		double w, h ; // half-extents of the face across and along e1
		double z_pl ; // distance from the grid point to the element plane

		int n_xi, n_upsilon ;
		int xi, upsilon ;
		double Xsq[2][2] ;
		double Ysq[2][2] ;

		double rmin, rmax ;
		int it_min, it_max, itime ;
		double rho_next, rho_next_sq ;
		double xalpha, xbeta, xalpha_next, xbeta_next ;
		double time, time_next, r_next, r_next_sq ;
		double area, I, I_next ;

		// variable to store computations
		double Xsq_min, Xsq_max, Xmin, Xmax ;
		double Ysq_min, Ysq_max, Ymin, Ymax ;

		// variable to track steps
		int it_min_upsample, it_max_upsample, step, istep ;
		double value = 0.0;
		double value_transducer = 0.0 ;

		for (int id_trans = 0 ; id_trans < nTrans ; id_trans++) {
			// gathering infos of the transducer
			value_transducer = 0.0 ;

			xc = infos_transducers[12*id_trans] ;
			yc = infos_transducers[12*id_trans+1] ;
			zc = infos_transducers[12*id_trans+2] ;

			e1x = infos_transducers[12*id_trans+3] ;
			e1y = infos_transducers[12*id_trans+4] ;
			e1z = infos_transducers[12*id_trans+5] ;

			e2x = infos_transducers[12*id_trans+6] ;
			e2y = infos_transducers[12*id_trans+7] ;
			e2z = infos_transducers[12*id_trans+8] ;

			w = infos_transducers[12*id_trans+10] ;
			h = infos_transducers[12*id_trans+11] ;

			Rvp = 1. / c ;
			s_transducer = &s[id_trans*nT] ;

			// converting grid coordinates to transducers system of coords:
			// x_cl along the normal e2, y_cl along e3 = e1 x e2, z_cl along e1
			xs = x - xc ;
			ys = y - yc ;
			zs = z - zc ;

			gridToCylinderCoords(xs, ys, zs, e1x, e1y, e1z, e2x, e2y, e2z, x_cl, y_cl, z_cl) ;

			z_pl = x_cl ;
			splitZDomain(z_cl, h, Xsq, n_xi) ; //writes in Xsq and n_xi
			splitZDomain(y_cl, w, Ysq, n_upsilon) ; //write in Ysq and in n_upsilon

			for (xi = 0 ; xi < n_xi ; xi++) {
				Xsq_min = Xsq[xi][0] ;
				Xsq_max = Xsq[xi][1] ;
				Xmin = sqrt(Xsq_min) ;
				Xmax = sqrt(Xsq_max) ;

				for (upsilon = 0 ; upsilon < n_upsilon ; upsilon++) {
					Ysq_min = Ysq[upsilon][0] ;
					Ysq_max = Ysq[upsilon][1] ;
					Ymin = sqrt(Ysq_min) ;
					Ymax = sqrt(Ysq_max) ;

					// compute limits for integration
					rmin = sqrt( Xsq_min + Ysq_min + z_pl*z_pl);
					rmax = sqrt( Xsq_max + Ysq_max + z_pl*z_pl);

					if (outsideWindow(rmin, rmax, c, tStart, dt, nT)) continue ;
					it_min = max(min((int)floor((rmin / c - tStart) / dt), nT - 1), 0);
					it_max = max(min((int)ceil((rmax / c - tStart) / dt), nT - 1), 0);
					it_min_upsample = upsample * (it_min / upsample + 1);
					it_max_upsample = upsample * (it_max / upsample);

					I = 0.0;
					xalpha = Xmin;
					xbeta = Xmin;

					for (itime = it_min ; itime < it_max ; ) {
						// first iterations from it_min to the next time on the coarse grid, compute every steps_border steps
						// then every upsample iteration compute area
						// last iterations from previous time on the coarse grid to it_max, compute every steps_border steps
						step = (itime < it_min_upsample)
							? min(steps_border, it_min_upsample - itime + 1)
							: (itime >= it_max_upsample)
								? min(steps_border, it_max - 1 - itime + 1)
								: steps;
						step = min(step, it_max - itime + 1) ; // do not step past it_max

						time = tStart + itime * dt;
						time_next = time + step*dt;
						r_next = c * time_next;
						r_next_sq = r_next * r_next;

						// compute the next squared radius
						rho_next_sq = r_next_sq - z_pl * z_pl;
						rho_next = (rho_next_sq > 0.0) ? sqrt(rho_next_sq) : 0.0;

						// X bounds: intersection between circle and vertical sides of detection window
						xalpha_next = fmax(Xmin, sqrt(fmax(0.0, rho_next_sq - Ysq_max)));
						xbeta_next  = fmin(Xmax,  sqrt(fmax(0.0, rho_next_sq - Ysq_min)));

						I_next =  compute_I_plane(rho_next, xbeta_next) - compute_I_plane(rho_next, xalpha_next) ;

						area = (xalpha_next - xalpha) * Ymax + I_next - I - (xbeta_next - xbeta) * Ymin;

						for (istep = 0 ; istep < step ; istep++) {
							value_transducer += s_transducer[itime+istep] * area / time / step ;
						}

						I = I_next ;
						xalpha = xalpha_next ;
						xbeta = xbeta_next ;
						itime += step ;
					}

					itime = it_max - 1 ;
					time = tStart + itime*dt ;
					xalpha_next = Xmax ;
					xbeta_next = Xmax ;
					I_next = 0. ;
					area = (xalpha_next - xalpha)*Ymax - I - (xbeta_next - xbeta) * Ymin ;

					value_transducer += s_transducer[itime] * area / time ;
				}
			}
			value += Rvp * value_transducer ;
		}
		p[Ny * Nz * i + Nz * j + k] = value;
	}
}

__device__ void gridToPlaneCoords(double xs, double ys, double zs,
					double e1x, double e1y, double e1z,
					double e2x, double e2y, double e2z,
					double &x_local, double &y_local, double &z_local) {
	// Planar's local y-axis (e3 = e1 x e2)
    double e3x = e1y * e2z - e1z * e2y;
    double e3y = e1z * e2x - e1x * e2z;
    double e3z = e1x * e2y - e1y * e2x;

    // Project P_relative onto the planar's local axes
    x_local = xs * e1x + ys * e1y + zs * e1z;
    y_local = xs * e2x + ys * e2y + zs * e2z;
    z_local = xs * e3x + ys * e3y + zs * e3z;
}

// ======================================
// === Utilities ===
// ======================================

__device__ double wrap_angle(double angle) {
	double two_pi = 2.0 * M_PI ;
	angle = angle - two_pi * floor(angle / two_pi) ;
	angle = (angle > M_PI) ? (angle - two_pi) : angle ;
	return angle ;
}

__device__ double fourth_power(double x) {
    double xx = x*x ;
    return xx*xx ;
}

__device__ inline double third_power(double x) {
    double xx = x*x ;
    return xx*x ;
}

__device__ inline double squared(double x) {
    return x*x ;
}

__device__ inline double fast_inversion(double x) {
    double inv = __frcp_rn(x);
    return inv * (2.0 - x * inv); // one NR iteration
}
