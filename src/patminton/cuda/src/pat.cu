#include "pat.h"
#include "pat_types.h"
#include "patfunctions.cuh"
#include "cudaUtility.h"
#include <cstddef>
#include <cufft.h>
#include "cufftUtility.h"


PAT::PAT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c,
		int nTrans,
		int upsample, int blockSize, double laser_pulse_variance,
		double* eir,
		bool use_sparse_optimization) {

	// saving
	Nx_ = Nx ;
	Ny_ = Ny ;
	Nz_ = Nz ;
	Lx_ = Lx ;
	Ly_ = Ly ;
	Lz_ = Lz ;
	nT_ = nT ;
	tStart_ = tStart ;
	dt_ = dt ;
	c_ = c ;
	nTransducers_ = nTrans ;
	upsampling_ = upsample ;
	blockSize_ = blockSize ;
	laser_pulse_variance_ = laser_pulse_variance ;
	use_sparse_optimization_ = use_sparse_optimization ;

	// sizes of upsampled signals and its ftt
	nTu_ = upsampling_ * nT_ ; // Upsampled number of time points.
	// dtu_ = (nT_ - 1)*dt_ / (nTu_ - 1)  ; // Upsampled time step.
	dtu_ = dt_ / upsampling_ ;
	fftNtu_ = nTu_ / 2 + 1 ;  // Number of frequency components after FFT (half plus one due to symmetry).

	// Compute spatial resolutions for the convolution kernel.
	// Must match the cell-centered voxel sampling used by the geometry
	// kernels (x = -Lx + (i+0.5)*dx), otherwise the support of the radial
	// basis function does not coincide with the voxel pitch.
	dx_ = 2 * (Lx_) / Nx_ ;
	dy_ = 2 * (Ly_) / Ny_ ;
	dz_ = 2 * (Lz_) / Nz_ ;
	scaleFFT_ = 1.0 / ((double)nTu_) ; // Scaling factor for inverse FFT to normalize results.

	// cufftHandle plan ;
	// init fft
	// HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, nTu_, CUFFT_D2Z, 1)) ;
	HANDLE_CUFFT_ERROR(cufftPlan1d(&planForwardBatch_, nTu_, CUFFT_D2Z, nTransducers_)) ;  // Forward FFT for all transducers.
	HANDLE_CUFFT_ERROR(cufftPlan1d(&planBackwardBatch_, nTu_, CUFFT_Z2D, nTransducers_)) ;  // backward FFT for all transducers.

	// init memory
	HANDLE_ERROR(cudaMalloc(&d_su_, nTu_*nTransducers_*sizeof(double))) ;
	HANDLE_ERROR(cudaMalloc(&d_fs_, fftNtu_*nTransducers_*sizeof(cufftDoubleComplex))) ;
	HANDLE_ERROR(cudaMalloc(&d_fg_, fftNtu_*sizeof(cufftDoubleComplex))) ;

	// init grid sizes
	dimBlock_ = dim3(blockSize_, blockSize_, blockSize_) ;
	dimBlock2d_ = dim3(blockSize_, blockSize_) ;
	// Main simulation grid for 3D propagation
	dimGrid_ = dim3(ceil( (float)Nx_ / (float)dimBlock_.x),
					ceil((float)Ny_ / (float)dimBlock_.y),
					ceil((float)Nz_ / (float)dimBlock_.z));
	// Grid for sensor data processing
	dimGridS_ = dim3(ceil((float)nTransducers_ / (float)dimBlock2d_.x), ceil( (float)nT_ / (float)dimBlock2d_.y ));

	// Grid for sensor upsampling
	dimGridSu_ = dim3(ceil((float)nTransducers_ / (float)dimBlock2d_.x), ceil( (float)nTu_ / (float)dimBlock2d_.y ));
	// Grid for FFT-based processing
	dimGridFsu_ = dim3(ceil((float)nTransducers_ / (float)dimBlock2d_.x), ceil( (float)fftNtu_ / (float)dimBlock2d_.y));

	// --------------------------------------------------------
	// Init conv kernel
	// --------------------------------------------------------
	const int upsampling_conv = 101 ;
    int nTu_conv = upsampling_conv * nTu_ ;
    // double dtu_conv = (nTu_ - 1)*dtu_ / (nTu_conv -1) ;
	double dtu_conv = dtu_ / (upsampling_conv) ;
	cufftHandle plan, plan_backward, plan_sub ;
	double *d_g, *d_indicator ; //convolution kernel
	cufftDoubleComplex *d_fg, *d_findicator ; // oversampled convolution kernel
	double *d_g_sub  ; // subsampled convolution kernel
	// Convolution grid
	dim3 dimGridG_conv = dim3(ceil((float)(nTu_conv) / (float)dimBlock2d_.x));
	dim3 dimGridG = dim3(ceil((float)(nTu_) / (float)dimBlock2d_.x));
	double fftNtu_conv = nTu_conv / 2 + 1 ;  // Number of frequency components after FFT (half plus one due to symmetry).
	double scaleFFT_conv = 1.0 / ((double)nTu_conv) ; // Scaling factor for inverse FFT to normalize results.

	// Allocate memory
	HANDLE_ERROR(cudaMalloc(&d_g, nTu_conv*sizeof(double))) ;
	HANDLE_ERROR(cudaMalloc(&d_indicator, nTu_conv*sizeof(double))) ;
	HANDLE_ERROR(cudaMalloc(&d_g_sub, nTu_*sizeof(double))) ;
	HANDLE_ERROR(cudaMalloc(&d_fg, fftNtu_conv*sizeof(cufftDoubleComplex))) ;
	HANDLE_ERROR(cudaMalloc(&d_findicator, fftNtu_conv*sizeof(cufftDoubleComplex))) ;

	// Create FFT plans
	HANDLE_CUFFT_ERROR(cufftPlan1d(&plan, nTu_conv, CUFFT_D2Z, 1)) ;
	HANDLE_CUFFT_ERROR(cufftPlan1d(&plan_backward, nTu_conv, CUFFT_Z2D, 1)) ;  // backward
	HANDLE_CUFFT_ERROR(cufftPlan1d(&plan_sub, nTu_, CUFFT_D2Z, 1)) ;  //  FFT

	// Initialize oversampled kernel
	initConvKernel<<<dimGridG_conv,dimBlock2d_>>>(d_g, nTu_conv, dx_, dtu_conv, c_);
	zeroingArray<<<dimGridG_conv,dimBlock2d_>>>(d_indicator, nTu_conv, 1, 1) ;
	initIndicator<<<dimGridG_conv,dimBlock2d_>>>(d_indicator, nTu_conv, upsampling_conv);

	cudaDeviceSynchronize() ;
	HANDLE_ERROR( cudaGetLastError() );

	// FFT both kernel and indicator
	HANDLE_CUFFT_ERROR(cufftExecD2Z(plan, d_g, d_fg));
	HANDLE_CUFFT_ERROR(cufftExecD2Z(plan, d_indicator, d_findicator));

	complexMultiplication1d<<<dimGridG_conv,dimBlock2d_>>>(d_fg, d_findicator, fftNtu_conv, 1) ;
	HANDLE_ERROR( cudaGetLastError() );

	if (laser_pulse_variance_ > 0 ) {
        // convolution with Gaussian
        double *d_gaussian ; //convolution kernel
        cufftDoubleComplex *d_fgaussian ;
		HANDLE_ERROR(cudaMalloc(&d_gaussian, nTu_conv*sizeof(double))) ;
		HANDLE_ERROR(cudaMalloc(&d_fgaussian, fftNtu_conv*sizeof(cufftDoubleComplex))) ;

		zeroingArray<<<dimGridG_conv,dimBlock2d_>>>(d_gaussian, nTu_conv, 1, 1) ;
		initGaussian<<<dimGridG_conv, dimBlock2d_>>>(d_gaussian, nTu_conv, dtu_conv, laser_pulse_variance_);
		cudaDeviceSynchronize() ;

		HANDLE_CUFFT_ERROR(cufftExecD2Z(plan, d_gaussian, d_fgaussian));
		complexMultiplication1d<<<dimGridG_conv,dimBlock2d_>>>(d_fg, d_fgaussian, fftNtu_conv, 1) ;

		HANDLE_ERROR( cudaGetLastError() );
		cudaFree(d_gaussian) ;
		cudaFree(d_fgaussian) ;
    }

	if (eir) {
	    double *d_eir ; // assumed to be of size nT
	    cufftDoubleComplex *d_feir ;
		cufftHandle plan_nT ;
		HANDLE_ERROR(cudaMalloc(&d_eir, nT*sizeof(double))) ;
		HANDLE_ERROR(cudaMalloc(&d_feir, fftNtu_conv*sizeof(cufftDoubleComplex))) ;

		zeroingArray<<<dimGridG_conv, dimBlock2d_ >>>(d_feir, fftNtu_conv, 1, 1) ;
		HANDLE_ERROR( cudaMemcpy(d_eir, eir, nT*sizeof(double), cudaMemcpyHostToDevice ) ) ;

		HANDLE_CUFFT_ERROR(cufftPlan1d(&plan_nT, nT, CUFFT_D2Z, 1)) ;
		HANDLE_CUFFT_ERROR(cufftExecD2Z(plan_nT, d_eir, d_feir));

		complexMultiplication1d<<<dimGridG_conv,dimBlock2d_>>>(d_fg, d_feir, fftNtu_conv, 1) ;
		HANDLE_ERROR( cudaGetLastError() );
		cudaFree(d_eir) ;
		cudaFree(d_feir) ;

		cufftDestroy(plan_nT) ;
	}

	// Transform back to time domain
	HANDLE_CUFFT_ERROR( cufftExecZ2D(plan_backward, d_fg, d_g) );
	multConstant<<<dimGridG_conv,dimBlock2d_>>>(d_g, scaleFFT_conv, nTu_conv, 1) ;

	// Subsample back to original upsampled grid
	subSampling<<<dimGridG, dimBlock2d_>>>(d_g, d_g_sub, nTu_, 1, upsampling_conv) ;
	cudaDeviceSynchronize() ;

	// FFT the final kernel (stored in d_fg_)
	HANDLE_CUFFT_ERROR(cufftExecD2Z(plan_sub, d_g_sub, d_fg_));

	// Free temporary variables
	HANDLE_ERROR( cudaFree( d_g )) ;
	HANDLE_ERROR( cudaFree( d_fg )) ;
	HANDLE_ERROR( cudaFree( d_indicator )) ;
	HANDLE_ERROR( cudaFree( d_findicator )) ;
	HANDLE_ERROR( cudaFree( d_g_sub )) ;
	cufftDestroy(plan) ;
	cufftDestroy(plan_backward) ;
	cufftDestroy(plan_sub) ;
}


void PAT::PMV(double d_p[], double d_s[]) {

	zeroingArray<<<dimGridSu_,dimBlock2d_>>>(d_su_, nTransducers_, nTu_, 1) ;
	HANDLE_ERROR( cudaGetLastError() );

	compute_signal_u(d_p) ; // stores result in d_su_
	HANDLE_ERROR( cudaGetLastError() );

	// Perform forward FFT on the upsampled sensor signals.
	HANDLE_CUFFT_ERROR( cufftExecD2Z(planForwardBatch_, d_su_, d_fs_) );

	// Multiply in the frequency domain using the precomputed convolution kernel.
	batchComplexMultiplication<<<dimGridFsu_,dimBlock2d_>>>(d_fs_, d_fg_, nTransducers_, fftNtu_) ;
	HANDLE_ERROR( cudaGetLastError() );

    // Compute the inverse FFT to return to the time domain.
	HANDLE_CUFFT_ERROR( cufftExecZ2D(planBackwardBatch_, d_fs_, d_su_) );

	subSampling<<<dimGridS_,dimBlock2d_>>>(d_su_, d_s, nTransducers_, nT_, upsampling_) ;
	multConstant<<<dimGridS_,dimBlock2d_>>>(d_s, scaleFFT_, nTransducers_, nT_) ; //scaling FFT done here to avoid useless multiplicatoins

	HANDLE_ERROR( cudaGetLastError() );


	cudaDeviceSynchronize() ;
}

void PAT::PMVT(double d_p[], double d_s[]) {

	zeroingArray<<<dimGrid_,dimBlock_>>>(d_p, Nx_, Ny_, Nz_) ;
	HANDLE_ERROR( cudaGetLastError() );

	upSampling<<<dimGridSu_,dimBlock2d_>>>(d_su_, d_s, nTransducers_, nT_, upsampling_) ;
	HANDLE_ERROR( cudaGetLastError() );

	HANDLE_CUFFT_ERROR( cufftExecD2Z(planForwardBatch_, d_su_, d_fs_) );
	batchComplexMultiplicationConj<<<dimGridFsu_,dimBlock2d_>>>(d_fs_, d_fg_, nTransducers_, fftNtu_) ;
	HANDLE_ERROR( cudaGetLastError() );
	HANDLE_CUFFT_ERROR( cufftExecZ2D(planBackwardBatch_, d_fs_, d_su_) );
	multConstant<<<dimGridSu_,dimBlock2d_>>>(d_su_, scaleFFT_, nTransducers_, nTu_) ; //scaling FFT done here to avoid useless multiplicatoins

	bp_signal_u(d_p) ;
	HANDLE_ERROR( cudaGetLastError() );

	cudaDeviceSynchronize() ;
}

PAT::~PAT() {
	// Free allocated memory
	HANDLE_ERROR( cudaFree( d_fg_ )) ;
	HANDLE_ERROR( cudaFree( d_su_ )) ;
	HANDLE_ERROR( cudaFree( d_fs_ )) ;
	cufftDestroy(planForwardBatch_) ;
	cufftDestroy(planBackwardBatch_) ;
}



//----------------------------------------------------------------------------
// --- Implementations for PAT_points (Derived Class) ---
//----------------------------------------------------------------------------

PAT_points::PAT_points(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
				int nT, double tStart, double dt, double c, double *area,
				int nTrans, int nPoints, double *locPoints,
				int upsample, int blockSize, double laser_pulse_variance,
				double* eir,
				bool use_sparse_optimization)
	: PAT(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c,
		nTrans, upsample, blockSize, laser_pulse_variance, eir, use_sparse_optimization),
		nPoints_(nPoints) {

	HANDLE_ERROR(cudaMalloc(&d_locPoints_, 3*nTransducers_*nPoints_*sizeof(double))) ;
 	// Copy point locations to GPU
	HANDLE_ERROR( cudaMemcpy(d_locPoints_, locPoints, 3*nTransducers_*nPoints_*sizeof(double), cudaMemcpyHostToDevice ) ) ;

	HANDLE_ERROR( cudaMalloc(&d_area_, nTransducers_*nPoints_*sizeof(double))) ;
	HANDLE_ERROR( cudaMemcpy(d_area_, area, nTransducers_*nPoints_*sizeof(double), cudaMemcpyHostToDevice ) ) ;

	printf("PAT_points: constructor finished. \n");
	printf("Use sparse optimization: %d \n", use_sparse_optimization_) ;
}

PAT_points::~PAT_points() {
	// Free allocated memory
	HANDLE_ERROR( cudaFree( d_locPoints_ )) ;
	HANDLE_ERROR( cudaFree( d_area_ )) ;
}

void PAT_points::compute_signal_u(double *d_p) {
	linkingPointsToGrid<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
							nPoints_, nTransducers_, tStart_,
							nTu_, dtu_, c_, d_area_, d_locPoints_,
							use_sparse_optimization_, d_p, d_su_) ;

}
void PAT_points::bp_signal_u(double *d_p) {
	linkingPointsToGridT<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
					    	nPoints_, nTransducers_, tStart_,
							nTu_, dtu_, c_, d_area_, d_locPoints_, d_p, d_su_) ;

}

//----------------------------------------------------------------------------
// --- Implementations for PAT_cylinder (Derived Class) ---
//----------------------------------------------------------------------------

PAT_cylinder::PAT_cylinder(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
				int nT, double tStart, double dt, double c,
				int nTrans, double *infos_transducers,
				int upsample, int steps_border, int steps, int blockSize,
				double laser_pulse_variance,
				double* eir,
				bool use_sparse_optimization
)
	: PAT(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c, nTrans, upsample, blockSize, laser_pulse_variance, eir, use_sparse_optimization),
	steps_border_(steps_border), steps_(steps) {

	HANDLE_ERROR(cudaMalloc(&d_infos_transducers_, 12*nTransducers_*sizeof(double))) ;
 	// Copy point locations to GPU
	HANDLE_ERROR( cudaMemcpy(d_infos_transducers_, infos_transducers, 12*nTransducers_*sizeof(double), cudaMemcpyHostToDevice ) ) ;

	printf("PAT_cylinder: constructor finished. \n");
}

PAT_cylinder::~PAT_cylinder() {
	// Free allocated memory
	HANDLE_ERROR( cudaFree( d_infos_transducers_ )) ;
}

//----------------------------------------------------------------------------
// --- implementations for pat_cylinder_arcs (derived class) ---
//---------------------------------------------------------------------

PAT_cylinder_arcs::PAT_cylinder_arcs(   int nx, int ny, int nz, double lx, double ly, double lz,
                                        int nt, double tstart, double dt, double c,
                                        int ntransducers, double *infos_transducers, // host pointer
                                        int n_arcs_per_cylinder,
                                        int upsample, int steps_border, int steps,
                                        int blockSize, double laser_pulse_variance,
                                        double* eir,
                                        bool use_sparse_optimization)
    : PAT_cylinder(nx, ny, nz, lx, ly, lz, nt, tstart, dt, c, ntransducers, infos_transducers,
        upsample, steps_border, steps, blockSize, laser_pulse_variance, eir, use_sparse_optimization),
    n_arcs_per_cylinder_(n_arcs_per_cylinder) {

    }

PAT_cylinder_arcs::~PAT_cylinder_arcs()
{
}




void PAT_cylinder_arcs::compute_signal_u(double* d_p) {
      		linkingArcsToGrid<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
								nTransducers_,
								tStart_, nTu_, dtu_, c_, d_infos_transducers_,
								n_arcs_per_cylinder_,
								upsampling_,steps_border_, steps_,
								use_sparse_optimization_,
								d_p, d_su_);
}

void PAT_cylinder_arcs::bp_signal_u(double* d_p) {
    	linkingArcsToGridT<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
								nTransducers_,
								tStart_, nTu_, dtu_, c_, d_infos_transducers_,
								n_arcs_per_cylinder_,
								upsampling_,steps_border_, steps_,
								d_p, d_su_);
}
//----------------------------------------------------------------------------
// --- Implementations for PAT_cylinder_planes (Derived Class) ---
//---------------------------------------------------------------------

PAT_cylinder_planes::PAT_cylinder_planes(   int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                                        int nT, double tStart, double dt, double c,
                                        int nTransducers, double *infos_transducers, // Host pointer
                                        int n_planes_per_cylinder,
                                        int upsample, int steps_border, int steps,
                                        int blockSize, double laser_pulse_variance,
                                        double* eir,
                                        bool use_sparse_optimization)
    : PAT_cylinder(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c, nTransducers, infos_transducers,
        upsample, steps_border, steps, blockSize, laser_pulse_variance, eir, use_sparse_optimization),
    n_planes_per_cylinder_(n_planes_per_cylinder) {

    }

PAT_cylinder_planes::~PAT_cylinder_planes()
{
}

void PAT_cylinder_planes::compute_signal_u(double* d_p) {
                linkingPlanesToGrid<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
    								nTransducers_,
    								tStart_, nTu_, dtu_, c_, d_infos_transducers_,
    								n_planes_per_cylinder_,
    								upsampling_,steps_border_, steps_,
    								use_sparse_optimization_,
    								d_p, d_su_);
}

void PAT_cylinder_planes::bp_signal_u(double* d_p) {
     	linkingPlanesToGridT<<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
																			nTransducers_,
																			tStart_, nTu_, dtu_, c_, d_infos_transducers_,
																			n_planes_per_cylinder_,
																			upsampling_,steps_border_, steps_,
																			d_p, d_su_);
}

//----------------------------------------------------------------------------
// --- Implementations for PAT_cylinder_exact (Derived Class) ---
//---------------------------------------------------------------------

PAT_cylinder_exact::PAT_cylinder_exact( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                                        int nT, double tStart, double dt, double c,
                                        int nTransducers, double *infos_transducers, // Host pointer
                                        int upsample, int steps_border, int steps,
                                        int blockSize, double laser_pulse_variance,
                                        double* eir,
                                        bool use_sparse_optimization)
    : PAT_cylinder(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c, nTransducers, infos_transducers,
        upsample, steps_border, steps, blockSize, laser_pulse_variance, eir, use_sparse_optimization)
    {
    }

PAT_cylinder_exact::~PAT_cylinder_exact()
{
}

void PAT_cylinder_exact::compute_signal_u(double* d_p) {
               linkingCylinderToGrid_Elliptic<EllipticMode::Exact><<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
                  	nTransducers_,
                    tStart_, nTu_, dtu_, c_, d_infos_transducers_,
            		nullptr,
            		nullptr, nullptr,
            		nullptr, nullptr,
            		0, 0, 0.0,
            		upsampling_,steps_border_,steps_,
            		use_sparse_optimization_,
            		d_p, d_su_);
}

void PAT_cylinder_exact::bp_signal_u(double* d_p) {
    linkingCylinderToGridT_Elliptic<EllipticMode::Exact><<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
       	nTransducers_,
         tStart_, nTu_, dtu_, c_, d_infos_transducers_,
 		nullptr,
 		nullptr, nullptr,
 		nullptr, nullptr,
 		0, 0, 0.0,
 		upsampling_,steps_border_,steps_,
 		d_p, d_su_);
}
//----------------------------------------------------------------------------
// --- Implementations for PAT_cylinder_exact (Derived Class) ---
//---------------------------------------------------------------------

PAT_cylinder_farfield::PAT_cylinder_farfield( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                                        int nT, double tStart, double dt, double c,
                                        int nTransducers, double *infos_transducers, // Host pointer
                                        int upsample, int steps_border, int steps,
                                        int blockSize, double laser_pulse_variance,
                                        double* eir,
                                        bool use_sparse_optimization)
    : PAT_cylinder(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c, nTransducers, infos_transducers,
        upsample, steps_border, steps, blockSize, laser_pulse_variance, eir, use_sparse_optimization)
    {
    }

PAT_cylinder_farfield::~PAT_cylinder_farfield()
{
}

void PAT_cylinder_farfield::compute_signal_u(double* d_p) {
               linkingCylinderToGrid_Elliptic<EllipticMode::Far_Field><<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
                  	nTransducers_,
                    tStart_, nTu_, dtu_, c_, d_infos_transducers_,
            		nullptr,
            		nullptr, nullptr,
            		nullptr, nullptr,
            		0, 0, 0.0,
            		upsampling_,steps_border_,steps_,
            		use_sparse_optimization_,
            		d_p, d_su_);
}

void PAT_cylinder_farfield::bp_signal_u(double* d_p) {
    linkingCylinderToGridT_Elliptic<EllipticMode::Far_Field><<<dimGrid_, dimBlock_>>>(Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
       	nTransducers_,
        tStart_, nTu_, dtu_, c_, d_infos_transducers_,
 		nullptr,
 		nullptr, nullptr,
 		nullptr, nullptr,
 		0, 0, 0.0,
 		upsampling_,steps_border_,steps_,
 		d_p, d_su_);
}
//----------------------------------------------------------------------------
// --- Implementations for PAT_cylinder_lut (Derived Class) ---
//--------------------------------------------------
//
PAT_cylinder_lut::PAT_cylinder_lut( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                                        int nT, double tStart, double dt, double c,
                                        int nTransducers, double *infos_transducers, // Host pointer
                                       	double *lut,
                                        double *sinphi0, double* k0,
                                        int Nphi, int Nk, double eps,
                                        int upsample, int steps_border, int steps,
                                        int blockSize, double laser_pulse_variance,
                                        double* eir,
                                        bool use_sparse_optimization)
    : PAT_cylinder(Nx, Ny, Nz, Lx, Ly, Lz, nT, tStart, dt, c, nTransducers, infos_transducers,
        upsample, steps_border, steps, blockSize, laser_pulse_variance, eir, use_sparse_optimization),
        Nphi_(Nphi), Nk_(Nk), eps_(eps)
    {
        cudaMalloc(&d_lut_, Nphi * Nk * 2 * sizeof(double));
        cudaMalloc(&d_sinphi0_, Nphi * sizeof(double));
        cudaMalloc(&d_k0_, Nk * sizeof(double));
        cudaMalloc(&d_delta_sinphi0_, (Nphi-1) * sizeof(double));
        cudaMalloc(&d_delta_k0_, (Nk-1) * sizeof(double));

        cudaMemcpy(d_lut_, lut, Nphi * Nk * 2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sinphi0_, sinphi0, Nphi * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k0_, k0, Nk * sizeof(double), cudaMemcpyHostToDevice);

        double *delta_sinphi, *delta_k ;
        delta_sinphi = new double[Nphi_-1] ;
        delta_k = new double[Nk_-1] ;

        compute_inv_increments_array(sinphi0, delta_sinphi, Nphi_) ;
        compute_inv_increments_array(k0, delta_k, Nk_) ;

        cudaMemcpy(d_delta_sinphi0_, delta_sinphi, (Nphi-1) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_delta_k0_, delta_k, (Nk-1) * sizeof(double), cudaMemcpyHostToDevice);

        delete[] delta_sinphi ;
        delete[] delta_k ;
    }

PAT_cylinder_lut::~PAT_cylinder_lut()
{
   HANDLE_ERROR( cudaFree(d_lut_)) ;
   HANDLE_ERROR( cudaFree(d_sinphi0_)) ;
   HANDLE_ERROR( cudaFree(d_k0_)) ;
   HANDLE_ERROR( cudaFree(d_delta_sinphi0_)) ;
   HANDLE_ERROR( cudaFree(d_delta_k0_)) ;
}

void PAT_cylinder_lut::compute_signal_u(double* d_p) {
		linkingCylinderToGrid_Elliptic<EllipticMode::Lut><<<dimGrid_, dimBlock_>>>(
		                Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
						nTransducers_,
						tStart_, nTu_, dtu_, c_, d_infos_transducers_,
						d_lut_,
						d_sinphi0_, d_k0_,
						d_delta_sinphi0_, d_delta_k0_,
						Nphi_, Nk_, eps_,
						upsampling_,steps_border_,steps_,
						use_sparse_optimization_,
						d_p, d_su_);
}

void PAT_cylinder_lut::bp_signal_u(double* d_p) {
  		linkingCylinderToGridT_Elliptic<EllipticMode::Lut><<<dimGrid_, dimBlock_>>>(
		                Nx_, Ny_, Nz_, Lx_, Ly_, Lz_,
						nTransducers_,
						tStart_, nTu_, dtu_, c_, d_infos_transducers_,
						d_lut_,
						d_sinphi0_, d_k0_,
						d_delta_sinphi0_, d_delta_k0_,
						Nphi_, Nk_, eps_,
						upsampling_,steps_border_,steps_,
						d_p, d_su_);
}
