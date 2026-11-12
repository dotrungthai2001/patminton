#ifndef __PAT__ /// used to prevent multiple inclusions of this header file, which can cause compilation errors.
#define __PAT__

#include <cufft.h>
#include "pat_types.h"

///////////////////////////////////////////////////////////
/// Class representing the photoacoustic experiment and implements
/// the matrix-vector products to compute the signals from the initial pressure field
/// and the adjoint
///
/// To instanciate the class:
///	Nx, Ny, Nz: are the sizes of the grid in pixels
///	Lx, Ly, Lz: are the sizes of the grid in meter
///	nT: is the number of time points
///	tStart: the time where the recording starts
///	dt: the time between two time points
///	c: the speed of sound in meter per second
///
///	Moreover an array of size (nTransducers x nSensors x 3) containing the
///	location of the discretization points of each sensor should be given
///
///	Additionally, the upsampling of the time domain should be given
///	and the size of the block for the cuda kernels
///
/// Once the class is isntantiated, the matrix vector product can be computed
/// given the adresses of the the preallocated arrays
///
///
class PAT {
	protected:
		// parameters of the experiment
		int Nx_, Ny_, Nz_ ; // gird size in pixel
		double Lx_, Ly_, Lz_ ; // grid size in meter
		int nT_ ; // numer of time points
		double dt_ ; // discretization length of time
		double tStart_ ; // starting time of acquisition
		double c_ ; // speed of sound in meter per second
		double dx_, dy_, dz_ ; // spatial resolution in each direction

		// sensors locations and sizes
		int nTransducers_ ;
		double laser_pulse_variance_ ; // Variance of the Gaussian laser pulse.
		// fft plans for the PMV
		cufftHandle planForwardBatch_, planBackwardBatch_ ; // Handles for forward and backward FFT plans.
		cufftDoubleComplex *d_fg_, *d_fs_ ; // Device pointers for FFT operations.
		double scaleFFT_ ;  // Scaling factor for FFT.

		// size of the upsampled signals
		int upsampling_, nTu_, fftNtu_ ; // Parameters related to upsampling.
		double dtu_ ;  // Upsampled time discretization length.
		double *d_su_ ;  // Pointer to upsampled signal data.

		// block sizes for kernel calls
		int blockSize_ ;
		dim3 dimBlock_, dimBlock2d_, dimGrid_ ;
		dim3 dimGridS_, dimGridSu_, dimGridFsu_ ;

		bool use_sparse_optimization_ ;

	protected:
		// --- PURE VIRTUAL METHODS ---

		// Calculates the intermediate signal 'u' (before convolution with h).
		// d_p: Input pressure field (GPU, Nx*Ny*Nz)
		// Output intermediate signal (GPU, nTransducers*fftNtu_) stored in d_su_
		//          Derived class MUST zero d_su_ before accumulating.
		virtual void compute_signal_u(double* d_p) = 0;

		// Back-propagates the correlated intermediate signal 'd_su_'.
		// Input correlated signal (GPU, nTransducers*fftNtu_) stored in ds_u_
		// d_p: Output pressure field (GPU, Nx*Ny*Nz).
		//          Derived class MUST zero d_p_out before accumulating.
		virtual void bp_signal_u(double* d_p) = 0;

	public:
    	PAT(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
    		int nT, double tStart, double dt, double c,
    		int nTrans,
    		int upsample, int blockSize, double laser_pulse_variance,
            double* eir,
            bool use_sparse_optimization
        ) ;
    	virtual ~PAT() ; // Cleans up resources when the object is destroyed.

    	void PMV(double p[], double s[]) ;
    	void PMVT(double p[], double s[]) ;

		// Access to the upsampled signal buffer d_su_ (GPU, nTransducers x nTu, row-major).
		// getSignalU: copies d_su_ as it is. After PMV it holds the convolved upsampled
		//   signal WITHOUT the 1/nTu FFT scaling; after PMVT the scaled correlated signal
		//   (elliptic cylinder modes: its suffix sums over time, see PAT_cylinder::scanSignalU).
		// computeSignalU: zeroes d_su_, runs only compute_signal_u(d_p) (no convolution)
		//   and copies the result.
		void getSignalU(double d_su[]) ;
		void computeSignalU(double d_p[], double d_su[]) ;
		int getNtu() const { return nTu_ ; }
		// Copies the convolution kernel d_fg_ (GPU, fftNtu_ = nTu/2+1 complex values, i.e.
		// interleaved real / imaginary parts): unnormalized FFT of the time kernel on the
		// upsampled grid (spherical-mean derivative * laser pulse * EIR).
		void getKernel(double d_fg[]) ;

} ;

class PAT_points : public PAT {
	private:
		int nPoints_ ;
		double *d_locPoints_ ; // 3D points locations (nTransducers x nPoints x 3)
		double *d_area_ ; // Area of each point transducer
	public:
		PAT_points(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
				   int nT, double tStart, double dt, double c, double *area,
				   int nTrans, int nPoints, double *locPoints,
				   int upsample, int blockSize, double laser_pulse_variance,
					double* eir,
				   bool use_sparse_optimization);
		~PAT_points();

	protected: // Implementations of pure virtual methods from PAT_Base
		virtual void compute_signal_u(double* d_p) override;
		virtual void bp_signal_u(double* d_p) override;
} ;

class PAT_cylinder : public PAT{
	protected: // Cylinder param
		double *d_infos_transducers_ ;
		int steps_border_ ;	// <= 0: same as the steps of the voxel
		int steps_ ;
		// optional map of steps in the transducer frame (used by the elliptic kernels instead of steps_)
		int *d_steps_map_ ;
		double *d_steps_map_x_, *d_steps_map_y_, *d_steps_map_z_ ;
		int steps_map_nx_, steps_map_ny_, steps_map_nz_ ;
		StepsMap stepsMap() const ; // device view of the map (steps = nullptr if no map)
		// The elliptic kernels write the jumps of the upsampled signal (one per block of time steps)
		// instead of every bin: compute_signal_u takes the prefix sums of d_su_ after the kernel
		// (reverse = false), bp_signal_u takes its suffix sums before the kernel (reverse = true).
		void scanSignalU(bool reverse) ;
		void freeStepsMap() ;


	public:
		PAT_cylinder(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
			int nT, double tStart, double dt, double c,
			int nTransducers, double *infos_transducers, // Host pointer
			int upsample, int steps_border, int steps,
			int blockSize, double laser_pulse_variance,
			double* eir,
			bool use_sparse_optimization
		);
    	virtual ~PAT_cylinder();
		// Steps map: steps (nx, ny, nz) row-major, ints >= 1, at the nodes x (nx), y (ny), z (nz)
		// (increasing) of the local frame of the transducers (x along e2, y along e1 x e2, z along
		// e1, origin at the transducer center). Each voxel uses the steps of the nearest node, for
		// every transducer. Host pointers, copied to the device. Replaces any previous map.
		void setStepsMap(const int *steps, const double *x, const double *y, const double *z,
		                 int nx, int ny, int nz) ;

	protected:
    	// Still virtual methods (no implementations)
    	virtual void compute_signal_u(double* d_p) override = 0;
    	virtual void bp_signal_u(double* d_p) override = 0 ;

} ;

// Flat rectangular elements. PAT_cylinder only stores the 12-column element
// rows and the integration steps, which the planar model shares (its kernels
// ignore the steps map and write every bin).
class PAT_plane : public PAT_cylinder {
    public:
       PAT_plane( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                  int nT, double tStart, double dt, double c,
                  int nTransducers, double *infos_transducers, // Host pointer
                  int upsample, int steps_border, int steps,
                  int blockSize, double laser_pulse_variance,
                  double* eir,
                  bool use_sparse_optimization);
       ~PAT_plane() ;
    protected:
        virtual void compute_signal_u(double* d_p) override ;
        virtual void bp_signal_u(double* d_p) override ;

} ;

class PAT_cylinder_exact : public PAT_cylinder {
    public:
       PAT_cylinder_exact( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_exact() ;
    protected:
        // implementation of the cylinder discretized in planes
   	    virtual void compute_signal_u(double* d_p) override ;
       	virtual void bp_signal_u(double* d_p) override ;

} ;

class PAT_cylinder_trapezoidal : public PAT_cylinder {
    public:
       PAT_cylinder_trapezoidal( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_trapezoidal() ;
    protected:
        // implementation of the cylinder discretized in planes
   	    virtual void compute_signal_u(double* d_p) override ;
       	virtual void bp_signal_u(double* d_p) override ;

} ;

class PAT_cylinder_lut : public PAT_cylinder {
    protected:
        double *d_lut_ ;
		double *d_sinphi0_, *d_k0_ ;
		double *d_delta_sinphi0_, *d_delta_k0_ ;
		int Nphi_, Nk_;	// Dimensions of the LUT
  	    double eps_ ;	// Range of the LUTs
    public:
       PAT_cylinder_lut( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                         	double *lut,
                            double *sinphi0, double* k0,
                            int Nphi, int Nk, double eps,
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_lut() ;
    protected:
        // implementation of the cylinder discretized in planes
   	    virtual void compute_signal_u(double* d_p) override ;
       	virtual void bp_signal_u(double* d_p) override ;

} ;
#endif
