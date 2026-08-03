#ifndef __PAT__ /// used to prevent multiple inclusions of this header file, which can cause compilation errors.
#define __PAT__

#include <cufft.h>

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
		int steps_border_ ;
		int steps_ ;


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

	protected:
    	// Still virtual methods (no implementations)
    	virtual void compute_signal_u(double* d_p) override = 0;
    	virtual void bp_signal_u(double* d_p) override = 0 ;

} ;

class PAT_cylinder_planes : public PAT_cylinder {
    protected:
        int n_planes_per_cylinder_ ;
    public:
       PAT_cylinder_planes( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                            int n_planes_per_cylinder,
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_planes() ;
    protected:
        // implementation of the cylinder discretized in planes
   	    virtual void compute_signal_u(double* d_p) override ;
       	virtual void bp_signal_u(double* d_p) override ;

} ;

class PAT_cylinder_arcs : public PAT_cylinder {
    protected:
        int n_arcs_per_cylinder_ ;
    public:
       PAT_cylinder_arcs( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                            int n_arcs_per_cylinder,
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_arcs() ;
    protected:
        // implementation of the cylinder discretized in planes
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

class PAT_cylinder_farfield : public PAT_cylinder {
    public:
       PAT_cylinder_farfield( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
                            int nT, double tStart, double dt, double c,
                            int nTransducers, double *infos_transducers, // Host pointer
                            int upsample, int steps_border, int steps,
                            int blockSize, double laser_pulse_variance,
                            double* eir,
                            bool use_sparse_optimization);
       ~PAT_cylinder_farfield() ;
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
