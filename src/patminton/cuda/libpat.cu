#include "include/pat.h"
#include "pat.h"
#include "pat_types.h"

extern "C" {

	PAT* createPAT_points(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c, double *area,
		int nTrans, int nPoints, double *locPoints,
		int upsample, int blockSize, double laser_pulse_variance,
		double* eir,
		bool use_sparse_optimization) {

		return new PAT_points(Nx, Ny, Nz, Lx, Ly, Lz,
			nT, tStart, dt, c, area,
			nTrans, nPoints, locPoints, upsample, blockSize, laser_pulse_variance, eir,
			(bool)use_sparse_optimization) ;
	}

	PAT* createPAT_cylinder_trapezoidal(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c,
		int nTrans, double *infos_transducers,
		int upsample, int steps_border, int steps, int blockSize,
		double laser_pulse_variance, double* eir,
		bool use_sparse_optimization) {

		return new PAT_cylinder_trapezoidal(Nx, Ny, Nz, Lx, Ly, Lz,
			nT, tStart, dt, c,
			nTrans, infos_transducers,
			upsample, steps_border, steps, blockSize,
			laser_pulse_variance, eir,
			(bool)use_sparse_optimization);
	}

	PAT* createPAT_cylinder_exact(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c,
		int nTrans, double *infos_transducers,
		int upsample, int steps_border, int steps, int blockSize,
		double laser_pulse_variance, double* eir,
		bool use_sparse_optimization) {

		return new PAT_cylinder_exact(Nx, Ny, Nz, Lx, Ly, Lz,
			nT, tStart, dt, c,
			nTrans, infos_transducers,
			upsample, steps_border, steps, blockSize,
			laser_pulse_variance, eir,
			(bool)use_sparse_optimization);
	}

	PAT* createPAT_cylinder_lut(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c,
		int nTrans, double *infos_transducers,
		double *lut,
		double *sinphi0, double *k0,
		int Nphi, int Nk, double eps,
		int upsample, int steps_border, int steps, int blockSize,
		double laser_pulse_variance, double* eir,
		bool use_sparse_optimization) {

		return new PAT_cylinder_lut(Nx, Ny, Nz, Lx, Ly, Lz,
			nT, tStart, dt, c,
			nTrans, infos_transducers,
			lut, sinphi0, k0, Nphi, Nk, eps,
			upsample, steps_border, steps, blockSize,
			laser_pulse_variance, eir,
			(bool)use_sparse_optimization);
	}

	PAT* createPAT_plane(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz,
		int nT, double tStart, double dt, double c,
		int nTrans, double *infos_transducers,
		int upsample, int steps_border, int steps, int blockSize,
		double laser_pulse_variance, double* eir,
		bool use_sparse_optimization) {

		return new PAT_plane(Nx, Ny, Nz, Lx, Ly, Lz,
			nT, tStart, dt, c,
			nTrans, infos_transducers,
			upsample, steps_border, steps, blockSize, laser_pulse_variance, eir,
			(bool)use_sparse_optimization);
	}

	void destroyPAT(PAT *pat) {
		delete pat ;
	}

	void PMV(PAT *pat, double d_p[], double d_s[]) {
		pat->PMV(d_p, d_s) ;
	}

	void PMVT(PAT *pat, double d_p[], double d_s[]) {
		pat->PMVT(d_p, d_s) ;
	}
};
