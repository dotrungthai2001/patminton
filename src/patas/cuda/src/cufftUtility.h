#ifndef __CUFFTUtility__
#define __CUFFTUtility__


#include <stdio.h>
#include <cufft.h>
#define HANDLE_CUFFT_ERROR( err ) ( HandleCUFFTError( err, __FILE__, __LINE__ ) )

/////////////////////////////////////////////////////////////////
/// This function translates the cufft error message to a
/// human readable text message
/// It can be used in a macro:
/// 	#define HANDLE_CUFFT_ERROR( err ) ( HandleCUFFTError( err, __FILE__, __LINE__ ) )
////////////////////////////////////////////////////////////////
void HandleCUFFTError( cufftResult status, const char *file, int line ) ;


#endif
