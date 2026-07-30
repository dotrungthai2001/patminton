#ifndef __CUDA_Utility__
#define __CUDA_Utility__

#include <stdio.h>
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )  //  It takes a CUDA error code err and calls the HandleError function, 
                                                                        // passing the error code, the current file name (__FILE__), 
                                                                        // and the current line number (__LINE__). 
                                                                        // This helps in identifying where the error occurred in the source code.

/////////////////////////////////////////////////////////////////
/// This function translates the cuda error message to a
/// human readable text message
/// It can be used in a macro:
/// 	#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
////////////////////////////////////////////////////////////////
void HandleError( cudaError_t err, const char *file, int line ) ;


#endif
