#include "cufftUtility.h"
#include <stdio.h>
#include <cufft.h>

//---------------------------------------------------------
// Body of functions
//---------------------------------------------------------

void HandleCUFFTError( cufftResult status, const char *file, int line ) {
    if ( status != CUFFT_SUCCESS ) {
        printf("CUFFT error %d in %s at line %d \n", status, file, line) ;
        exit( EXIT_FAILURE ) ;
    }
}
