/*
 * const.h
 *
 * Constants definition file
 *
 *      Author: Pablo David Gutiérrez
 *      Author: Miguel Lastra
 *      Author: Jaume Barcardit
 *      Author: José Manuel Benítez
 *      Author: Francisco Herrera
 */

#ifndef CONST_H_
#define CONST_H_

/* GPU-SME-kNN parameters */
// Test chunk size
const int TESTCHUNK = 12*1024;
// Number of threads of the distance kenrel
const int THREADSMULT = 256*2;
// Number of distances computed by each thread of the distance kernel
const int NREP = 8;

/* GPU-Komvarov-kNN parameters */
/*// Test chunk size
const int TESTCHUNK = 2*1024;
// Number of threads of the distance kenrel
const int THREADSMULT = 256;
// Number of distances computed by each thread of the distance kernel
const int NREP = 256;*/

/* Training chunk size */
 const int TRAININGCHUNK = THREADSMULT*NREP;

/* Maximum value of k*/
 const int MAXK = 1024;

/* Maxium number of clases for center kNN */
 const int MAXCLASSES = 10;
 /* Maxium number of variables for center kNN */
 const int MAXVARIABLES = 50;
 /* Size of centers array of center kNN */
 const int MAXSIZECENTERS = sizeof(float)*MAXVARIABLES*MAXCLASSES;

 #endif /* CONST_H_ */
