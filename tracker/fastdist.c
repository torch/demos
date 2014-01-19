// fastdist.c
// radial basis function
// RBF
// JH Jin 2013

#include <math.h>
#include "fastdist.h"

#define max(a,b)  ((a)>(b) ? (a) : (b))
#define abs(a)    (a) < 0 ? -(a) : (a)
#define square(a) (a)*(a)
      
int RBF(float *input, int ichannels, int iheight, int iwidth, 
   float *output, int numProto, float *code, float *weight, float *std) 
{
   int i,k,x,y;
   float dist, yi_hat, sigma;

   for (y=0; y<iheight; y++) {
      for (x=0; x<iwidth; x++) {
         yi_hat = 0;
         for (i=0; i<numProto; i++) {
            dist = 0;
            sigma = max(0.00001, std[i]);
            for (k=0; k<ichannels; k++) 
               dist += square(input[(y+k*iheight)*iwidth+x] - code[k+i*ichannels]);
            yi_hat += weight[i]*exp(-dist/(2*sigma*sigma));
//             printf("dist: %%f, std: %%f, weight: %%f, yi_hat: %%f\n", dist, std[i], weight[i], yi_hat);
         }
         output[y*iwidth+x] = yi_hat;
      }
   }
   return 1;
}
