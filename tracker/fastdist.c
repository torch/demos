/*
   fastdist.c

   different kinds of template matching

   Jonghoon Jin 2013
*/

#include <math.h>
#include <stdio.h>
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
         }
         output[y*iwidth+x] = yi_hat;
      }
   }
   return 1;
}


int exponential(float *input, int ichannels, int iheight, int iwidth,
   float *output, int numProto, float *code, float *weight, float *std)
{
   int i,k,x,y;
   float dist, yi_hat;

   for (y=0; y<iheight; y++) {
      for (x=0; x<iwidth; x++) {
         yi_hat = 0;
         for (i=0; i<numProto; i++) {
            dist = 0;
            for (k=0; k<ichannels; k++)
               dist += abs(input[(y+k*iheight)*iwidth+x] - code[k+i*ichannels]);
            yi_hat += weight[i]*exp(-dist);
         }
         output[y*iwidth+x] = yi_hat;
      }
   }
   return 0;
}


int SMR(float *input, int ichannels, int iheight, int iwidth,
   float *output, int numProto, float *code, float *weight, float th)
{
   int i,k,x,y;
   float dist, yi_hat;

   for (y=0; y<iheight; y++) {
      for (x=0; x<iwidth; x++) {
         yi_hat = 0;
         for (i=0; i<numProto; i++) {
            for (k=0; k<ichannels; k++) {
               dist = abs(input[(y+k*iheight)*iwidth+x] - code[k+i*ichannels]);
               if (2*dist < th)    yi_hat += weight[i]*exp(-2*dist);
            }
         }
         output[y*iwidth+x] = yi_hat;
      }
   }
   return 0;
}
