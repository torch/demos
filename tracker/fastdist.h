int RBF(float *input, int ichannels, int iheight, int iwidth, 
   float *output, int numProto, float *code, float *weight, float *std);

int exponential(float *input, int ichannels, int iheight, int iwidth, 
   float *output, int numProto, float *code, float *weight, float *std);

int SMR(float *input, int ichannels, int iheight, int iwidth, 
   float *output, int numProto, float *code, float *weight, float th);
