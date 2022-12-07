// MIT License

// Copyright (c) Microsoft Corporation and SuperHacker UEFI.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void myBioCell_forward_kernel(const float* input, const float* weight, float* output, const int Neuros, const int InputDim) 
{
	//Here InputDim == NumberOfSynapses
	const int CellID = threadIdx.x;
	const int BatchID = blockIdx.x;
	const float *myWeightBase = weight + CellID * InputDim;
	const float *myInputBase = input + BatchID * InputDim;
	float *myOutput = output + BatchID * Neuros + CellID;
	
	*myOutput = 0.0;

	for(int i=0; i<InputDim; i++)
	{
		*myOutput += myWeightBase[i] * myInputBase[i];
	}

	return;
}

std::vector<torch::Tensor> mybiocell_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor consume)
{
    const int Batchsize = input.size(0);
    const int InputDim = input.size(1);
    const int Neuros = weights.size(0);

    auto output = torch::zeros({Batchsize, Neuros}, torch::TensorOptions().device(torch::kCUDA));

    float *pGPUinput = input.data<float>();
    float *pGPUweights = weights.data<float>();
    float *pGPUoutput = output.data<float>();
    
    myBioCell_forward_kernel<<<Batchsize, Neuros>>>((float *)pGPUinput, (float *)pGPUweights, (float *)pGPUoutput, Neuros, InputDim);

    return {output};
}
