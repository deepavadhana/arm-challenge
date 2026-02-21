#include <ap_int.h>

#define H 32
#define W 32
#define K 3
#define IN_CH 3
#define OUT_CH 8
#define POOL 2
#define FC_OUT 10

#define CONV_H (H-K+1)
#define CONV_W (W-K+1)

#define POOL_H (CONV_H/2)
#define POOL_W (CONV_W/2)

void cnn_accel(
    ap_int<8>* input,
    ap_int<8>* conv_weights,
    ap_int<8>* fc_weights,
    ap_int<8>* output
) {

#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=conv_weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=fc_weights offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static ap_int<16> conv_out[OUT_CH][CONV_H][CONV_W];
    static ap_int<16> pool_out[OUT_CH][POOL_H][POOL_W];
    
    // Convolution + ReLU

    for (int oc = 0; oc < OUT_CH; oc++) {
        for (int i = 0; i < CONV_H; i++) {
            for (int j = 0; j < CONV_W; j++) {

                ap_int<16> sum = 0;

                for (int ic = 0; ic < IN_CH; ic++) {
                    for (int ki = 0; ki < K; ki++) {
                        for (int kj = 0; kj < K; kj++) {

                            int in_idx =
                                ic*H*W +
                                (i+ki)*W +
                                (j+kj);

                            int w_idx =
                                oc*(IN_CH*K*K) +
                                ic*(K*K) +
                                ki*K +
                                kj;

                            sum += input[in_idx] * conv_weights[w_idx];
                        }
                    }
                }

                // ReLU
                if (sum < 0)
                    sum = 0;

                conv_out[oc][i][j] = sum;
            }
        }
    }


    //Max Pooling (2x2)

    for (int oc = 0; oc < OUT_CH; oc++) {
        for (int i = 0; i < POOL_H; i++) {
            for (int j = 0; j < POOL_W; j++) {

                ap_int<16> max_val = 0;

                for (int pi = 0; pi < POOL; pi++) {
                    for (int pj = 0; pj < POOL; pj++) {

                        ap_int<16> val =
                            conv_out[oc][i*2+pi][j*2+pj];

                        if (val > max_val)
                            max_val = val;
                    }
                }

                pool_out[oc][i][j] = max_val;
            }
        }
    }

    // Fully Connected Layer
    

    int flat_size = OUT_CH * POOL_H * POOL_W;

    for (int f = 0; f < FC_OUT; f++) {

        ap_int<32> sum = 0;

        for (int oc = 0; oc < OUT_CH; oc++) {
            for (int i = 0; i < POOL_H; i++) {
                for (int j = 0; j < POOL_W; j++) {

                    int flat_idx =
                        oc*(POOL_H*POOL_W) +
                        i*(POOL_W) +
                        j;

                    int w_idx =
                        f*flat_size + flat_idx;

                    sum += pool_out[oc][i][j] *
                           fc_weights[w_idx];
                }
            }
        }

        output[f] = (ap_int<8>)sum;
    }
}

