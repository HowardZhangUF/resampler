# DSR on the particle-resampler benchmark

Differentiable systematic resampling (Cumlin & Chatterjee, NeurIPS 2026; [code](https://github.com/fcumlin/differentiable-systematic-resampling), [arXiv:2610.12094](https://arxiv.org/abs/2610.12094)) run through the shared harness. The adapter calls the unmodified upstream `resampling.py`. Variants: `dsr` = paper default (Morton sort, tau=0.1, transport gradient clamped to [-1,1]); `dsr_unsorted` = no Morton sort; `dsr_noclip` = default without the gradient clamp; `dsr_tau0.5` = tau=0.5; `dsr_scaled` = tau=2.5/N, a local diagnostic (not from the paper) that keeps tau*N at the paper's N=25 operating point. Baselines come from the completed `results/standard` summaries (same data, seeds and metrics). Lower is better unless noted.

Cells: 380 complete, 0 failed. Numerical trial failures: 0.

## Fixed weighted clouds (N=512, D=2)

### gaussian

| method id | sliced w1 | rff mmd2 | mean error | covariance error | estimator mse x2 |
| --- | --- | --- | --- | --- | --- |
| dsr/dsr | 0.3549 | 0.06949 | 0.006921 | 1.164 | 0.6255 |
| dsr/dsr_noclip | 0.3549 | 0.06949 | 0.006921 | 1.164 | 0.6255 |
| dsr/dsr_scaled | 0.07198 | 0.001336 | 0.002374 | 0.2768 | 0.02903 |
| dsr/dsr_tau0.5 | 0.7472 | 0.1414 | 0.02022 | 1.528 | 1.154 |
| dsr/dsr_unsorted | 0.7916 | 0.1432 | 0.007603 | 1.536 | 1.179 |
| particles/systematic | 0.0226 | 0.0001754 | 0.02299 | 0.05498 | 0.00144 |
| particles/multinomial | 0.05524 | 0.0008844 | 0.05705 | 0.1007 | 0.005077 |
| pydpf/systematic_stop | 0.0219 | 0.0001688 | 0.02191 | 0.05485 | 0.001466 |
| pydpf/soft | 0.05779 | 0.0009872 | 0.06119 | 0.109 | 0.006855 |
| torchfilter/soft | 0.05779 | 0.0009872 | 0.06119 | 0.109 | 0.006855 |
| pydpf/ot | 0.0481 | 0.0006042 | 0.003943 | 0.144 | 0.01084 |
| pfjax/ot | 0.0478 | 0.000594 | 0.0006558 | 0.1476 | 0.01097 |
| filterflow/ot | 0.08119 | 0.002175 | 3.867e-08 | 0.2617 | 0.03423 |
| custom/eigen | 0.0518 | 0.0002462 | 6.245e-17 | 1.989e-15 | 6.705e-30 |
| custom/gaussian | 0.07431 | 0.001498 | 0.05436 | 0.1069 | 0.005835 |
| custom/gumbel | 0.2777 | 0.02755 | 0.06262 | 0.7981 | 0.334 |
| variational/kl | 0.1185 | 0.005354 | 0.1344 | 0.1187 | 0.03155 |

### mixture

| method id | sliced w1 | rff mmd2 | mean error | covariance error | estimator mse x2 | mode mass error |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 0.6604 | 0.04564 | 0.03512 | 4.018 | 7.484 | 0.006172 |
| dsr/dsr_noclip | 0.6604 | 0.04564 | 0.03512 | 4.018 | 7.484 | 0.006172 |
| dsr/dsr_scaled | 0.07395 | 0.0002526 | 0.003821 | 0.3449 | 0.03829 | 0.0005898 |
| dsr/dsr_tau0.5 | 1.81 | 0.2095 | 0.0505 | 8.249 | 33.29 | 0.35 |
| dsr/dsr_unsorted | 1.993 | 0.2193 | 0.0003627 | 8.37 | 35.02 | 0.35 |
| particles/systematic | 0.8652 | 0.08493 | 1.218 | 2.773 | 9.263e-05 | 0.204 |
| particles/multinomial | 0.1041 | 0.001009 | 0.1177 | 0.2812 | 0.008733 | 0.01812 |
| pydpf/systematic_stop | 0.9077 | 0.09351 | 1.277 | 3.138 | 0.000102 | 0.214 |
| pydpf/soft | 0.09785 | 0.0008643 | 0.1082 | 0.281 | 0.009961 | 0.01614 |
| torchfilter/soft | 0.09785 | 0.0008643 | 0.1082 | 0.281 | 0.009961 | 0.01614 |
| pydpf/ot | 0.3735 | 0.008529 | 0.0425 | 1.645 | 1.468 | 0.03281 |
| pfjax/ot | 0.3638 | 0.008595 | 0.000735 | 1.792 | 1.608 | 0.01914 |
| filterflow/ot | 0.5755 | 0.02049 | 4.326e-07 | 2.61 | 3.407 | 0.0582 |
| custom/eigen | 0.6883 | 0.02257 | 1.111e-15 | 3.729e-15 | 1.602e-30 | 0.15 |
| custom/gaussian | 0.7928 | 0.02813 | 0.1088 | 0.4626 | 0.1532 | 0.02831 |
| custom/gumbel | 0.7122 | 0.0458 | 0.1658 | 4.305 | 8.079 | 0.05703 |
| variational/kl | 0.6362 | 0.0386 | 0.8954 | 0.8013 | 4.21e-05 | 0.15 |

### skewed

| method id | sliced w1 | rff mmd2 | mean error | covariance error | estimator mse x2 |
| --- | --- | --- | --- | --- | --- |
| dsr/dsr | 0.3332 | 0.05536 | 0.01428 | 1.083 | 0.5745 |
| dsr/dsr_noclip | 0.3332 | 0.05536 | 0.01428 | 1.083 | 0.5745 |
| dsr/dsr_scaled | 0.0709 | 0.000885 | 0.002774 | 0.2758 | 0.02691 |
| dsr/dsr_tau0.5 | 0.7305 | 0.1166 | 0.008332 | 1.508 | 1.111 |
| dsr/dsr_unsorted | 0.6887 | 0.1136 | 0.005663 | 1.494 | 1.091 |
| particles/systematic | 0.01546 | 5.774e-05 | 0.0111 | 0.03466 | 0.0004023 |
| particles/multinomial | 0.05423 | 0.0007913 | 0.04975 | 0.1151 | 0.004805 |
| pydpf/systematic_stop | 0.01616 | 6.189e-05 | 0.01324 | 0.03526 | 0.0004005 |
| pydpf/soft | 0.07351 | 0.001507 | 0.07528 | 0.1442 | 0.007387 |
| torchfilter/soft | 0.07351 | 0.001507 | 0.07528 | 0.1442 | 0.007387 |
| pydpf/ot | 0.05167 | 0.000479 | 0.004058 | 0.1319 | 0.008642 |
| pfjax/ot | 0.0521 | 0.0004903 | 0.0008615 | 0.1326 | 0.008688 |
| filterflow/ot | 0.08125 | 0.001926 | 2.402e-08 | 0.2539 | 0.0319 |
| custom/eigen | 0.08885 | 0.000883 | 9.859e-17 | 2.692e-15 | 5.448e-30 |
| custom/gaussian | 0.1202 | 0.00259 | 0.05404 | 0.1041 | 0.004381 |
| custom/gumbel | 0.2698 | 0.02257 | 0.08824 | 0.7673 | 0.2798 |
| variational/kl | 0.01018 | 2.406e-05 | 0.003521 | 0.02 | 0.0001534 |

Median sliced-W1 rank of each DSR variant among all benchmark methods, across every static cell:

| method id | median rank | methods per cell |
| --- | --- | --- |
| dsr/dsr | 37 | 39 |
| dsr/dsr_noclip | 37 | 39 |
| dsr/dsr_scaled | 28 | 39 |
| dsr/dsr_tau0.5 | 38.5 | 39 |
| dsr/dsr_unsorted | 38 | 39 |

## Particle filters (50 steps, ESS < 0.5N; 95% CIs cluster by observation sequence)

### linear, N=128

| method id | n | rmse | rmse ci95 half | coverage95 | posterior rmse | loglik error | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 128 | 0.5064 | 0.04703 | 0.9234 | 0.07426 | -0.1902 | 0.3739 |
| dsr/dsr_noclip | 128 | 0.5064 | 0.04703 | 0.9234 | 0.07426 | -0.1902 | 0.3739 |
| dsr/dsr_scaled | 128 | 0.5066 | 0.04566 | 0.9286 | 0.06677 | -0.2048 | 0.3808 |
| dsr/dsr_tau0.5 | 128 | 0.509 | 0.05058 | 0.9122 | 0.1005 | -0.2345 | 0.5068 |
| dsr/dsr_unsorted | 128 | 0.5097 | 0.05055 | 0.9126 | 0.1003 | -0.2648 | 0.5086 |
| particles/systematic | 128 | 0.5061 | 0.04637 | 0.9292 | 0.06753 | -0.1858 | 0.3526 |
| particles/multinomial | 128 | 0.5065 | 0.04512 | 0.9276 | 0.06923 | -0.2071 | 0.4062 |
| pydpf/systematic_stop | 128 | 0.5064 | 0.0469 | 0.9308 | 0.06778 | -0.1714 | 0.327 |
| pydpf/soft | 128 | 0.5082 | 0.04603 | 0.9274 | 0.07543 | -0.333 | 0.4185 |
| torchfilter/soft | 128 | 0.5082 | 0.04603 | 0.9274 | 0.07543 | -0.333 | 0.4185 |
| pydpf/ot | 128 | 0.5145 | 0.04344 | 0.928 | 0.06701 | -0.277 | nan |
| pfjax/ot | 128 | 0.5146 | 0.04344 | 0.93 | 0.0672 | -0.2759 | nan |
| filterflow/ot | 128 | 0.5145 | 0.04393 | 0.924 | 0.06789 | -0.2685 | nan |
| custom/eigen | 128 | 0.5069 | 0.0455 | 0.9288 | 0.06779 | -0.2297 | 0.3199 |
| custom/gaussian | 128 | 0.5072 | 0.04556 | 0.9292 | 0.07074 | -0.1434 | 0.4173 |
| custom/gumbel | 128 | 0.5144 | 0.04802 | 0.928 | 0.0793 | -0.2359 | nan |
| variational/kl | 128 | 0.507 | 0.04685 | 0.9294 | 0.06701 | -0.1453 | 0.358 |

### linear, N=512

| method id | n | rmse | rmse ci95 half | coverage95 | posterior rmse | loglik error | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 512 | 0.5049 | 0.05077 | 0.926 | 0.04423 | -0.04856 | 0.1245 |
| dsr/dsr_noclip | 512 | 0.5049 | 0.05077 | 0.926 | 0.04423 | -0.04856 | 0.1245 |
| dsr/dsr_scaled | 512 | 0.5044 | 0.04827 | 0.9306 | 0.03259 | -0.04146 | 0.1104 |
| dsr/dsr_tau0.5 | 512 | 0.5089 | 0.05453 | 0.9164 | 0.07973 | -0.1947 | 0.1602 |
| dsr/dsr_unsorted | 512 | 0.5093 | 0.05469 | 0.9158 | 0.08026 | -0.1946 | 0.1587 |
| particles/systematic | 512 | 0.5046 | 0.04775 | 0.931 | 0.03241 | -0.05083 | 0.09804 |
| particles/multinomial | 512 | 0.5043 | 0.04804 | 0.9314 | 0.03356 | -0.03606 | 0.1159 |
| pydpf/systematic_stop | 512 | 0.5045 | 0.04805 | 0.9306 | 0.03294 | -0.01962 | 0.09114 |
| pydpf/soft | 512 | 0.5048 | 0.04779 | 0.9306 | 0.03574 | -0.03281 | 0.08982 |
| torchfilter/soft | 512 | 0.5048 | 0.04779 | 0.9306 | 0.03574 | -0.03281 | 0.08982 |
| custom/eigen | 512 | 0.5048 | 0.04812 | 0.931 | 0.03215 | -0.0483 | 0.08608 |
| custom/gaussian | 512 | 0.5039 | 0.04778 | 0.9314 | 0.03352 | -0.06401 | 0.1031 |
| variational/kl | 512 | 0.5044 | 0.04816 | 0.9302 | 0.03339 | -0.04099 | 0.1048 |

### ungm, N=128

| method id | n | rmse | rmse ci95 half | coverage95 | predictive nll | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 128 | 5.153 | 0.9947 | 0.9256 | 3.234 | 1181 |
| dsr/dsr_noclip | 128 | 5.153 | 0.9947 | 0.9256 | 3.234 | 1181 |
| dsr/dsr_scaled | 128 | 4.977 | 1.026 | 0.9294 | 3.186 | 1029 |
| dsr/dsr_tau0.5 | 128 | 8.516 | 1.409 | 0.7378 | 7.064 | 5422 |
| dsr/dsr_unsorted | 128 | 7.627 | 1.197 | 0.7838 | 6.094 | 9219 |
| particles/systematic | 128 | 4.906 | 0.9289 | 0.9304 | 3.181 | 653.3 |
| particles/multinomial | 128 | 4.916 | 0.8803 | 0.9284 | 3.295 | 4174 |
| pydpf/systematic_stop | 128 | 4.881 | 0.8893 | 0.932 | 3.14 | 669.5 |
| pydpf/soft | 128 | 4.825 | 0.7488 | 0.928 | 3.135 | 438.4 |
| torchfilter/soft | 128 | 4.825 | 0.7488 | 0.928 | 3.135 | 438.4 |
| pydpf/ot | 128 | 5.416 | 1.463 | 0.914 | 3.182 | nan |
| pfjax/ot | 128 | 5.328 | 1.421 | 0.918 | 3.186 | nan |
| filterflow/ot | 128 | 5.48 | 1.493 | 0.908 | 3.203 | nan |
| custom/eigen | 128 | 5.17 | 1.039 | 0.93 | 3.166 | 809.7 |
| custom/gaussian | 128 | 4.936 | 0.8551 | 0.933 | 3.131 | 845.1 |
| custom/gumbel | 128 | 5.271 | 1.41 | 0.918 | 3.158 | nan |
| variational/kl | 128 | 4.831 | 0.8575 | 0.93 | 3.182 | 756 |

### ungm, N=512

| method id | n | rmse | rmse ci95 half | coverage95 | predictive nll | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 512 | 4.786 | 0.754 | 0.955 | 2.858 | 346.3 |
| dsr/dsr_noclip | 512 | 4.786 | 0.754 | 0.955 | 2.858 | 346.3 |
| dsr/dsr_scaled | 512 | 4.562 | 0.6678 | 0.9634 | 2.8 | 269.4 |
| dsr/dsr_tau0.5 | 512 | 8.806 | 1.765 | 0.752 | 6.214 | 6265 |
| dsr/dsr_unsorted | 512 | 8.637 | 1.519 | 0.7628 | 6.2 | 6861 |
| particles/systematic | 512 | 4.494 | 0.5496 | 0.9658 | 2.781 | 73.73 |
| particles/multinomial | 512 | 4.567 | 0.6364 | 0.964 | 2.822 | 339.1 |
| pydpf/systematic_stop | 512 | 4.512 | 0.5836 | 0.966 | 2.787 | 64.81 |
| pydpf/soft | 512 | 4.542 | 0.5879 | 0.9618 | 2.811 | 155.5 |
| torchfilter/soft | 512 | 4.542 | 0.5879 | 0.9618 | 2.811 | 155.5 |
| custom/eigen | 512 | 4.98 | 0.9941 | 0.953 | 2.885 | 456.7 |
| custom/gaussian | 512 | 4.798 | 0.6957 | 0.962 | 2.824 | 193.9 |
| variational/kl | 512 | 4.559 | 0.6333 | 0.9628 | 2.832 | 240.1 |

### volatility, N=128

| method id | n | rmse | rmse ci95 half | coverage95 | predictive nll | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 128 | 0.4127 | 0.06689 | 0.947 | 1.659 | 0.09584 |
| dsr/dsr_noclip | 128 | 0.4127 | 0.06689 | 0.947 | 1.659 | 0.09584 |
| dsr/dsr_scaled | 128 | 0.4151 | 0.06442 | 0.9622 | 1.658 | 0.08494 |
| dsr/dsr_tau0.5 | 128 | 0.4229 | 0.07096 | 0.893 | 1.663 | 0.2082 |
| dsr/dsr_unsorted | 128 | 0.423 | 0.07113 | 0.8964 | 1.663 | 0.2157 |
| particles/systematic | 128 | 0.4173 | 0.0634 | 0.9658 | 1.658 | 0.08285 |
| particles/multinomial | 128 | 0.4174 | 0.06486 | 0.9614 | 1.658 | 0.107 |
| pydpf/systematic_stop | 128 | 0.4165 | 0.06394 | 0.967 | 1.658 | 0.08059 |
| pydpf/soft | 128 | 0.419 | 0.06346 | 0.9594 | 1.657 | 0.1155 |
| torchfilter/soft | 128 | 0.419 | 0.06346 | 0.9594 | 1.657 | 0.1155 |
| pydpf/ot | 128 | 0.4181 | 0.06481 | 0.97 | 1.659 | nan |
| pfjax/ot | 128 | 0.4183 | 0.06485 | 0.97 | 1.659 | nan |
| filterflow/ot | 128 | 0.4173 | 0.065 | 0.968 | 1.659 | nan |
| custom/eigen | 128 | 0.4164 | 0.06297 | 0.9662 | 1.657 | 0.07457 |
| custom/gaussian | 128 | 0.4158 | 0.06298 | 0.9646 | 1.657 | 0.07859 |
| custom/gumbel | 128 | 0.4148 | 0.06775 | 0.946 | 1.66 | nan |
| variational/kl | 128 | 0.4174 | 0.06379 | 0.9604 | 1.658 | 0.09313 |

### volatility, N=512

| method id | n | rmse | rmse ci95 half | coverage95 | predictive nll | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 512 | 0.412 | 0.0693 | 0.954 | 1.657 | 0.02034 |
| dsr/dsr_noclip | 512 | 0.412 | 0.0693 | 0.954 | 1.657 | 0.02034 |
| dsr/dsr_scaled | 512 | 0.4148 | 0.06631 | 0.9746 | 1.656 | 0.01593 |
| dsr/dsr_tau0.5 | 512 | 0.4183 | 0.07329 | 0.907 | 1.661 | 0.08011 |
| dsr/dsr_unsorted | 512 | 0.4181 | 0.07386 | 0.9048 | 1.661 | 0.07833 |
| particles/systematic | 512 | 0.415 | 0.06599 | 0.9744 | 1.656 | 0.01796 |
| particles/multinomial | 512 | 0.4149 | 0.06643 | 0.9736 | 1.656 | 0.0193 |
| pydpf/systematic_stop | 512 | 0.4141 | 0.06599 | 0.9756 | 1.656 | 0.02008 |
| pydpf/soft | 512 | 0.4154 | 0.06645 | 0.9726 | 1.656 | 0.02581 |
| torchfilter/soft | 512 | 0.4154 | 0.06645 | 0.9726 | 1.656 | 0.02581 |
| custom/eigen | 512 | 0.4148 | 0.06626 | 0.9752 | 1.656 | 0.01391 |
| custom/gaussian | 512 | 0.4149 | 0.06553 | 0.9766 | 1.656 | 0.01971 |
| variational/kl | 512 | 0.4127 | 0.06664 | 0.9716 | 1.655 | 0.01489 |

### bearing, N=128

| method id | n | rmse | rmse ci95 half | coverage95 | post occlusion rmse | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 128 | 4.828 | 2.729 | 0.0836 | 9.25 | 546.9 |
| dsr/dsr_noclip | 128 | 4.828 | 2.729 | 0.0836 | 9.25 | 546.9 |
| dsr/dsr_scaled | 128 | 5.044 | 2.328 | 0.2986 | 9.534 | 92.74 |
| dsr/dsr_tau0.5 | 128 | 4.807 | 2.234 | 0.0234 | 9.217 | 415.1 |
| dsr/dsr_unsorted | 128 | 4.95 | 2.253 | 0.0202 | 9.52 | 562.5 |
| particles/systematic | 128 | 4.548 | 1.657 | 0.6364 | 8.437 | 4.739 |
| particles/multinomial | 128 | 4.787 | 1.635 | 0.5396 | 8.919 | 19.1 |
| pydpf/systematic_stop | 128 | 4.763 | 1.746 | 0.6492 | 8.964 | 4.651 |
| pydpf/soft | 128 | 4.799 | 1.711 | 0.5192 | 9.005 | 9.433 |
| torchfilter/soft | 128 | 4.799 | 1.711 | 0.5192 | 9.005 | 9.433 |
| pydpf/ot | 128 | 5.648 | 2.284 | 0.04 | 10.91 | nan |
| pfjax/ot | 128 | 5.559 | 2.289 | 0.042 | 10.7 | nan |
| filterflow/ot | 128 | 5.296 | 1.728 | 0.052 | 10.19 | nan |
| custom/eigen | 128 | 4.51 | 1.71 | 0.7776 | 8.283 | 0.586 |
| custom/gaussian | 128 | 4.508 | 1.737 | 0.724 | 8.302 | 1.992 |
| custom/gumbel | 128 | 5.188 | 2.52 | 0.216 | 9.894 | nan |
| variational/kl | 128 | 4.662 | 1.626 | 0.6326 | 8.758 | 7.522 |

### bearing, N=512

| method id | n | rmse | rmse ci95 half | coverage95 | post occlusion rmse | conditional loglik variance |
| --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 512 | 4.866 | 2.827 | 0.0874 | 9.309 | 301.8 |
| dsr/dsr_noclip | 512 | 4.866 | 2.827 | 0.0874 | 9.309 | 301.8 |
| dsr/dsr_scaled | 512 | 4.81 | 2.151 | 0.4942 | 9.031 | 31.03 |
| dsr/dsr_tau0.5 | 512 | 4.869 | 2.28 | 0.0216 | 9.381 | 250.9 |
| dsr/dsr_unsorted | 512 | 4.843 | 2.218 | 0.0204 | 9.334 | 338.4 |
| particles/systematic | 512 | 4.135 | 1.683 | 0.8258 | 7.606 | 0.4486 |
| particles/multinomial | 512 | 4.215 | 1.744 | 0.8094 | 7.759 | 0.4514 |
| pydpf/systematic_stop | 512 | 4.111 | 1.732 | 0.85 | 7.566 | 0.2706 |
| pydpf/soft | 512 | 4.283 | 1.767 | 0.8018 | 7.92 | 0.5432 |
| torchfilter/soft | 512 | 4.283 | 1.767 | 0.8018 | 7.92 | 0.5432 |
| custom/eigen | 512 | 4.21 | 1.596 | 0.8702 | 7.587 | 0.08255 |
| custom/gaussian | 512 | 4.361 | 1.633 | 0.8504 | 7.979 | 0.2627 |
| variational/kl | 512 | 4.196 | 1.701 | 0.8302 | 7.732 | 0.6877 |

RMSE rank of each DSR variant among the benchmark methods in the same cell, itself included (1 = lowest RMSE; methods per cell N=32: 39, N=128: 39, N=512: 34, N=2048: 34, since dense methods stop at N=128):

| method id | bearing/N32 | bearing/N128 | bearing/N512 | bearing/N2048 | linear/N32 | linear/N128 | linear/N512 | linear/N2048 | ungm/N32 | ungm/N128 | ungm/N512 | ungm/N2048 | volatility/N32 | volatility/N128 | volatility/N512 | volatility/N2048 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 7 | 34 | 34 | 34 | 13 | 13 | 32 | 33 | 38 | 33 | 32 | 33 | 2 | 1 | 1 | 1 |
| dsr/dsr_noclip | 7 | 34 | 34 | 34 | 13 | 13 | 32 | 33 | 38 | 33 | 32 | 33 | 2 | 1 | 1 | 1 |
| dsr/dsr_scaled | 6 | 35 | 34 | 34 | 14 | 21 | 20 | 29 | 33 | 30 | 27 | 18 | 2 | 3 | 23 | 27 |
| dsr/dsr_tau0.5 | 8 | 33 | 34 | 34 | 31 | 34 | 34 | 34 | 39 | 39 | 34 | 34 | 35 | 39 | 34 | 34 |
| dsr/dsr_unsorted | 2 | 35 | 34 | 34 | 34 | 34 | 34 | 34 | 39 | 39 | 34 | 34 | 35 | 39 | 34 | 34 |

## Local gradients of a resampled expectation vs the exact weighted-distribution derivative (D=2, N=128)

Bias = |mean gradient - exact|; variance over 100 seeds (10 for dense methods).

### gaussian, f=x2

| method id | gradient | gradient bias | gradient variance | squared error |
| --- | --- | --- | --- | --- |
| dsr/dsr | 0.7832 | 0.226 | 2.669e-05 | 0.05109 |
| dsr/dsr_noclip | 0.7832 | 0.226 | 2.669e-05 | 0.05109 |
| dsr/dsr_scaled | 0.6786 | 0.1214 | 0.0001347 | 0.01486 |
| dsr/dsr_tau0.5 | 0.9296 | 0.3724 | 2.646e-07 | 0.1386 |
| dsr/dsr_unsorted | 0.9313 | 0.3741 | 1.732e-07 | 0.14 |
| pydpf/systematic_stop | 0.5385 | 0.01871 | 0.01701 | 0.01719 |
| pydpf/soft | 0.1429 | 0.4143 | 0.004716 | 0.1763 |
| torchfilter/soft | 0.1429 | 0.4143 | 0.004716 | 0.1763 |
| pydpf/ot | 0.006873 | 0.5503 | 0 | 0.3029 |
| pfjax/ot | 0.5856 | 0.02838 | 0 | 0.0008056 |
| filterflow/ot | 1.004 | 0.4473 | 0 | 0.2001 |
| custom/eigen | 0.5572 | 2.22e-16 | 0 | 4.93e-32 |
| custom/gaussian | 0.5678 | 0.01063 | 0.01552 | 0.01548 |
| custom/gumbel | 0.7313 | 0.174 | 0.006168 | 0.03584 |
| cuthbert/systematic_stop | 0.54 | 0.01725 | 0.01385 | 0.01401 |
| pydpf/systematic | 0 | 0.5572 | 0 | 0.3105 |

### gaussian, f=sin

| method id | gradient | gradient bias | gradient variance | squared error |
| --- | --- | --- | --- | --- |
| dsr/dsr | 0.5389 | 0.1273 | 4.368e-06 | 0.01622 |
| dsr/dsr_noclip | 0.5389 | 0.1273 | 4.368e-06 | 0.01622 |
| dsr/dsr_scaled | 0.4358 | 0.02424 | 3.112e-06 | 0.0005906 |
| dsr/dsr_tau0.5 | 0.5643 | 0.1528 | 1.719e-07 | 0.02335 |
| dsr/dsr_unsorted | 0.5748 | 0.1632 | 3.401e-08 | 0.02664 |
| pydpf/systematic_stop | 0.4162 | 0.004615 | 0.001689 | 0.001693 |
| pydpf/soft | 0.2292 | 0.1823 | 0.001332 | 0.03457 |
| torchfilter/soft | 0.2292 | 0.1823 | 0.001332 | 0.03457 |
| pydpf/ot | 0.04289 | 0.3686 | 0 | 0.1359 |
| pfjax/ot | 0.4258 | 0.01425 | 0 | 0.0002031 |
| filterflow/ot | 0.1555 | 0.2561 | 0 | 0.06557 |
| custom/eigen | 0.4028 | 0.00871 | 0 | 7.586e-05 |
| custom/gaussian | 0.4134 | 0.001833 | 0.001872 | 0.001856 |
| custom/gumbel | 0.5105 | 0.09894 | 0.00884 | 0.01774 |
| cuthbert/systematic_stop | 0.4136 | 0.002081 | 0.001491 | 0.00148 |
| pydpf/systematic | 0 | 0.4115 | 0 | 0.1694 |

### mixture, f=x2

| method id | gradient | gradient bias | gradient variance | squared error |
| --- | --- | --- | --- | --- |
| dsr/dsr | 2.383 | 1.604 | 0.006309 | 2.579 |
| dsr/dsr_noclip | 2.383 | 1.604 | 0.006309 | 2.579 |
| dsr/dsr_scaled | 0.9787 | 0.1995 | 0.000514 | 0.04031 |
| dsr/dsr_tau0.5 | 17.63 | 16.85 | 0.00135 | 283.8 |
| dsr/dsr_unsorted | 16.37 | 15.59 | 2.389e-06 | 243 |
| pydpf/systematic_stop | 0.726 | 0.05329 | 0.1071 | 0.1088 |
| pydpf/soft | 0.2778 | 0.5015 | 0.1098 | 0.3602 |
| torchfilter/soft | 0.2778 | 0.5015 | 0.1098 | 0.3602 |
| pydpf/ot | -0.2883 | 1.068 | 0 | 1.14 |
| pfjax/ot | -7.399 | 8.178 | 0 | 66.89 |
| filterflow/ot | 3.183 | 2.404 | 0 | 5.779 |
| custom/eigen | 0.7793 | 9.104e-15 | 0 | 8.288e-29 |
| custom/gaussian | 1.158 | 0.3786 | 18.3 | 18.26 |
| custom/gumbel | 8.264 | 7.484 | 8.257 | 63.45 |
| cuthbert/systematic_stop | 0.726 | 0.05329 | 0.1071 | 0.1088 |
| pydpf/systematic | 0 | 0.7793 | 0 | 0.6072 |

### mixture, f=sin

| method id | gradient | gradient bias | gradient variance | squared error |
| --- | --- | --- | --- | --- |
| dsr/dsr | -0.1432 | 0.08006 | 5.836e-05 | 0.006467 |
| dsr/dsr_noclip | -0.1432 | 0.08006 | 5.836e-05 | 0.006467 |
| dsr/dsr_scaled | -0.2528 | 0.02957 | 6.408e-07 | 0.0008753 |
| dsr/dsr_tau0.5 | 4.533 | 4.757 | 0.0003827 | 22.63 |
| dsr/dsr_unsorted | 4.92 | 5.143 | 5.12e-08 | 26.45 |
| pydpf/systematic_stop | -0.2126 | 0.01058 | 2.226e-07 | 0.0001122 |
| pydpf/soft | -0.1086 | 0.1146 | 0.01021 | 0.02323 |
| torchfilter/soft | -0.1086 | 0.1146 | 0.01021 | 0.02323 |
| pydpf/ot | 0.1759 | 0.3991 | 0 | 0.1593 |
| pfjax/ot | 1.16 | 1.383 | 0 | 1.912 |
| filterflow/ot | -2.662 | 2.439 | 0 | 5.95 |
| custom/eigen | -4.46 | 4.236 | 0 | 17.95 |
| custom/gaussian | 0.02462 | 0.2478 | 0.2901 | 0.3486 |
| custom/gumbel | 1.113 | 1.336 | 0.2035 | 1.968 |
| cuthbert/systematic_stop | -0.2126 | 0.01058 | 2.226e-07 | 0.0001122 |
| pydpf/systematic | 0 | 0.2232 | 0 | 0.04982 |

## PF log-likelihood gradient vs exact Kalman gradient (random-walk model, theta = log(q, r))

### N=128

| method id | trials | gradient bias | gradient variance | gradient mse | loglik bias | loglik variance | forward ms | backward ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 100 | 8.766 | 11.3 | 88.04 | -2.161 | 1.135 | 55.06 | 29.58 |
| dsr/dsr_noclip | 100 | 7.296 | 7.658 | 60.82 | -2.161 | 1.135 | 37.33 | 19.15 |
| dsr/dsr_scaled | 100 | 2.39 | 7.385 | 13.02 | -0.4354 | 0.6944 | 44.23 | 23.29 |
| dsr/dsr_tau0.5 | 100 | 21.51 | 24.16 | 486.5 | -6.505 | 2.987 | 28.81 | 16.12 |
| dsr/dsr_unsorted | 100 | 21.47 | 23.65 | 484.4 | -6.34 | 2.486 | 29.83 | 19.18 |
| pydpf/systematic | 100 | 11.27 | 41.82 | 168.4 | -0.2953 | 0.6556 | 23.99 | 10.13 |
| pydpf/systematic_stop | 100 | 1.748 | 150.6 | 152.2 | -0.2953 | 0.6556 | 27.19 | 13.36 |
| pydpf/soft | 100 | 6.675 | 51.45 | 95.5 | -0.3859 | 1.081 | 34.43 | 17.76 |
| torchfilter/soft | 100 | 6.675 | 51.45 | 95.5 | -0.3859 | 1.081 | 43.14 | 14.7 |
| pydpf/ot | 10 | 14.21 | 714.6 | 844.9 | -0.4472 | 0.3098 | 609.8 | 34.68 |
| custom/eigen | 100 | 0.5272 | 5.207 | 5.433 | -0.1536 | 0.5643 | 27.6 | 25 |
| custom/gaussian | 100 | 0.8231 | 5.602 | 6.223 | -0.2781 | 0.86 | 24 | 14.59 |
| custom/gumbel | 10 | 8.148 | 20.28 | 84.64 | -2.662 | 0.7297 | 29.09 | 11.69 |

### N=512

| method id | trials | gradient bias | gradient variance | gradient mse | loglik bias | loglik variance | forward ms | backward ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dsr/dsr | 100 | 7.127 | 1.69 | 52.46 | -1.912 | 0.2388 | 184.5 | 192.4 |
| dsr/dsr_noclip | 100 | 7.096 | 1.669 | 52.01 | -1.912 | 0.2388 | 141.1 | 133.3 |
| dsr/dsr_scaled | 100 | 0.2646 | 1.328 | 1.385 | -0.06867 | 0.1275 | 164.4 | 157.2 |
| dsr/dsr_tau0.5 | 100 | 20.03 | 8.307 | 409.4 | -6.344 | 1.541 | 110.4 | 100.4 |
| dsr/dsr_unsorted | 100 | 20.49 | 8.688 | 428.4 | -6.57 | 1.632 | 121.4 | 133.1 |
| pydpf/systematic | 100 | 12.45 | 10.43 | 165.2 | -0.04578 | 0.1218 | 26.33 | 10.88 |
| pydpf/systematic_stop | 100 | 0.8362 | 26.6 | 27.03 | -0.04578 | 0.1218 | 28.71 | 13.39 |
| pydpf/soft | 100 | 5.477 | 13.78 | 43.63 | -0.05534 | 0.147 | 35.5 | 17.43 |
| torchfilter/soft | 100 | 5.477 | 13.78 | 43.63 | -0.05534 | 0.147 | 55.89 | 16.28 |
| custom/eigen | 100 | 0.06764 | 1.066 | 1.06 | -0.007648 | 0.1062 | 29.52 | 26.7 |
| custom/gaussian | 100 | 0.1476 | 1.195 | 1.205 | -0.03749 | 0.1848 | 25.27 | 15.5 |

## Stochastic-volatility scale fitting (N=128, 10 datasets)

| method id | parameter absolute error | mle reference error | score error | score rmse |
| --- | --- | --- | --- | --- |
| dsr/dsr | 0.24 | 0.08 | 1.319 | 6.985 |
| custom/eigen | 0.26 | 0.06 | -0.8041 | 2.112 |
| custom/gaussian | 0.22 | 0.06 | -0.6728 | 5.252 |
| custom/gumbel | 0.26 | 0.1 | 0.602 | 4.606 |
| cuthbert/systematic_stop | 0.22 | 0.02 | -4.491 | 7.468 |
| filterflow/ot | 0.28 | 0.08 | -0.3849 | 1.307 |
| julia/systematic | 0.28 | 0.12 | -2.871 | 7.394 |
| particles/systematic | 0.22 | 0.02 | 1.668 | 8.793 |
| pfjax/ot | 0.26 | 0.06 | -0.3394 | 1.179 |
| pydpf/ot | 0.26 | 0.06 | -0.3225 | 1.146 |
| stonesoup/systematic | 0.24 | 0.08 | 1.127 | 10.19 |
| torchfilter/soft | 0.2 | 0.08 | -2.006 | 10.14 |
| variational/kl | 0.24 | 0.04 | -1.545 | 6.566 |

## Forward time per resampling call (serial pass, CPU, 1 thread; median ms, Gaussian cloud, D=2)

| method id | N=32 | N=128 | N=512 | N=2048 |
| --- | --- | --- | --- | --- |
| dsr/dsr | 0.8988 | 1.069 | 7.652 | 201.3 |
| dsr/dsr_scaled | 0.9076 | 1.081 | 7.755 | 201 |
| particles/systematic | 0.1735 | 0.1817 | 0.2124 | 0.3183 |
| pydpf/systematic | 0.8728 | 0.7239 | 0.8026 | 0.9614 |
| pydpf/soft | 0.7313 | 0.8116 | 0.7998 | 1.058 |
| custom/eigen | 0.818 | 0.8809 | 0.9094 | 1.008 |
| custom/gumbel | 0.6637 | 0.9226 | 17.2 | 252.5 |
| pfjax/ot | 2.717 | 30.88 | 115 | 2302 |
| pydpf/ot | 19.24 | 22.84 | 193.3 | 2.35e+04 |

