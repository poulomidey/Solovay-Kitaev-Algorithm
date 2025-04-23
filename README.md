# Solovay-Kitaev-Algorithm
ECE 491 Project

By Abdallah Daha, Poulomi Dey, Eleanor Murray, and Adrian Self

## Running:
`uv run SolovayKitaev.py`

## Results:
Using gateset `{H, T, T^\dagger}`, we obtained the following decreasing error as we increased recursion depth:
![Missing Alt Text](/results/error_plot_l10_nthru6_using_h_t_tdg.png)
```
n = 1 l = 10 distance(U, mul) = 0.017916870089199868
n = 2 l = 10 distance(U, mul) = 0.007382221900853907
n = 3 l = 10 distance(U, mul) = 0.0010558044239194118
n = 4 l = 10 distance(U, mul) = 0.00031146838500513885
n = 5 l = 10 distance(U, mul) = 1.5168473060635768e-05
n = 6 l = 10 distance(U, mul) = 2.0297672446653485e-07
```

Using gateset `{H, T}`, we obtained the following decreasing error as we increased recursion depth:
![Missing Alt Text](/results/error_plot_l10_nthru6_using_h_t.png)
```
n = 1 l = 10 distance(U, mul) = 0.017916870089199868
n = 2 l = 10 distance(U, mul) = 0.007818099676896529
n = 3 l = 10 distance(U, mul) = 0.0034292954633870325
n = 4 l = 10 distance(U, mul) = 0.0031057140197034948
n = 5 l = 10 distance(U, mul) = 0.00024389304684193513
n = 6 l = 10 distance(U, mul) = 5.702814512534562e-06
```

The more limited gateset led to less accuracy and slower convergence, but in both cases the error became quite small as we let the algorithm run.
