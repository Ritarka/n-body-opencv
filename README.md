## Intro
This was a project I originally worked on during my 2nd year of college.

![Demo animation](nbody_simulation.gif)

I always felt this was too slow. I got tired of my python heavy job and wanted to scratch the itch of doing some performance engineering again. 

Compared to my previous implementation of of this 2D n-body, this new implementation works 3-4x faster and can simulate more celestial bodies! If I recall correctly, my old laptop would crash after just 1024 bodies, but I am able to simulate more than 10x that. I haven't bothered testing the new upper bound.

## Major Optimizations
I made several 'layers' of improvements - all of which centered around trying to make the heavily called function calcForce as fast as possible.

The first was rewriting how I stored my bodies. I initially instantiated a class for every body and added a bunch of irrelevant fields like a `vector<pair<int, int>>` denoting the path the body took, its color, velocity, and acceleration. None of these were used in the calcForce function.
All of the information I DID need, such as mass and position, I removed it from the Array of Structures (AoS) data structure and rewrote it as a Structure of Arrays (SoA).

This helped make the memory accesses a lot smoother. The generated assembly didn't have to awkwardly navigate through my `Body` struct, instead all of the relevant information was places 8 bytes apart in a neat line. LIkely the branch predictor and caches were a lot happier too.


With this done, I used perf to idenity that my inverse square root operation was taking quite a lot of time. I remembered learning that some processors implement it as an instruction, so I used inline assembly to optimize that.

I then started looking into a SIMD implementation of this. The "-O3 -march=native" wasn't quite doing everything I would have liked it to do, so I started implementing it on my own. The inline assembly soon became quite cumbersome, so I started using C++ compiler intrinsics, which was a lot easier to read and maintain.

Interestingly, I found that using 512-bit zmm registers was actually slower than using the 256-bit ymm registers. Based on [Downs](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html) and [Krasnov](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/), it seems like operations with zmm registers operate at a lower frequency due to power considerations. The frequency and voltage switching end up taking in the order of microseconds, and the underlying operations run at 1/4th the speed.


## Minor Optimizations
After that, it was a series of smaller optimizations - all of which made my program faster, but did not bring about an Nx speedup.

I noticed that I could combine a multiply and add operation into a single faster fused multiple-add operation. I also noticed that all of my ymm registers were not being used, so I unrolled my inner loop a bit, calculating the forces of 2 bodies in one iteration instead of one at a time.

I did not see any improvement in unrolling the loop further. Partially because that would create some register pressure and I think I was being bottlenecked by the number of simultaneous FMA operations my cores could do (2).

Finally I noticed that I had a pretty high rate of L1 cache misses, which confused me since almost all of my critical loop was using registers. The only thing being loaded from cache was the positions and masses of the bodies, and I wasn't sure why it was suddenly so high. I realized I was using `_mm256_load_pd` and might have been accessing elements across two cache lines, so my read and writes were likely now creating some issues (especially with 6x more bodies simulated). I did some research and came across `alignas` and `aligned_alloc`. I rewrote my vectors of positions and masses as a C-array sized with `aligned_alloc` and it helped reduce the cache misses a bit!


Helpful Resources:  
1. https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html  
2. https://agner.org/optimize/ 
3. https://stackoverflow.com/questions/44578967/what-is-the-most-efficient-way-to-clear-a-single-or-a-few-zmm-registers-on-knigh 
4. https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html