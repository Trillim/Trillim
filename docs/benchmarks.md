# Benchmarks

## Benchmark Methodology

All runs in this page used the same repeatable process:

- Fresh system restart before benchmark sessions.
- 5 warmup runs of both engines before recorded benchmark runs to bring the system to steady state.
- Interleaved execution between compared models to reduce time-drift bias.
- Cool-down between runs until CPU temperature returned to `45C`.

These controls reduce run-to-run noise, but consumer CPU measurements should still be treated as directional.

## Results

This page summarizes benchmark plots comparing BitNet and Darknet behavior across decode and runtime quantization scenarios.

## Decode Throughput

Run A:

![Decode throughput run A](imgs/DecodeA.png)

Run B:

![Decode throughput run B](imgs/DecodeB.png)

### Result Summary

- Decode performance is broadly comparable to BitNet.
- Darknet shows higher peak throughput values.
- The advantage is most consistent once `num_threads >= 4`.

## Runtime Quantization

### Q4_0

Run A:

![Q4_0 run A](imgs/Q4_0A.png)

Run B:

![Q4_0 run B](imgs/Q4_0B.png)

### Q5_0

Run A:

![Q5_0 run A](imgs/Q5_0A.png)

Run B:

![Q5_0 run B](imgs/Q5_0B.png)

### Q6_K

Run A:

![Q6_K run A](imgs/Q6_KA.png)

Run B:

![Q6_K run B](imgs/Q6_KB.png)

### Q8_0

Run A:

![Q8_0 run A](imgs/Q8_0A.png)

Run B:

![Q8_0 run B](imgs/Q8_0B.png)

## Where Darknet Is Better

Darknet is better under these conditions:

- `num_threads >= 4`
- Decode throughput is approximately equal to BitNet on average, but Darknet reaches higher peaks

## Limitations (Consumer CPU Benchmarking)

These results should be interpreted as directional rather than absolute:

- Consumer CPUs vary heavily in boost behavior, thermal limits, and power settings.
- Background processes and OS scheduling noise can materially affect short runs.
- Memory bandwidth/cache differences can dominate results as thread count scales.
- SMT/Hyper-Threading behavior differs by CPU generation and workload shape.
- Results can change with compiler flags, kernel versions, and microcode updates.
- Prompt mix, context length, and warm-up policy can shift measured decode rates.
