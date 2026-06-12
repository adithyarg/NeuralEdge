# NeuralEdge — FPGA AI Co-Processor

A low-cost, low-power FPGA-based neural network inference accelerator targeting real-time AI on embedded systems. Built on the Digilent Basys3 (Xilinx Artix-7 XC7A35T), NeuralEdge offloads MLP inference from the host CPU, running under **500 mW** at a bill-of-materials cost under **₹1,250**.

The accelerator is demonstrated with two applications:
- **MNIST digit recognition** — 784-input, 94% accuracy, ~0.1 ms inference
- **Thermal human detection** — 32×32 infrared patches from the LLVIP dataset, 78.8% accuracy

---

## Motivation

Drones and small UAVs increasingly need on-device AI — object detection, human tracking, autonomous decision-making — without cloud dependency. Existing solutions are either too expensive (NVIDIA Jetson Nano: ₹8,300+, Google Coral: ₹5,000+) or too slow (CPU-only inference on ESP32/RPi). NeuralEdge bridges that gap with dedicated hardware logic on a $30 FPGA dev board.

---

## Architecture

```
Host PC / MCU
    │  UART 115200 baud
    ▼
┌─────────────────────────────────────┐
│           Basys3 (Artix-7)          │
│                                     │
│  UART RX → Pixel Buffer (1024 B)    │
│       ↓                             │
│  MLP Accelerator (FSM + MAC)        │
│  1024 → 64 → 32 → 2                 │
│  Fixed-point Q4.12 weights          │
│  Q8.8 activations                   │
│       ↓                             │
│  UART TX → Result (1 byte)          │
│  LED[1] = Human  LED[0] = No Human  │
└─────────────────────────────────────┘
```

**Key specs:**
| Parameter | Value |
|-----------|-------|
| FPGA | Xilinx Artix-7 XC7A35T |
| Clock | 100 MHz |
| UART | 115200 baud |
| Weight format | Q4.12 signed 16-bit |
| Activation format | Q8.8 signed 16-bit |
| Inference latency | < 1 ms (compute) |
| Power | < 500 mW |

---

## Repository Structure

```
NeuralEdge/
├── rtl/
│   ├── mlp_accel.v          # MLP accelerator core (FSM + MAC + BRAMs)
│   ├── top_basys3.v         # Basys3 top-level (UART + pixel buffer)
│   ├── uart/
│   │   ├── uart_rx.v        # UART receiver
│   │   └── uart_tx.v        # UART transmitter
│   └── hex/
│       └── thermal/         # Quantized weights for thermal detection
│           ├── weight_l1.hex
│           ├── weight_l2.hex
│           ├── weight_l3.hex
│           ├── bias_l1.hex
│           ├── bias_l2.hex
│           └── bias_l3.hex
├── training/
│   ├── train.py             # MNIST MLP training
│   ├── quantize.py          # MNIST weight export to hex
│   └── thermal/
│       ├── prepare_dataset.py   # LLVIP patch extraction
│       ├── train_thermal.py     # Thermal MLP training
│       └── quantize_thermal.py  # Thermal weight export to hex
├── host/
│   ├── webcam_detect.py     # MNIST digit recognition via webcam
│   └── thermal_detect.py   # Thermal human detection (live/batch/image)
├── syn/
│   ├── basys3.xdc           # Vivado constraints
│   └── build_basys3.tcl     # Batch build script
└── README.md
```

---

## Getting Started

### 1. Prerequisites

- Digilent Basys3 board
- Vivado 2022.x or later (free WebPACK edition works)
- Python 3.8+ with: `torch`, `opencv-python`, `pyserial`, `numpy`

```bash
pip install torch torchvision opencv-python pyserial numpy
```

### 2. Build the Bitstream

From the project root:

```bash
vivado -mode batch -source syn/build_basys3.tcl
```

Bitstream is output to `build/top_basys3.bit`.

### 3. Program the Basys3

Open Vivado Hardware Manager → Auto Connect → Program Device → select `top_basys3.bit`.

### 4. Run Inference

**Live webcam (simulated thermal):**
```bash
python host/thermal_detect.py --port COM3
```
Press `SPACE` to run inference on the current frame. Replace `COM3` with your actual serial port (check Device Manager → Ports).

**Single image:**
```bash
python host/thermal_detect.py --port COM3 --image path/to/image.jpg
```

**Batch folder:**
```bash
python host/thermal_detect.py --port COM3 --folder path/to/images/
```

---

## Training from Scratch

### Thermal Human Detection

1. Download the [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/) (~9.3 GB)

2. Extract patches:
```bash
python training/thermal/prepare_dataset.py --llvip_dir /path/to/LLVIP
```

3. Train:
```bash
python training/thermal/train_thermal.py
```

4. Export weights to hex:
```bash
python training/thermal/quantize_thermal.py
```

### MNIST Digit Recognition

```bash
python training/train.py
python training/quantize.py
```

---

## Results

| Application | Accuracy | Dataset | Inference (FPGA) |
|-------------|----------|---------|-----------------|
| MNIST digits | 94.1% | MNIST test set (10k) | ~0.1 ms |
| Thermal human detection | 78.8% | LLVIP test set (3k) | ~0.1 ms |

---

## How It Works

The accelerator implements a 3-layer fully connected MLP entirely in RTL:

- **Weight BRAMs** — layer weights loaded from hex files at bitstream generation time via `$readmemh`
- **MAC unit** — pipelined multiply-accumulate with saturation, 2-cycle latency
- **FSM** — sequences through layers, neurons, and inputs; eliminates expensive multiply on the critical path using a registered base-address accumulator
- **Fixed-point arithmetic** — Q4.12 weights × Q8.8 activations, result truncated back to Q8.8 after each layer
- **UART interface** — host sends raw pixel bytes, FPGA responds with 1-byte class index

The same RTL core is retargetable to any MLP by changing layer size parameters and hex files — demonstrated by switching between MNIST (784-in, 10-class) and thermal detection (1024-in, 2-class).

---

## License

MIT License — see [LICENSE](LICENSE).
