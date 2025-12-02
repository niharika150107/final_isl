# Humanoid Project — Setup & Training Guide

This guide explains how to:

1. Create and activate a virtual environment
2. Install dependencies using `req.txt`
3. Run the training script

---

## 📌 1. Create a Virtual Environment

### **Step 1 — Install venv (if not installed)**

```bash
sudo apt install python3-venv
```

### **Step 2 — Create a virtual environment**

```bash
python3 -m venv venv
```

### **Step 3 — Activate the environment**

```bash
source venv/bin/activate
```

You should now see `(venv)` in your terminal prompt.

---

## 📌 2. Install Requirements

Make sure `req.txt` is inside your project folder.

### Install all dependencies:

```bash
pip install --upgrade pip
pip install -r req.txt
```

If installation is slow, it is normal — MuJoCo, Torch, and MediaPipe are large packages.

---

## 📌 3. Run Training

Once the virtual environment is active and dependencies are installed, run:

```bash
python train.py --epochs 100 --n-steps 512 --batch-size 32
```

You can adjust the arguments:

* `--epochs` → number of training epochs
* `--n-steps` → rollout length
* `--batch-size` → PPO minibatch size

Example:

```bash
python train.py --epochs 200 --n-steps 1024 --batch-size 64
```

---

## 📌 4. Deactivating the Virtual Environment

When you're done:

```bash
deactivate
```

---

## 📌 5. Troubleshooting

### **Process killed during training**

WSL may run out of RAM. Increase memory in `.wslconfig`:

```
[wsl2]
memory=12GB
swap=16GB
```

Then restart WSL:

```bash
wsl --shutdown
```

---

## 📌 6. Project Structure

```
humanoid_project/
├── train.py
├── test.py
├── module1.py
├── module2.py
├── module3.py
├── req.txt
└── pose_images/
    ├── person1.png
    ├── person2.png
    └── ...
```

Place your pose images inside `pose_images/`.

---

Feel free to ask if you want a version with screenshots or a Colab setup guide.
