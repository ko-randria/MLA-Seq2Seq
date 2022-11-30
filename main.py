{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import time\n",
    "import tensorflow as tf\n",
    "import os\n",
    "import argparse\n",
    "import sys\n",
    "#import nvidia_smi\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "parser = argparse.ArgumentParser(\n",
    "\tusage=__doc__,\n",
    "\tformatter_class=argparse.RawDescriptionHelpFormatter)\n",
    "parser.add_argument('-metric', choices=['MPJPE', '3DPCK'], nargs='?', type=str, default='MPJPE')\n",
    "parser.add_argument('-gpu', nargs='?', type=int, default=-1)\n",
    "parser.add_argument('-epochs', nargs='?', type=int, default=50)\n",
    "parser.add_argument('-batch_size', nargs='?', type=int, default=64)\n",
    "parser.add_argument('-lr', nargs='?', type=float, default=0.0001)\n",
    "parser.add_argument('-load', nargs='?', type=bool, default=False)\n",
    "args = parser.parse_args()\n",
    "\n",
    "\"\"\"\n",
    "########Si on a accès à un GPU \n",
    "args.gpu=check(gpu=args.gpu)\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = str(args.gpu)\n",
    "if args.gpu >= 0:\n",
    "\ttry:\n",
    "\t\tgpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "\t\tgpu = gpus[0]\n",
    "\t\ttf.config.experimental.set_memory_growth(gpu, True)\n",
    "\texcept RuntimeError as e:\n",
    "\t\tprint(e)\"\"\"\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "if not args.load:\n",
    "\n",
    "    S2S=create_model()\n",
    "    S2S.summary()\n",
    "    S2S=train_loop(args.lr,args.epochs,args.batch_size,args.metric......)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def check(gpu):\n",
    "\tif gpu >= 0:\n",
    "\t\tnvidia_smi.nvmlInit()\n",
    "\t\tdeviceCount = nvidia_smi.nvmlDeviceGetCount()\n",
    "\t\tgpus=[]\n",
    "\t\tfor i in range(deviceCount):\n",
    "\t\t\thandle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)\n",
    "\t\t\tmem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)\n",
    "\t\t\tpercentage_used = 100 * (mem_res.used / mem_res.total)\n",
    "\t\t\tgpus.append(percentage_used)\n",
    "\t\tgpu = np.argmin(gpus)\n",
    "\t\tif gpus[gpu] < 50:\n",
    "\t\t\treturn gpu\n",
    "\t\telse:\n",
    "\t\t\tprint('No GPU availalble (all used with at least 50% memory)')\n",
    "\t\t\tsys.exit()\n",
    "\treturn gpu"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.12 ('g')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "d793707d6e04838b1e2f2e46fee5aacca74f8fa90dc0d2d6a47de7395102c672"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
