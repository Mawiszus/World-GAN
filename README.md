If you are using this code in your own project, please cite our paper(s):

IEEE Transactions on Games:
```
@article{awiszus2022wor,
  title={Wor(l)d-GAN: Towards Natural Language Based PCG in Minecraft},
  author={Awiszus, Maren and Schubert, Frederik and Rosenhahn, Bodo},
  journal={IEEE Transactions on Games},
  year={2022},
  publisher={IEEE}
}
```

CoG 2021:
```
@inproceedings{awiszus2021worldgan,
  title={World-GAN: a Generative Model for Minecraft Worlds},
  author={Awiszus, Maren and Schubert, Frederik and Rosenhahn, Bodo},
  booktitle={Proceedings of the IEEE Conference on Games},
  year={2021}
}
```

## This is the MCPI branch!

This branch allows to train and display directly on a Minecraft [Spigot](https://www.spigotmc.org/) server that's running the [RaspberryJuice](https://dev.bukkit.org/projects/raspberryjuice) plugin.
**See [here](https://gist.github.com/evgkarasev/04d562babc6185e5f0dd9999cd1cc1ca)** for a tutorial on how to set up and run such a server.
You can connect to the server and see Wor(l)d-GAN learn in real time.

Once the server is running, you can run the `main.py` with the new settings in `config.py`:
```shell
$ python main.py --server_train --server_start_pos 0 0 0 --server_end_pos 20 10 20 --server_render_pos 30 0 0 --render --scales 0.75 0.5 --num_layer 3 --alpha 100 --niter 4000 --nfc 64 --pad_with_noise
```
**Note:** The coordinates are relative to the spawn point, not the true origin!

Also, per default, the method will read the Blocks with Data, which can take quite a while. 
The argument `--server_get_block_data` can speed up reading the original sample, but it will not keep data (i.e. direction of stairs etc).



# World-GAN

Official pytorch implementation of the paper: "World-GAN: a Generative Model for Minecraft Worlds"
For more information on TOAD-GAN, please refer to the paper ([arxiv](https://arxiv.org/pdf/2106.10155v1.pdf)).


This Project includes graphics from the game _Minecraft_ **It is not affiliated with or endorsed by Mojang.
The project was built for research purposes only.**

### CoG 2021

Our paper "World-GAN: a Generative Model for Minecraft Worlds" was accepted at [CoG 2021](https://ieee-cog.org/2021/index.html)!


## Getting Started

This section includes the necessary steps to train World-GAN on your system.

### Python

You will need [Python 3](https://www.python.org/downloads) and the packages specified in requirements.txt.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

**NOTE:** If you have a **GPU**, check if it is usable with [Pytorch](https://pytorch.org) and change the requirement in the file to use +gpu instead of +cpu.
Training on a GPU is significantly faster.

Install packages with:
```shell
$ pip3 install -r requirements.txt -f "https://download.pytorch.org/whl/torch_stable.html"
```
Make sure you use the `pip3` that belongs to your previously defined virtual environment.

## World-GAN

### Training

Once all prerequisites are installed, World-GAN can be trained by running `main.py`.
Make sure you are using the python installation you installed the prerequisites into.

There are several command line options available for training. These are defined in `config.py`.
An example call which will train a 3-layer World-GAN *TODO* with 4000 iterations each scale would be:

```shell
$ python main.py *TODO* --alpha 100 --niter 4000 --nfc 64
```

### Generating samples

If you want to use your trained World-GAN to generate more samples, use `generate_samples.py`.
Make sure you define the path to a pretrained World-GAN and the correct input parameters it was trained with.

```shell
$ python generate_samples.py  --out_ path/to/pretrained/World-GAN --input-dir input --input-name lvl_1-1.txt --num_layer 3 --alpha 100 --niter 4000 --nfc 64
```

### Experiments

We supply the code for experiments made for our paper.
These files come with their own command line parameters each to control the experiment.
Check the files for more info.


## Built With

* Pytorch - Deep Learning Framework
* PyAnvilEditor - a framework for reading and writing nbt files for Minecraft 1.16

## Authors

* **[Maren Awiszus](https://www.tnt.uni-hannover.de/de/staff/awiszus/)** - Institut für Informationsverarbeitung, Leibniz University Hanover
* **[Frederik Schubert](https://www.tnt.uni-hannover.de/de/staff/schubert/)** - Institut für Informationsverarbeitung, Leibniz University Hanover

## Copyright

This code is not endorsed by Mojang and is only intended for research purposes. 

