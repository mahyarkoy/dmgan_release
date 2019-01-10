# Disconnected Manifold Learning for GANs
Find our paper at [NeurIPS 2018](https://papers.nips.cc/paper/7964-disconnected-manifold-learning-for-generative-adversarial-networks) and [ArXiv](https://arxiv.org/abs/1806.00880). Please cite the following if using the code:

```
@incollection{NIPS2018_7964,
  title = {Disconnected Manifold Learning for Generative Adversarial Networks},
  author = {Khayatkhoei, Mahyar and Singh, Maneesh K. and Elgammal, Ahmed},
  booktitle = {Advances in Neural Information Processing Systems 31},
  editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
  pages = {7354--7364},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/7964-disconnected-manifold-learning-for-generative-adversarial-networks.pdf}
}
```


### Running the code:
After installing the necessary python dependencies, simply run:
```bash
$ python run_dmgan.py -l logs -e 5000 -s 0
```
This code implements the line segments experiments from the paper.
To change the number of generators, modify ```self.g_num``` from inside ```DMGAN.__init__``` (default is 10 generators).
To disable prior learning, uncomment the following line from inside ```DMGAN.step```:
```python
z_data = np.random.randint(low=0, high=self.g_num, size=batch_size)
```
To use modified GAN objective instead of WGAN, set the following from inside ```DMGAN.__init__``` (default setting is for wgan with one sided gradient penalty):
```python
self.d_loss_type = 'log'
self.g_loss_type = 'mod'
```
