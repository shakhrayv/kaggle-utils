## AutoEnsembler

AutoEnsembler is a simple tool that allows to achieve +2%-3% validation accuracy during models ensembling. 
It works by ierarchically selecting best blending parameters and by consequently shrinking the grid size. It has `n_jobs` parameter to allow for faster computations.

AutoEnsembler was used in these competitions:
* [Humpback Whales Identification](https://www.kaggle.com/c/humpback-whale-identification): 0.6% LB improvement, 2% CV improvement
