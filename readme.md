# Interpretable AMR-Based Question Decomposition for Multi-hop Question Answering
A code implementation of this paper (<a href="https://www.ijcai.org/proceedings/2022/0568.pdf">IJACI 2022</a>). 


## QuickStart

1. Download raw data from <a href="https://hotpotqa.github.io/">HotpotQA</a>.

2. Download a python library <a href="https://github.com/bjascob/amrlib">amrlib</a>. Follow this tutorial to load AMR-paring and AMR-to-Text generation model: https://amrlib.readthedocs.io/en/latest/

```bash
stog = amrlib.load_stog_model()
gtos = amrlib.load_gtos_model()
```  

```  
3. Question Decomposition (QD)
```bash
python QD_bridge.py		# QD1 for bridge questions
python QD_comp.py		# QD2 for comparison questions
```

4. Follow this paper <a href="https://github.com/shmsw25/DecompRC">DecompRC> to answer all sub-questions and predict the final answer. 

## Citation

If you use this code useful, please star our repo or consider citing:
```
@article{deng2022interpretable,
  title={Interpretable AMR-based question decomposition for multi-hop question answering},
  author={Deng, Zhenyun and Zhu, Yonghua and Chen, Yang and Witbrock, Michael and Riddle, Patricia},
  journal={arXiv preprint arXiv:2206.08486},
  year={2022}
}
```