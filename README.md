# HSE at LSCDiscovery in Spanish
This repository contains the source code of our submission for **Lexical Semantic Change Discovery Shared Task in Spanish**<br>
[CodaLab competition page](https://codalab.lisn.upsaclay.fr/competitions/2243)

Paper: Kseniia Kashleva, Alexander Shein, Elizaveta Tukhtina, and Svetlana Vydrina. 2022. HSE at LSCDiscovery in Spanish: Clustering and Profiling for Lexical Semantic Change Discovery. In _Proceedings of the 3rd Workshop on Computational Approaches to Historical Language Change_, pages 193–197, Dublin, Ireland. Association for Computational Linguistics.<br>
[PDF](https://aclanthology.org/2022.lchange-1.21.pdf) | [ACL Anthology](https://aclanthology.org/2022.lchange-1.21/)

## Approaches
During the LSCDiscovery shared task we addressed two problems: 
- graded change discovery 
- binary change detection

Our team used 3 following approaches:
1. BERT embeddings with clustering
2. Grammatical profiling
3. Grammatical profiling enhanced with permutation-based statistical tests

## Data

Download corpora:
```bash
wget https://users.dcc.uchile.cl/~fzamora/old_corpus.tar.bz2
wget https://users.dcc.uchile.cl/~fzamora/modern_corpus.tar.bz2
```
Download target words from evaluation phase:
```bash
wget https://users.dcc.uchile.cl/~fzamora/targets.tsv
```

## Directories

— **code_embeddings**: code for BERT + clustering approach<br>
— **profiling_with_permutations**: code for grammatical profiling + permutation tests approach<br>

The source code for grammatical profiling approach can be found here:<br>
https://github.com/glnmario/semchange-profiling
