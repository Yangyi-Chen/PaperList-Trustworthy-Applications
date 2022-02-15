# PaperList

## Survey
- **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing；** Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig
- **Unsolved Problems in ML Safety;** Dan Hendrycks, Nicholas Carlini, John Schulman, Jacob Steinhardt
- **Backdoor Learning: A Survey;** Yiming Li, Baoyuan Wu, Yong Jiang, Zhifeng Li, Shu-Tao Xia
- **Explanation-Based Human Debugging of NLP Models: A Survey;** Piyawat Lertvittayakumjorn and Francesca Toni
- **Interpreting Deep Learning Models in Natural Language Processing: A Review;** Xiaofei Sun, Diyi Yang, Xiaoya Li, Tianwei Zhang, Yuxian Meng, Han Qiu, Guoyin Wang, Eduard Hovy, Jiwei Li
- **A Survey of Data Augmentation Approaches for NLP;** Steven Y. Feng et al 
- **Data Augmentation Approaches in Natural Language Processing: A Survey;** Bohan Li, Yutai Hou, Wanxiang Che
- **An Empirical Survey of Data Augmentation for Limited Data Learning in NLP;** Jiaao Chen, Derek Tam, Colin Raffel, Mohit Bansal, Diyi Yang 
- **Shortcut Learning in Deep Neural Networks;** Robert Geirhos et al 







## Adversarial Sample
- **Repairing Adversarial Texts through Perturbation;** Guoliang Dong, Jingyi Wang, Jun Sun, Sudipta Chattopadhyay, Xinyu Wang, Ting Dai, Jie Shi and Jin Song Dong; Introduce method to detect textual adversarial samples and "repair" them. 
- **Adversarial Examples Are Not Bugs, They Are Features;** Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry
- **Detecting Adversarial Samples from Artifacts; (Detection)** Reuben Feinman, Ryan R. Curtin, Saurabh Shintre, Andrew B. Gardner
- **Detection of Adversarial Examples in NLP: Benchmark and Baseline via Robust Density Estimation；** Anonymous





## Trustworthy (broad topic)
- **Explaining Prediction Uncertainty of Pre-trained Language Models by Detecting Uncertain Words in Inputs; (Uncertainty)** Hanjie Chen, Yangfeng Ji 
- **Calibration of Pre-trained Transformers; (Uncertainty)** Shrey Desai, Greg Durrett; Empirically study the calibration of PLMs, in both in-domain & out-out-domain. Also include label smoothing and temperature scaling in the experiments. 
- **Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data; (Uncertainty)** Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, Chao Zhang
- **Types of Out-of-Distribution Texts and How to Detect Them; (OOD detection)** Udit Arora, William Huang and He He; Analyze two types of OOD data and benchmark two popular OOD detection methods, get some interesting findings. 
- **On the Trade-off between Adversarial and Backdoor Robustness;** Cheng-Hsin Weng, Yan-Ting Lee, Shan-Hung Wu
- **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels; (Noisy labels)** Bo Han et al; Propose a training algorithm that combat the noisy labels issue. 
- **Learning to Reweight Examples for Robust Deep Learning;(Noisy labels)** Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun
- **DIVIDEMIX: LEARNING WITH NOISY LABELS AS SEMI-SUPERVISED LEARNING; (Noisy labels)** Junnan Li, Richard Socher, Steven C.H. Hoi
- **Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets;** Nelson F. Liu, Roy Schwartz, Noah A. Smith
- **FINE-TUNING DISTORTS PRETRAINED FEATURES AND UNDERPERFORMS OUT-OF-DISTRIBUTION;** Anonymous


## Data Augmentation
- **Few-Shot Text Classification with Triplet Networks, Data Augmentation, and Curriculum Learning; (Few-shot setting)** Jason Wei, Chengyu Huang, Soroush Vosoughi, Yu Cheng, Shiqi Xu
- **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks;** Jason Wei, Kai Zou
- **CODA: CONTRAST-ENHANCED AND DIVERSITY-PROMOTING DATA AUGMENTATION FOR NATURAL LANGUAGE UNDERSTANDING;** Yanru Qu, Dinghan Shen, Yelong Shen, Sandra Sajeev, Jiawei Han, Weizhu Chen; Propose to integrate data augmentation & adversarial training method and utilize the contrastive learning algorithm to incorporate the obtained augmented samples into the training process.
- **Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification;** Shuhuai Ren, Jinchao Zhang, Lei Li, Xu Sun, Jie Zhou

## Data Points Selection
- **On Training Instance Selection for Few-Shot Neural Text Generation; (Prompt-based generation)** Ernie Chang, Xiaoyu Shen, Hui-Syuan Yeh, Vera Demberg; Select informative and representative samples based on K-means clustering.
- **RESOLVING TRAINING BIASES VIA INFLUENCEBASED DATA RELABELING; (Data relabeling)** Anonymous 
- **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics;** Swabha Swayamdipta et al 
- **Are All Training Examples Created Equal? An Empirical Study;** Kailas Vodrahalli, Ke Li, Jitendra Malik; In MNIST, using a subset of the whole training dataset can achieve comparable performance while this conclusion doesn't hold in CIFAR10 & CIFAR100. They propose a method based on the magnitude of the gradients to select important samples. 



## Backdoor Learning
- **Rethink Stealthy Backdoor Attacks in Natural Language Processing;** Lingfeng Shen, Haiyun Jiang, Lemao Liu, Shuming Shi; Results in Table 1 are a little bit weird. But present an interesting idea, measuring what actually contributes to the high ASR of backdoor attack (e.g. trigger? mislabel? et al). Also, propose two effective methods to defend against stealthy backdoor attacks. 
- **Excess Capacity and Backdoor Poisoning;** Naren Sarayu Manoj, Avrim Blum
- **Anti-Backdoor Learning: Training Clean Models on Poisoned Data;** Yige Li et al


## Parameter Efficient Tuning
- **Parameter-Efficient Transfer Learning for NLP;** Neil Houlsby et al 


## Interesting Topics
- **Advancing mathematics by guiding human intuition with AI;** Alex Davies et al
- **LAMOL: LANGUAGE MODELING FOR LIFELONG LANGUAGE LEARNING;** Fan-Keng Sun, Cheng-Hao Ho, Hung-Yi Lee; Use an LM to both generate previous tasks' data and solve the tasks.
- **Explaining and Improving Model Behavior with k Nearest Neighbor Representations;** Nazneen Fatema Rajani, Ben Krause, Wenpeng Yin, Tong Niu, Richard Socher, Caiming Xiong; The related work section is informative.
- **NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework;** Xingcheng Yao, Yanan Zheng, Xiaocong Yang, Zhilin Yang; "Given some labeled task data and a large general corpus, TLM uses task data as queries to retrieve a tiny subset of the general corpus and jointly optimizes the task objective and the language modeling objective from scratch"
- **Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning;** Nicolas Papernot and Patrick McDaniel
- **UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION;** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
- **SHOULD WE BE Pre-TRAINING? EXPLORING END-TASK AWARE TRAINING IN LIEU OF CONTINUED PRE-TRAINING; (PLM Pre-training)** Lucio M. Dery, Paul Michel, Ameet Talwalkar, Graham Neubig; Study the end-task aware pre-training. 
- **What are the best Systems? New Perspectives on NLP Benchmarking;** Pierre Colombo, Nathan Noiry, Ekhine Irurozki, Stephan Clemencon; Study how to aggregate metrics in multi-task evaluation. "benchmarks are made of datasets, metrics, and a way to aggregate performance. ... If the bulk of the NLP community efforts on this domain is about collecting new datasets and introducing new metrics, little work is concerned with the third part, namely how to aggregate various performances."
- **On Human Predictions with Explanations and Predictions of Machine Learning Models: A Case Study on Deception Detection;** Vivian Lai, Chenhao Tan; Use ML models and explanation technology to help the human decision process. 





## Resources
- **Transfer Learning in NLP; (Tutorial)** https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5882add69e_6_646; Gives a hands-on code to conduct pre-training.
- **Robustness and Adversarial Examples in NLP; (Tutorial)** https://docs.google.com/presentation/d/1E_0qEwQkS43FJGzOEUrpee9zqi8y5lx6D-ABQl3KFas/edit#slide=id.p
- **How to write research paper?** https://students.uu.nl/sites/default/files/ge0-aw-guide-for-scientific-writing-2016.pdf

