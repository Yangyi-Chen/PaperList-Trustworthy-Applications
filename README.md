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
- **Learning Neural Models for Natural Language Processing in the Face of Distributional Shift;** Paul Michel
- **A Survey of Evaluation Metrics Used for NLG Systems;** ANANYA B. SAI et al
- **Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models;** Ning Ding et al
- **Analysis Methods in Neural Language Processing: A Survey;** Yonatan Belinkov, James Glass
- **A Roadmap for Big Model;** Sha Yuan et al
- **Language (Technology) is Power: A Critical Survey of “Bias” in NLP;** Su Lin Blodgett et al
- **A Review on Language Models as Knowledge Bases;** Badr AlKhamissi et al
- **Vision-and-Language Pretrained Models: A Survey;** Siqu Long et al
- **Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods;** Aditya Mogadala et al
- **Multimodal Machine Learning: A Survey and Taxonomy;** Tadas Baltrusaitis et al
- **Measure and Improve Robustness in NLP Models: A Survey;** Xuezhi Wang et al
- **Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses;** Micah Goldblum et al
- **Graph Neural Networks for Natural Language Processing: A Survey;** Lingfei Wu et al
- **Faithfulness in Natural Language Generation: A Systematic Survey of Analysis, Evaluation and Optimization Methods;** Wei Li et al
- **Survey of Hallucination in Natural Language Generation;** Ziwei Ji et al
- **Meta Learning for Natural Language Processing: A Survey;** Hung-yi Lee et al
- **HOW TO KEEP TEXT PRIVATE? A SYSTEMATIC REVIEW OF DEEP LEARNING METHODS FOR PRIVACY-PRESERVING NATURAL LANGUAGE PROCESSING;** Samuel Sousa et al
- **A Survey on Machine Reading Comprehension: Tasks, Evaluation Metrics and Benchmark Datasets;** Changchang Zeng et al
- **Vision-and-Language Navigation: A Survey of Tasks, Methods, and Future Directions;** Jing Gu et al
- **Adapting to the Long Tail: A Meta-Analysis of Transfer Learning Research for Language Understanding Tasks;** Aakanksha Naik et al
- **Temporal Effects on Pre-trained Models for Language Processing Tasks;** Oshin Agarwal, Ani Nenkova
- **Neural Unsupervised Domain Adaptation in NLP—A Survey;** Alan Ramponi, Barbara Plank
- **Core Challenges in Embodied Vision-Language Planning;** Jonathan Francis et al
- **Multimodal Machine Learning: A Survey and Taxonomy;** Tadas Baltrusaitis, Chaitanya Ahuja, Louis-Philippe Morency; Introduce 4 challenges for multi-modal learning, including representation, translation, alignment, fusion, and co-learning. 




## Adversarial Sample
- **Repairing Adversarial Texts through Perturbation;** Guoliang Dong, Jingyi Wang, Jun Sun, Sudipta Chattopadhyay, Xinyu Wang, Ting Dai, Jie Shi and Jin Song Dong; Introduce method to detect textual adversarial samples and "repair" them. 
- **Adversarial Examples Are Not Bugs, They Are Features;** Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry
- **Detecting Adversarial Samples from Artifacts; (Detection)** Reuben Feinman, Ryan R. Curtin, Saurabh Shintre, Andrew B. Gardner
- **Detection of Adversarial Examples in NLP: Benchmark and Baseline via Robust Density Estimation；** Anonymous
- **AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples; (Adversarial Training)** Dongyeop Kang et al
- **Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge; (Adversarial Training)** Pasquale Minervini, Sebastian Riedel
- **Reevaluating Adversarial Examples in Natural Language; (Evaluation)** John X. Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi; Discuss how to align automatic evaluation metrics with human intuition; and integrate automatic metrics in the searching process to preserve quality and validity of adversarial samples. 
- **Evaluating the Robustness of Neural Language Models to Input Perturbations;** Milad Moradi, Matthias Samwald; Benchmark models' robustness to real-world user inputs; they use adversarial samples to simulate user inputs. 
- **Interpreting the Robustness of Neural NLP Models to Textual Perturbations;** Yunxiang Zhang et al
- **Distinguishing Non-natural from Natural Adversarial Samples for More Robust Pre-trained Language Model;** Anonymous; Using outlier detection to filter non-natural adversarial samples. 
- **Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense;** Thai Le et al
- **Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification;** Cheng-Han Chiang, Hung-yi Lee
- **Consistency Training with Virtual Adversarial Discrete Perturbation;** Jungsoo Park et al
- **Improving Robustness of Language Models from a Geometry-aware Perspective;** Bin Zhu et al
- **AEON: A Method for Automatic Evaluation of NLP Test Cases;** Jen-tse Huang et al; Validity verification. 
- **Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks; (Adversarial Training)** Zhao Meng et al;
- **Token-Aware Virtual Adversarial Training in Natural Language Understanding; (Adversarial Training)** Linyang Li, Xipeng Qiu
- **It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations;** Samson Tan et al

## Calibration
- **Explaining Prediction Uncertainty of Pre-trained Language Models by Detecting Uncertain Words in Inputs** Hanjie Chen, Yangfeng Ji 
- **Calibration of Pre-trained Transformers;** Shrey Desai, Greg Durrett; Empirically study the calibration of PLMs, in both in-domain & out-out-domain. Also include label smoothing and temperature scaling in the experiments. 
- **Teaching models to express their uncertainty in words;** Stephanie Lin et al
- **Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data;** Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, Chao Zhang

## Trustworthy (broad topic)
- **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels; (Noisy labels)** Bo Han et al; Propose a training algorithm that combat the noisy labels issue. 
- **Learning to Reweight Examples for Robust Deep Learning;(Noisy labels)** Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun
- **DIVIDEMIX: LEARNING WITH NOISY LABELS AS SEMI-SUPERVISED LEARNING; (Noisy labels)** Junnan Li, Richard Socher, Steven C.H. Hoi
- **On the Calibration of Pre-trained Language Models using Mixup Guided by Area Under the Margin and Saliency; (Uncertainty)** Seo Yeon Park, Cornelia Caragea
- **Tailor: Generating and Perturbing Text with Semantic Controls;** Alexis Ross et al 
- **When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data;** Peter Hase, Mohit Bansal
- **Systematicity, Compositionality and Transitivity of Deep NLP Models: a Metamorphic Testing Perspective;** Edoardo Manino et al
- **A Metamorphic Testing Approach for Assessing Question Answering Systems;** Kaiyi Tu et al
- **White-box Testing of NLP models with Mask Neuron Coverage;** Arshdeep Sekhon et al; Evaluation. Employ white-box information to reduce test cases. The intuition is to identify when an input’s activation of attention neurons is subsumed by that of prior test inputs. 
- **Detecting Label Errors using Pre-Trained Language Models; (Noisy labels)** Derek Chong et al
## Fairness
- **Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints;** Jieyu Zhao et al 
- **Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings;** Tolga Bolukbasi et al
- **Easy Adaptation to Mitigate Gender Bias in Multilingual Text Classification;** Xiaolei Huang

## OOD Detection
- **Contrastive Out-of-Distribution Detection for Pretrained Transformers;** Wenxuan Zhou, Fangyu Liu, Muhao Chen
- **Types of Out-of-Distribution Texts and How to Detect Them;** Udit Arora, William Huang and He He; Analyze two types of OOD data and benchmark two popular OOD detection methods, get some interesting findings. 
- **Towards Textual Out-of-Domain Detection without In-Domain Labels;** Di Jin, Shuyang Gao, Seokhwan Kim, Yang Liu, Dilek Hakkani-Tur

## Robustness
- **Challenges in Generalization in Open Domain Question Answering; (Generalization)** Linqing Liu et al
- **IDANI: Inference-time Domain Adaptation via Neuron-level Interventions; (Domain Adaptation)** Omer Antverg et al
- **FINE-TUNING DISTORTS PRETRAINED FEATURES AND UNDERPERFORMS OUT-OF-DISTRIBUTION;** Anonymous
- **UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION; (Generalization)** Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
- **Robustness Gym: Unifying the NLP Evaluation Landscape;** Karan Goel et al
- **TextFlint: Unified Multilingual Robustness Evaluation Toolkit for Natural Language Processing;** Tao Gui et al
- **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList; (NLP Model Evaluation)** Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh
- **On the Trade-off between Adversarial and Backdoor Robustness;** Cheng-Hsin Weng, Yan-Ting Lee, Shan-Hung Wu
- **Measuring Robustness to Natural Distribution Shifts in Image Classification;** Rohan Taori et al
- **Can Rationalization Improve Robustness?;** Howard Chen et a
- **Adaptive Testing and Debugging of NLP Models;** Marco Tulio Ribeiro, Scott M. Lundberg; Human-in-the-loop NLP model debugging. 
- **Improving the Adversarial Robustness of NLP Models by Information Bottleneck;** Cenyuan Zhang et al; Consider robust and non-robust features. Use the information bottleneck to filter out non-robust features.




## Explanation
- **Rethinking Explainability as a Dialogue: A Practitioner’s Perspective;** Himabindu Lakkaraju et al
- **EXSUM: From Local Explanations to Model Understanding;** Yilun Zhou et al
- **On Human Predictions with Explanations and Predictions of Machine Learning Models: A Case Study on Deception Detection;** Vivian Lai, Chenhao Tan; Use ML models and explanation technology to help the human decision process. 
- **On the Sensitivity and Stability of Model Interpretations in NLP;** Fan Yin, Zhouxing Shi, Cho-Jui Hsieh, Kai-Wei Chang; Propose two new metrics in explainable NLP & propose a adversarial robustness based explainable method. 
- **EVALUATIONS AND METHODS FOR EXPLANATION THROUGH ROBUSTNESS ANALYSIS;** Cheng-Yu Hsieh et al
- **Explaining NLP Models via Minimal Contrastive Editing (MICE);** Alexis Ross et al
- **Necessity and Sufficiency for Explaining Text Classifiers: A Case Study in Hate Speech Detection;** Esma Balkir et al
- **The Solvability of Interpretability Evaluation Metrics;** Yilun Zhou, Julie Shah
- **ER-TEST: Evaluating Explanation Regularization Methods for NLP Models;** Brihi Joshi et al; Set an evaluation benchmark for explanation regularization methods, including OOD generalization (e.g., Unseen Datasets, Contrast Set Tests, Functional Tests).
- **Is Attention Explanation? An Introduction to the Debate;** Adrien Bibal et al




## Data Augmentation
- **Few-Shot Text Classification with Triplet Networks, Data Augmentation, and Curriculum Learning; (Few-shot setting)** Jason Wei, Chengyu Huang, Soroush Vosoughi, Yu Cheng, Shiqi Xu
- **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks;** Jason Wei, Kai Zou
- **CODA: CONTRAST-ENHANCED AND DIVERSITY-PROMOTING DATA AUGMENTATION FOR NATURAL LANGUAGE UNDERSTANDING;** Yanru Qu, Dinghan Shen, Yelong Shen, Sandra Sajeev, Jiawei Han, Weizhu Chen; Propose to integrate data augmentation & adversarial training method and utilize the contrastive learning algorithm to incorporate the obtained augmented samples into the training process.
- **Text AutoAugment: Learning Compositional Augmentation Policy for Text Classification;** Shuhuai Ren, Jinchao Zhang, Lei Li, Xu Sun, Jie Zhou
- **Generalized but not Robust? Comparing the Effects of Data Modification Methods on Out-of-Domain Generalization and Adversarial Robustness;** Tejas Gokhale et al
- **PromDA: Prompt-based Data Augmentation for Low-Resource NLU Tasks;** Yufei Wang et al
- **When Chosen Wisely, More Data Is What You Need: A Universal Sample-Efficient Strategy For Data Augmentation;** Ehsan Kamalloo, Mehdi Rezagholizadeh, Ali Ghodsi
- **EPiDA: An Easy Plug-in Data Augmentation Framework for High Performance Text Classification;** Minyi Zhao et al
- **Sibylvariant Transformations for Robust Text Classification;** Fabrice Harel-Canada et al
- **TreeMix: Compositional Constituency-based Data Augmentation for Natural Language Understanding;** Le Zhang, Zichao Yang, Diyi Yang
- **Intermediate Training on Question Answering Datasets Improves Generative Data Augmentation;** Dheeraj Mekala et al; Employing QA models to generate synthetic data. Most tasks can be reformulated as QA. 

## Training Data Points
- **On Training Instance Selection for Few-Shot Neural Text Generation; (Prompt-based generation)** Ernie Chang, Xiaoyu Shen, Hui-Syuan Yeh, Vera Demberg; Select informative and representative samples based on K-means clustering.
- **RESOLVING TRAINING BIASES VIA INFLUENCEBASED DATA RELABELING; (Data relabeling)** Anonymous 
- **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics;** Swabha Swayamdipta et al 
- **Are All Training Examples Created Equal? An Empirical Study;** Kailas Vodrahalli, Ke Li, Jitendra Malik; In MNIST, using a subset of the whole training dataset can achieve comparable performance while this conclusion doesn't hold in CIFAR10 & CIFAR100. They propose a method based on the magnitude of the gradients to select important samples. 
- **FIRST IS BETTER THAN LAST FOR TRAINING DATA INFLUENCE;** Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu, and Pradeep Ravikumar
- **Generative Data Augmentation for Commonsense Reasoning;** Yiben Yang et al; Filtering data-augmentation generated synthetic data. 
- **ILDAE: Instance-Level Difficulty Analysis of Evaluation Data;** Neeraj Varshney, Swaroop Mishra, Chitta Baral; Introduce difficulty scores for evaluation samples; have some interesting applications. 
- **Feeding What You Need by Understanding What You Learned;** Xiaoqiang Wang et al; Define different capability measures for both data and models. Employ the curriculum learning paradigm to train a better MRC model. 
- **Explaining and Improving Model Behavior with k Nearest Neighbor Representations;** Nazneen Fatema Rajani, Ben Krause, Wenpeng Yin, Tong Niu, Richard Socher, Caiming Xiong; The related work section is informative.
- **Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning;** Nicolas Papernot and Patrick McDaniel
- **Training Data is More Valuable than You Think: A Simple and Effective Method by Retrieving from Training Data;** Shuohang Wang, Yichong Xu, Yuwei Fang, Yang Liu, Siqi Sun, Ruochen Xu, Chenguang Zhu, Michael Zeng
- **AN EMPIRICAL STUDY OF EXAMPLE FORGETTING DURING DEEP NEURAL NETWORK LEARNING; (Memorization)** Mariya Toneva et al
- **Data Contamination: From Memorization to Exploitation; (Memorization)** Inbal Magar, Roy Schwartz; Study how data contamination issue in pretraining data affect memorization and exploitation. 
- **Does Learning Require Memorization? A Short Tale about a Long Tail; (Memorization)** Vitaly Feldman
- **Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation; (Memorization)** Aparna Elangovan, Jiayuan He, Karin Verspoor
- **Memorisation versus Generalisation in Pre-trained Language Models; (Memorization)** Michael Tänzer, Sebastian Ruder, Marek Rei
- **An Empirical Study of Memorization in NLP; (Memorization)** Xiaosen Zheng, Jing Jiang
- **Are Neural Networks Extracting Linguistic Properties or Memorizing Training Data? An Observation with a Multilingual Probe for Predicting Tense; (Memorization)** Bingzhi Li, Guillaume Wisniewski; Discuss probing tasks. 
- **Selecting Informative Contexts Improves Language Model Fine-tuning;** Richard Antonello et al
- **Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models;** Kushal Tirumala et al
- **ORCA: Interpreting Prompted Language Models via Locating Supporting Data Evidence in the Ocean of Pretraining Data;** Xiaochuang Han, Yulia Tsvetkov; Discuss why pre-training helps zero/few shot learning.


## Transfer & Lifelong & Few-shot & Zero-shot Learning
- **CROSSFIT: A Few-shot Learning Challenge for Cross-task Generalization in NLP;** Qinyuan Ye, Bill Yuchen Lin, Xiang Ren
- **FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding;** Yanan Zheng et al
- **FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS;** Jason Wei et al
- **MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION;** Victor Sanh et al
- **LAMOL: LANGUAGE MODELING FOR LIFELONG LANGUAGE LEARNING;** Fan-Keng Sun, Cheng-Hao Ho, Hung-Yi Lee; Use an LM to both generate previous tasks' data and solve the tasks.
- **CLUES: A Benchmark for Learning Classifiers using Natural Language Explanations;** Rakesh R Menon et al
- **Meta-learning via Language Model In-context Tuning;** Yanda Chen et al
- **Muppet: Massive Multi-task Representations with Pre-Finetuning;** Armen Aghajanyan et al
- **Cross-Task Generalization via Natural Language Crowdsourcing Instructions;** Swaroop Mishra et al
- **Unsupervised Cross-Task Generalization via Retrieval Augmentation;** Bill Yuchen Lin et al
- **Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks;** Yizhong Wang et al
- **Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners;** Shashank Gupta et al
- **Exploring and Predicting Transferability across NLP Tasks;** Tu Vu et al
- **On Continual Model Refinement in Out-of-Distribution Data Streams;** Bill Yuchen Lin et al
- **Continual-T0: Progressively Instructing 50+ Tasks to Language Models Without Forgetting;** Thomas Scialom et al


## Backdoor Learning
- **Rethink Stealthy Backdoor Attacks in Natural Language Processing;** Lingfeng Shen, Haiyun Jiang, Lemao Liu, Shuming Shi; Results in Table 1 are a little bit weird. But present an interesting idea, measuring what actually contributes to the high ASR of backdoor attack (e.g. trigger? mislabel? et al). Also, propose two effective methods to defend against stealthy backdoor attacks. 
- **Excess Capacity and Backdoor Poisoning;** Naren Sarayu Manoj, Avrim Blum
- **Anti-Backdoor Learning: Training Clean Models on Poisoned Data;** Yige Li et al
- **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering;** Bryant Chen et al
- **BACKDOOR DEFENSE VIA DECOUPLING THE TRAINING PROCESS;** Kunzhe Huang, Yiming Li, Baoyuan Wu, Zhan Qin, Kui Ren



## Prompt-based Learning
- **LFPT5: A UNIFIED FRAMEWORK FOR LIFELONG FEW-SHOT LANGUAGE LEARNING BASED ON PROMPT TUNING OF T5;** Chengwei Qin, Shafiq Joty; Life-long few-shot learning, using T5. 
- **Pre-trained Token-replaced Detection Model as Few-shot Learner;** Zicheng Li, Shoushan Li, Guodong Zhou
- **FlipDA: Effective and Robust Data Augmentation for Few-Shot Learning;** Jing Zhou et al
- **Example-based Hypernetworks for Out-of-Distribution Generalization;** Tomer Volk et al
- **Few-Shot Learning with Siamese Networks and Label Tuning;** Thomas Muller et al
- **Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks;** Yizhong Wang et al


## PLM
- **Adversarial Training for Large Neural Language Models;** Xiaodong Liu et al
- **SHOULD WE BE Pre-TRAINING? EXPLORING END-TASK AWARE TRAINING IN LIEU OF CONTINUED PRE-TRAINING; (PLM Pre-training)** Lucio M. Dery, Paul Michel, Ameet Talwalkar, Graham Neubig; Study the end-task aware pre-training. 
- **NLP From Scratch Without Large-Scale Pretraining: A Simple and Efficient Framework;** Xingcheng Yao, Yanan Zheng, Xiaocong Yang, Zhilin Yang; "Given some labeled task data and a large general corpus, TLM uses task data as queries to retrieve a tiny subset of the general corpus and jointly optimizes the task objective and the language modeling objective from scratch"
- **On the Transferability of Pre-trained Language Models: A Study from Artificial Datasets;** Cheng-Han Chiang, Hung-yi Lee
- **NoisyTune: A Little Noise Can Help You Finetune Pretrained Language Models Better; (Robust Fine-tuning)** Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, Xing Xie
- **Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting; (Robust Fine-tuning)** Sanyuan Chen, Yutai Hou, Yiming Cui, Wanxiang Che, Ting Liu, Xiangzhan Yu
- **BETTER FINE-TUNING BY REDUCING REPRESENTATIONAL COLLAPSE;** Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, Sonal Gupta; Discuss representation collapse.
- **SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization; (Robust Fine-tuning)** Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Tuo Zhao 
- **REVISITING FEW-SAMPLE BERT FINE-TUNING; (Stable Fine-tuning)** Tianyi Zhang et al
- **Raise a Child in Large Language Model: Towards Effective and Generalizable Fine-tuning; (Robust Fine-tuning)** Runxin Xu et al
- **MIXOUT: EFFECTIVE REGULARIZATION TO FINETUNE LARGE-SCALE PRETRAINED LANGUAGE MODELS; (Robust Fine-tuning)** Cheolhyoung Lee, Kyunghyun Cho, Wanmo Kang
- **Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping;** Jesse Dodge et al; Empirical study on the fine-tuning strategies. 
- **Impossible Triangle: What’s Next for Pre-trained Language Models?;** Chenguang Zhu, Michael Zeng
- **Towards Efficient NLP: A Standard Evaluation and A Strong Baseline;** Xiangyang Liu et al
- **Language Models are General-Purpose Interfaces;** Yaru Hao et al

## Parameter Efficient Tuning
- **Parameter-Efficient Transfer Learning for NLP;** Neil Houlsby et al 
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation;** Xiang Lisa Li, Percy Liang
- **LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS;** Edward Hu et al 
- **The Power of Scale for Parameter-Efficient Prompt Tuning;** Brian Lester, Rami Al-Rfou, Noah Constant
- **Robust Transfer Learning with Pretrained Language Models through Adapters;** Wenjuan Han, Bo Pang, Yingnian Wu
- **COMPACTER: Efficient Low-Rank Hypercomplex Adapter Layers;** Rabeeh Karimi Mahabad, James Henderson, Sebastian Ruder
- **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models;** Elad Ben-Zaken, Shauli Ravfogel, Yoav Goldberg
- **Parameter-Efficient Transfer Learning with Diff Pruning;** Demi Guo, Alexander M. Rush, Yoon Kim
- **ON ROBUST PREFIX-TUNING FOR TEXT CLASSIFICATION; (Robustness)** Anonymous 
- **Ensembles and Cocktails: Robust Finetuning for Natural Language Generation; (OOD Robustness)** John Hewitt et al
- **AdapterDrop: On the Efficiency of Adapters in Transformers;** Andreas Rücklé et al; Drop the adapters to speed up the inference time while maintaining performance. 
- **MEASURING THE INTRINSIC DIMENSION OF OBJECTIVE LANDSCAPES;** Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski
- **INTRINSIC DIMENSIONALITY EXPLAINS THE EFFECTIVENESS OF LANGUAGE MODEL FINE-TUNING;** Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta
- **Exploring Low-dimensional Intrinsic Task Subspace via Prompt Tuning**; Yujia Qin et al
- **TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNIN;** Junxian He et al
- **STANDING ON THE SHOULDERS OF GIANT FROZEN LANGUAGE MODELS;** Yoav Levine et al
- **Adaptable Adapters;** Nafise Sadat Moosavi et al
- **Efficient Hierarchical Domain Adaptation for Pretrained Language Models;** Alexandra Chronopoulou et al
- **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning;** Haokun Liu et al
- **LiST: Lite Prompted Self-training Makes Parameter-efficient Few-shot Learners;** Yaqing Wang et al
- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning;** Yi-Lin Sung et al
- **Sparse Structure Search for Parameter-Efficient Tuning;** Shengding Hu et al



## Spurious Correlation
- **Competency Problems: On Finding and Removing Artifacts in Language Data;** Matt Gardner et al
- **Annotation Artifacts in Natural Language Inference Data;** Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, Noah A. Smith
- **Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases;** Christopher Clark, Mark Yatskar, Luke Zettlemoyer
- **Unlearn Dataset Bias in Natural Language Inference by Fitting the Residual;** He He, Sheng Zha, Haohan Wang
- **An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models;** Lifu Tu, Garima Lalwani, Spandana Gella, He He
- **End-to-End Bias Mitigation by Modelling Biases in Corpora;** Rabeeh Karimi Mahabadi, Yonatan Belinkov, James Henderson
- **Predictive Biases in Natural Language Processing Models: A Conceptual Framework and Overview;** Deven Shah, H. Andrew Schwartz, Dirk Hovy
- **Learning to Model and Ignore Dataset Bias with Mixed Capacity Ensembles;** Christopher Clark, Mark Yatskar, Luke Zettlemoyer
- **Don’t Take the Premise for Granted: Mitigating Artifacts in Natural Language Inference;** Yonatan Belinkov et al
- **Towards Debiasing NLU Models from Unknown Biases;** Prasetya Ajie Utama, Nafise Sadat Moosav, Iryna Gurevych
- **LEARNING FROM OTHERS’ MISTAKES: AVOIDING DATASET BIASES WITHOUT MODELING THEM;** Victor Sanh, Thomas Wolf, Yonatan Belinkov, Alexander M. Rush
- **Towards Robustifying NLI Models Against Lexical Dataset Biases;** Xiang Zhou, Mohit Bansal
- **On Adversarial Removal of Hypothesis-only Bias in Natural Language Inference;** Yonatan Belinkov et al
- **Increasing Robustness to Spurious Correlations using Forgettable Examples;** Yadollah Yaghoobzadeh et al
- **Mind the Trade-off: Debiasing NLU Models without Degrading the In-distribution Performance;** Prasetya Ajie Utama, Nafise Sadat Moosavi, Iryna Gurevych
- **Learning Robust Global Representations by Penalizing Local Predictive Power;** Haohan Wang, Songwei Ge, Eric P. Xing, Zachary C. Lipton
- **Invariant Risk Minimization;** Martin Arjovsky, L ́eon Bottou, Ishaan Gulrajani, David Lopez-Paz
- **Examining and Combating Spurious Features under Distribution Shift;** Chunting Zhou, Xuezhe Ma, Paul Michel, Graham Neubig
- **Adversarial Filters of Dataset Biases;** Ronan Le Bras et al; Discuss AFLITE.
- **Generating Data to Mitigate Spurious Correlations in Natural Language Inference Datasets;** Yuxiang Wu et al
- **Combining Feature and Instance Attribution to Detect Artifacts;** Pouya Pezeshkpour et al
- **Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets;** Mor Geva et al
- **Uninformative Input Features and Counterfactual Invariance: Two Perspectives on Spurious Correlations in Natural Language;** Jacob Eisenstein
- **Avoiding inference heuristics in few-shot prompt-based finetuning; (Prompt-based Learning Paradigm)** Prasetya Ajie Utama et al
- **On the Limitations of Dataset Balancing: The Lost Battle Against Spurious Correlations;** Roy Schwartz, Gabriel Stanovsky
- **Are Prompt-based Models Clueless?** Pride Kavumba et al
- **Avoiding Inference Heuristics in Few-shot Prompt-based Finetuning;** Prasetya Ajie Utama et al
- **Partial-input baselines show that NLI models can ignore context, but they don’t;** Neha Srikanth, Rachel Rudinger




## Dataset and Benchmark
- **Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets;** Nelson F. Liu, Roy Schwartz, Noah A. Smith
- **Stress Test Evaluation for Natural Language Inference; (Stress NLI)** Aakanksha Naik et al
- **Adversarial NLI: A New Benchmark for Natural Language Understanding; (ANLI)** Yixin Nie et al
- **Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference; (HANS)** R. Thomas McCoy, Ellie Pavlick, Tal Linzen
- **PAWS: Paraphrase Adversaries from Word Scrambling; (PAWS)** Yuan Zhang, Jason Baldridge, Luheng He
- **Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models;** Boxin Wang et al
- **What will it take to fix benchmarking in natural language understanding?;** Samuel R. Bowman, George E. Dahl
- **GLUE: A MULTI-TASK BENCHMARK AND ANALYSIS PLATFORM FOR NATURAL LANGUAGE UNDERSTANDING;** Alex Wang et al
- **From Hero to Zeroe: A Benchmark of Low-Level Adversarial Attacks;** Steffen Eger, Yannik Benz
- **Quantifying Adaptability in Pre-trained Language Models with 500 Tasks;** Belinda Z. Li et al
- **Dynabench: Rethinking Benchmarking in NLP;** Douwe Kiela et al
- **Dynatask: A Framework for Creating Dynamic AI Benchmark Tasks;** Tristan Thrush et al
- **NUMGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks;** Swaroop Mishra et al
- **WINOGRANDE: An Adversarial Winograd Schema Challenge at Scale; (AFLITE)** Keisuke Sakaguchi et al
- **Swag: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference; (AF)** Rowan Zellers, Yonatan Bisk, Roy Schwartz, Yejin Choi
- **ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models;** Chunyuan Li et al
- **KILT: a Benchmark for Knowledge Intensive Language Tasks;** Fabio Petroni et al
- **DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs;** Dheeru Dua et al
- **TRUE: Re-evaluating Factual Consistency Evaluation;** Or Honovich et al
- **CURRICULUM: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding;** Zeming Chen, Qiyue Gao
- **ERASER: A Benchmark to Evaluate Rationalized NLP Models;** Jay DeYoung et al
- **VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models;** Wangchunshu Zhou et al



## Model Analysis
- **A Closer Look at How Fine-tuning Changes BERT;** Yichu Zhou, Vivek Srikumar; Analyze fine-tuning. 
- **Word Order Does Matter (And Shuffled Language Models Know It);** Vinit Ravishankar et al
- **BERT Rediscovers the Classical NLP Pipeline;** Ian Tenney, Dipanjan Das, Ellie Pavlick
- **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space;** Thomas Muller et al
- **A Primer in BERTology: What We Know About How BERT Works;** Anna Rogers, Olga Kovaleva, Anna Rumshisky 
- **Is BERT a Cross-Disciplinary Knowledge Learner? A Surprising Finding of Pre-trained Models’ Transferability;** Wei-Tsung Kao, Hung-yi Lee







## Language and robotics
- **Survey on frontiers of language and robotics; (Survey)** T. Taniguchi et al
- **Grounded Language Learning: Where Robotics and NLP Meet;** Cynthia Matuszek
- **Spoken language interaction with robots: Recommendations for future research; (Survey)** Matthew Marge et al
- **Language to Action: Towards Interactive Task Learning with Physical Agents; (Survey)** Joyce Y. Chai et al
- **Continual Learning for Grounded Instruction Generation by Observing Human Following Behavior;** Noriyuki Kojima, Alane Suhr, Yoav Artzi
- **Teaching Robots New Tasks through Natural Interaction; (Survey)** Joyce Y. Chai, Maya Cakmak, Candace Sidner
- **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback;** Yuntao Bai et al
- **A Persistent Spatial Semantic Representation for High-level Natural Language Instruction Execution;** Valts Blukis et al
- **Analysis of Language Change in Collaborative Instruction Following;** Anna Effenberger et al
- **Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight;** Valts Blukis et al
- **Few-shot Object Grounding and Mapping for Natural Language Robot Instruction Following;** Valts Blukis et al
- **ALFRED A Benchmark for Interpreting Grounded Instructions for Everyday Tasks;** Mohit Shridhar et al; Create the dataset ALFRED.
- **Learning to Execute Actions or Ask Clarification Questions;** Zhengxiang Shi et al
- **Help, Anna! Visual Navigation with Natural Multimodal Assistance via Retrospective Curiosity-Encouraging Imitation Learning;** Khanh Nguyen, Hal Daume III; Create the dataset HANNA. 
- **Vision-and-Dialog Navigation;** Jesse Thomason et al; Create the dataset CVDN.
- **Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents;** Wenlong Huang et al
- **Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation;** Vihan Jain et al
- **Asking for Knowledge: Training RL Agents to Query External Knowledge Using Language;** Iou-Jen Liu et al
- **Few-shot Subgoal Planning with Language Models;** Lajanugen Logeswaran et al





## Multimodal
- **Visually Grounded Neural Syntax Acquisition;** Haoyue Shi et al
- **What is Learned in Visually Grounded Neural Syntax Acquisition;** Noriyuki Kojima et al
- **CLMLF:A Contrastive Learning and Multi-Layer Fusion Method for Multimodal Sentiment Detection;** Zhen Li et al
- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision;** Hao Tan, Mohit Bansal
- **Learning Visually-Grounded Semantics from Contrastive Adversarial Samples;** Haoyue Shi et al
- **UniVSE: Robust Visual Semantic Embeddings via Structured Semantic Representations;** Hao Wu et al
- **Visual Referring Expression Recognition: What Do Systems Actually Learn?;** Volkan Cirik et al
- **K-LITE: Learning Transferable Visual Models with External Knowledge;** Sheng Shen et al
- **i-Code: An Integrative and Composable Multimodal Learning Framework;** Ziyi Yang et al
- **Flamingo: a Visual Language Model for Few-Shot Learning;** Jean-Baptiste Alayrac et al
- **Multimodal Knowledge Alignment with Reinforcement Learning;** Youngjae Yu et al
- **BRAINISH: FORMALIZING A MULTIMODAL LANGUAGE FOR INTELLIGENCE AND CONSCIOUSNESS;** Paul Pu Liang
- **A Unified Continuous Learning Framework for Multi-modal Knowledge Discovery and Pre-training;** Zhihao Fan et al
- **OFA: UNIFYING ARCHITECTURES, TASKS, AND MODALITIES THROUGH A SIMPLE SEQUENCE-TO-SEQUENCE LEARNING FRAMEWORK;** Peng Wang et al

## Scene Graph
- **Visual Distant Supervision for Scene Graph Generation;** Yuan Yao et al
- **Learning to Generate Scene Graph from Natural Language Supervision;** Yiwu Zhong et al
- **Weakly Supervised Visual Semantic Parsing;** Alireza Zareian, Svebor Karaman, Shih-Fu Chang
- **Scene Graph Prediction with Limited Labels;** Vincent S. Chen, Paroma Varma, Ranjay Krishna, Michael Bernstein, Christopher Re, Li Fei-Fei



## NLP Reasoning
- **Visual Goal-Step Inference using wikiHow;** Yue Yang et al
- **Goal-Oriented Script Construction;** Qing Lyu et al
- **Reasoning about Goals, Steps, and Temporal Ordering with WikiHow;** Qing Lyu et al
- **Chain of Thought Prompting Elicits Reasoning in Large Language Models;** Jason Wei et al
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models;** Xuezhi Wang et al
- **Evaluating Commonsense in Pre-trained Language Models; (Commonsense)** Xuhui Zhou et al
- **Do Neural Language Representations Learn Physical Commonsense?; (Commonsense)** Maxwell Forbes et al
- **COMMONSENSEQA: A Question Answering Challenge Targeting Commonsense Knowledge; (Commonse Benchmark)** Alon Talmor et al
- **Inferring and Executing Programs for Visual Reasoning;** Justin Johnson et al
- **Relational World Knowledge Representation in Contextual Language Models: A Review;** Tara Safavi, Danai Koutra
- **Leap-Of-Thought: Teaching Pre-Trained Models to Systematically Reason Over Implicit Knowledge;** Alon Talmor et al
- **Probing Script Knowledge from Pre-Trained Models;** Zijia Jin et al
- **Towards Teachable Reasoning Systems;** Bhavana Dalvi, Oyvind Tafjord, Peter Clark
- **Inferring Implicit Relations with Language Models;** Uri Katz et al
- **OPERA: Operation-Pivoted Discrete Reasoning over Text;** Yongwei Zhou et al
- **Entailment Tree Explanations via Iterative Retrieval-Generation Reasoner;** Danilo Neves Ribeiro et al
- **Penguins Don’t Fly: Reasoning about Generics through Instantiations and Exceptions; (Commonsense)** Emily Allaway et al
- **Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations;** Jaehun Jung et al
- **Impact of Pretraining Term Frequencies on Few-Shot Reasoning;** Yasaman Razeghi et al



## CV Reasoning
- **MERLOT: Multimodal Neural Script Knowledge Models;** Rowan Zellers et al
- **MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound;** Rowan Zellers et al
- **THE NEURO-SYMBOLIC CONCEPT LEARNER: INTERPRETING SCENES, WORDS, AND SENTENCES FROM NATURAL SUPERVISION; (Vision Reasoning)** Jiayuan Mao et al
- **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning; (Image Reasoning Benchmark)** Justin Johnson et al
- **A Corpus for Reasoning About Natural Language Grounded in Photographs; (Image & Language Reasoning)** Alane Suhr et al
- **From Recognition to Cognition: Visual Commonsense Reasoning; (Image & Language Reasoning)** Rowan Zellers et al






## MRC Reasoning
- **NEURAL MODULE NETWORKS FOR REASONING OVER TEXT;** Nitish Gupta et al
- **NEURAL SYMBOLIC READER: SCALABLE INTEGRATION OF DISTRIBUTED AND SYMBOLIC REPRESENTATIONS FOR READING COMPREHENSION;** Xinyun Chen et al
- **Is a Question Decomposition Unit All We Need?;** Pruthvi Patel et al; Decompose a hard question into several easy ones via human-in-the-loop.




## Grounding
- **Climbing towards NLU: On meaning, form, and understanding in the age of data;** Emily M. Bender, Alexander Koller
- **Provable Limitations of Acquiring Meaning from Ungrounded Form: What Will Future Language Models Understand?;** William Merrill et al
- **What Does BERT with Vision Look At?;** Liunian Harold Li et al
- **Visual Grounding Strategies for Text-Only Natural Language Processing;** Damien Sileo; Discuss how multi-modal pretraining improves NLU tasks. 
- **Experience Grounds Language;** Yonatan Bisk et al
- **ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension;** Sanjay Subramanian et al




## NLG Hallucination
- **On Faithfulness and Factuality in Abstractive Summarization;** Joshua Maynez et al
- **Entity-Based Knowledge Conflicts in Question Answering;** Shayne Longpre et al; QA Task hallucination. 
- **Evaluating the Factual Consistency of Abstractive Text Summarization;** Wojciech Krys ́cin ́ski, Bryan McCann, Caiming Xiong, Richard Socher
- **Annotating and Modeling Fine-grained Factuality in Summarization;** Tanya Goyal, Greg Durrett
- **FACTPEGASUS: Factuality-Aware Pre-training and Fine-tuning for Abstractive Summarizationl** David Wan, Mohit Bansal
- **Evidentiality-guided Generation for Knowledge-Intensive NLP Tasks;** Akari Asai et al






## Interesting Topics
- **Advancing mathematics by guiding human intuition with AI;** Alex Davies et al
- **What are the best Systems? New Perspectives on NLP Benchmarking;** Pierre Colombo, Nathan Noiry, Ekhine Irurozki, Stephan Clemencon; Study how to aggregate metrics in multi-task evaluation. "benchmarks are made of datasets, metrics, and a way to aggregate performance. ... If the bulk of the NLP community efforts on this domain is about collecting new datasets and introducing new metrics, little work is concerned with the third part, namely how to aggregate various performances."
- **A Neural-Symbolic Approach to Natural Language Understanding;** Zhixuan Liu et al
- **Fast Few-shot Debugging for NLU Test Suites;** Christopher Malon et al
- **Shedding New Light on the Language of the Dark Web;** Youngjin Jin et al
- **Training language models to follow instructions with human feedback;** Long Ouyang et al; Directly use RL to train language generation models according to the human feedback. 
- **Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training;** Yuanxin Liu et al
- **Machine-in-the-Loop Rewriting for Creative Image Captioning;** Vishakh Padmakumar, He He
- **Describing Differences between Text Distributions with Natural Language;** Ruiqi Zhong et al
- **The Dangers of Underclaiming: Reasons for Caution When Reporting How NLP Systems Fail;** Samuel R. Bowman





## Resources
- **Transfer Learning in NLP; (Tutorial)** https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5882add69e_6_646; Gives a hands-on code to conduct pre-training.
- **Robustness and Adversarial Examples in NLP; (Tutorial)** https://docs.google.com/presentation/d/1E_0qEwQkS43FJGzOEUrpee9zqi8y5lx6D-ABQl3KFas/edit#slide=id.p
- **How to write research paper?** https://students.uu.nl/sites/default/files/ge0-aw-guide-for-scientific-writing-2016.pdf
- **CausalNLP: A Practical Toolkit for Causal Inference with Text;** Arun S. Maiya
- **Challenges and Opportunities in NLP Benchmarking;** Sebastian Ruder
