# PaperList

## Table of Contents 
- [Survey](#survey)
- [Adversarial Sample](#adversarial-sample)
- [Adversarial Training](#adversarial-training)
- [Calibration and Uncertainty](#calibration-and-uncertainty) 
- [Compositional Generalization](#compositional-generalization) 
- [Trustworthy (broad topic)](#trustworthy-(broad-topic))
- [Noisy Label](#noisy-label) 
- [Fairness](#fairness)
- [OOD/Anomaly/Openset Detection](#ood/anomaly/openset-detection)
- [Robustness](#robustness) 
- [Explanation](#explanation)
- [Data Augmentation](#data-augmentation)
- [Training Data Points](#training-data-points)
- [Multi-task & Transfer & Lifelong & Few-shot & Zero-shot Learning](#transfer-&-lifelong-&-few-shot-&-zero-shot-learning)
- [Backdoor Learning](#backdoor-learning)
- [Prompt-based Learning](#prompt-based-learning)
- [Parameter Efficient Tuning](#parameter-efficient-tuning)
- [Spurious Correlation](#spurious-correlation)
- [NLP for Social Good](#nlp-for-social-good)
- [Dataset and Benchmark](#dataset-and-benchmark)
- [Foundation Model](#foundation-model)
- [Incontext Learning](#incontext-learning)
- [Model Analysis](#model-analysis)
- [Theory](#theory)
- [Language and Robotics](#language-and-robotics)
- [Multimodal](#multimodal)
- [Scene Graph](#scene-graph)
- [NLP Reasoning](#nlp-reasoning)
- [CV Reasoning](#cv-reasoning)
- [MRC Reasoning](#mrc-reasoning)
- [Grounding](#grounding) 
- [NLG Hallucination](#nlg-hallucination)
- [Text Editing](#text-editing)
- [Information Extraction](#information-extraction)
- [Retrieval-augmented LLM](#retrieval-augmented-llm)
- [Code](#code)
- [Security of LLM](#security-of-llm)
- [Interesting Topics](#interesting-topics)
- [Learning](#learning)
- [Interesting Fields (CV)](#interesting-fields (CV))
- [Resources](#resources) 




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
- **Modern Question Answering Datasets and Benchmarks: A Survey;** Zhen Wang 
- **A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges;** Mohammadreza Salehi et al
- **Survey on deep learning with class imbalance;** Justin M. Johnson, Taghi M. Khoshgoftaar
- **WHAT IS THE STATE OF NEURAL NETWORK PRUNING?;** Davis Blalock et al
- **Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond;** Amir Feder et al
- **Multimodal Learning with Transformers: A Survey;** Peng Xu, Xiatian Zhu, David A. Clifton
- **Teach Me to Explain: A Review of Datasets for Explainable Natural Language Processing;** Sarah Wiegreffe, Ana Marasovic
- **Neuron-level Interpretation of Deep NLP Models: A Survey;** Hassan Sajjad et al
- **A Compact Survey on Event Extraction: Approaches and Applications;** Qian Li et al 
- **Efficient Methods for Natural Language Processing: A Survey;** Marcos Treviso et al
- **A Survey on Measuring and Mitigating Reasoning Shortcuts in Machine Reading Comprehension;** Xanh Ho et al
- **FOUNDATIONS & RECENT TRENDS IN MULTIMODAL MACHINE LEARNING: PRINCIPLES, CHALLENGES, & OPEN QUESTIONS;** Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency
- **A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities;** XINYI ZHOU et al
- **A Survey on Automated Fact-Checking;** Zhijiang Guo et al
- **Automated Fact Checking: Task formulations, methods and future directions;** James Thorne et al
- **Advances and Open Problems in Federated Learning;** Peter Kairouz et al
- **Event Extraction: A Survey;** Viet Dac Lai et al
- **State-of-the-art generalisation research in NLP: a taxonomy and review;** Dieuwke Hupkes et al
- **A Survey of Active Learning for Natural Language Processing;** Zhisong Zhang et al
- **A Survey of Data Optimization for Problems in Computer Vision Datasets;** Zhijing Wan et al
- **Towards Data-and Knowledge-Driven Artificial Intelligence: A Survey on Neuro-Symbolic Computing;** Wenguan Wang et al
- **Self-Training: A Survey;** Massih-Reza Amini et al
- **On the Domain Adaptation and Generalization of Pretrained Language Models: A Survey;** Xu Guo et al
- **A Survey of Knowledge-Enhanced Pre-trained Language Models;** Linmei Hu et al
- **Continual Learning of Natural Language Processing Tasks: A Survey;** Zixuan Ke et al
- **A Survey on Model Compression and Acceleration for Pretrained Language Models;** Canwen Xu et al
- **Learning from Disagreement: A Survey;** Alexandra N. Uma et al
- **A Survey on Natural Language Processing for Programming;** Qingfu Zhu et al
- **When Neural Model Meets NL2Code: A Survey;** Daoguang Zan et al
- **Towards Reasoning in Large Language Models: A Survey;** Jie Huang et al
- **Reasoning with Language Model Prompting: A Survey;** Shuofei Qiao et al
- **A Survey for In-context Learning;** Qingxiu Dong et al
- **The Life Cycle of Knowledge in Big Language Models: A Survey;** Boxi Cao et al
- **Sparks of Artificial General Intelligence: Early experiments with GPT-4;** S´ebastien Bubeck et al


## Adversarial Sample
- **Repairing Adversarial Texts through Perturbation;** Guoliang Dong, Jingyi Wang, Jun Sun, Sudipta Chattopadhyay, Xinyu Wang, Ting Dai, Jie Shi and Jin Song Dong; Introduce method to detect textual adversarial samples and "repair" them. 
- **Adversarial Examples Are Not Bugs, They Are Features;** Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, Aleksander Madry
- **Detecting Adversarial Samples from Artifacts; (Detection)** Reuben Feinman, Ryan R. Curtin, Saurabh Shintre, Andrew B. Gardner
- **Detection of Adversarial Examples in NLP: Benchmark and Baseline via Robust Density Estimation；** Anonymous
- **Reevaluating Adversarial Examples in Natural Language; (Evaluation)** John X. Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi; Discuss how to align automatic evaluation metrics with human intuition; and integrate automatic metrics in the searching process to preserve quality and validity of adversarial samples. 
- **Evaluating the Robustness of Neural Language Models to Input Perturbations;** Milad Moradi, Matthias Samwald; Benchmark models' robustness to real-world user inputs; they use adversarial samples to simulate user inputs. 
- **Interpreting the Robustness of Neural NLP Models to Textual Perturbations;** Yunxiang Zhang et al
- **Distinguishing Non-natural from Natural Adversarial Samples for More Robust Pre-trained Language Model;** Anonymous; Using outlier detection to filter non-natural adversarial samples. 
- **Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense;** Thai Le et al
- **Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification;** Cheng-Han Chiang, Hung-yi Lee
- **Consistency Training with Virtual Adversarial Discrete Perturbation;** Jungsoo Park et al
- **Improving Robustness of Language Models from a Geometry-aware Perspective;** Bin Zhu et al
- **AEON: A Method for Automatic Evaluation of NLP Test Cases;** Jen-tse Huang et al; Validity verification. 
- **It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations;** Samson Tan et al
- **Tailor: Generating and Perturbing Text with Semantic Controls;** Alexis Ross et al 
- **TEXTGRAD: ADVANCING ROBUSTNESS EVALUATION IN NLP BY GRADIENT-DRIVEN OPTIMIZATION;** Bairu Hou et al
- **In and Out-of-Domain Text Adversarial Robustness via Label Smoothing;** Yahan Yang et al
- **Learning the Legibility of Visual Text Perturbations;** Dev Seth et al


## Adversarial Training
- **ADVERSARIAL SELF-ATTENTION FOR LANGUAGE UNDERSTANDING;** Hongqiu Wu, Hai Zhao
- **Token-Aware Virtual Adversarial Training in Natural Language Understanding;** Linyang Li, Xipeng Qiu
- **AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples;** Dongyeop Kang et al
- **Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge;** Pasquale Minervini, Sebastian Riedel
- **Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks;** Zhao Meng et al;
- **Overfitting in adversarially robust deep learning;** Leslie Rice et al
- **Impact of Adversarial Training on Robustness and Generalizability of Language Models;** Enes Altinisik et al

## Calibration and Uncertainty
- **On Calibration of Modern Neural Networks;** Chuan Guo et al
- **Explaining Prediction Uncertainty of Pre-trained Language Models by Detecting Uncertain Words in Inputs** Hanjie Chen, Yangfeng Ji 
- **Calibration of Pre-trained Transformers;** Shrey Desai, Greg Durrett; Empirically study the calibration of PLMs, in both in-domain & out-out-domain. Also include label smoothing and temperature scaling in the experiments. 
- **Teaching models to express their uncertainty in words;** Stephanie Lin et al
- **Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data;** Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, Chao Zhang
- **Revisiting Calibration for Question Answering;** Chenglei Si et al
- **On the Calibration of Pre-trained Language Models using Mixup Guided by Area Under the Margin and Saliency;** Seo Yeon Park, Cornelia Caragea
- **Language Models (Mostly) Know What They Know;** Saurav Kadavath et al
- **Investigating Selective Prediction Approaches Across Several Tasks in IID, OOD, and Adversarial Settings;** Neeraj Varshney, Swaroop Mishra, Chitta Baral
- **Uncertainty Quantification with Pre-trained Language Models: A Large-Scale Empirical Analysis;** Anonymous
- **On the Effects of Transformer Size on In- and Out-of-Domain Calibration;** Soham Dan, Dan Roth
- **Calibrating Structured Output Predictors for Natural Language Processing;** Abhyuday Jagannatha, Hong Yu
- **Intrinsic Uncertainty-Aware Calibration Metric;** Anonymous 
- **IN DEFENSE OF PSEUDO-LABELING: AN UNCERTAINTY-AWARE PSEUDO-LABEL SELECTION FRAMEWORK FOR SEMI-SUPERVISED LEARNING;** Mamshad Nayeem Rizve et al; Application. 
- **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles;** Balaji Lakshminarayanan et al
- **Uncertainty Estimation of Transformer Predictions for Misclassification Detection;** Artem Vazhentsev et al
- **Evaluating model calibration in classification;** Juozas Vaicenavicius et al
- **CALIBRATION OF NEURAL NETWORKS USING SPLINES;** Kartik Gupta et al
- **Mitigating Bias in Calibration Error Estimation;** Rebecca Roelofs et al
- **Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning;** Jize Zhang et al
- **Revisiting the Calibration of Modern Neural Networks;** Matthias Minderer et al
- **Measuring Calibration in Deep Learning;** Jeremy Nixon et al
- **Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift;** Yaniv Ovadia et al
- **Calibrated ensembles can mitigate accuracy tradeoffs under distribution shift;** Ananya Kumar et al
- **TOP-LABEL CALIBRATION AND MULTICLASS-TO-BINARY REDUCTIONS;** Chirag Gupta, Aaditya Ramdas
- **Can Explanations Be Useful for Calibrating Black Box Models?;** Xi Ye, Greg Durrett
- **Confidence Modeling for Neural Semantic Parsing;** Li Dong et al
- **Knowing More About Questions Can Help: Improving Calibration in Question Answering;** Shujian Zhang et al
- **IMPROVING THE CALIBRATION OF FINE-TUNED LANGUAGE MODELS VIA DENOISING VARIATIONAL AUTO-ENCODERS;** Anonymous
- **TO SOFTMAX, OR NOT TO SOFTMAX: THAT IS THE QUESTION WHEN APPLYING ACTIVE LEARNING FOR TRANSFORMER MODELS;** Julius Gonsior et al
- **Model Cascading: Towards Jointly Improving Efficiency and Accuracy of NLP Systems;** Neeraj Varshney et al
- **Improving the Reliability for Confidence Estimation;** Haoxuan Qu et al
- **To Trust Or Not To Trust A Classifier;** Heinrich Jiang et al
- **Addressing Failure Prediction by Learning Model Confidence;** Charles Corbière et al
- **Revisiting Uncertainty-based Query Strategies for Active Learning with Transformers;** Christopher Schröder et al
- **Stop Measuring Calibration When Humans Disagree;** Joris Baan et al
- **On the Calibration of Massively Multilingual Language Models;** Kabir Ahuja et al
- **ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation;** Fan Yin et al
- **Hard Gate Knowledge Distillation-Leverage Calibration for a Robust and Reliable Language Model;** Dongkyu Lee et al
- **Calibrating Deep Neural Networks using Focal Loss;** Jishnu Mukhoti et al
- **Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity;** Dennis Ulmer et al
- **CascadeBERT: Accelerating Inference of Pre-trained Language Models via Calibrated Complete Models Cascade;** Lei Li et al
- **Calibration Meets Explanation: A Simple and Effective Approach for Model Confidence Estimates;** Dongfang Li et al
- **AdaFocal: Calibration-aware Adaptive Focal Loss;** Arindam Ghosh et al
- **Calibrated Interpretation: Confidence Estimation in Semantic Parsing;** Elias Stengel-Eskin et al
- **CONAL: ANTICIPATING OUTLIERS WITH LARGE LANGUAGE MODELS;** Albert Xu et al 
- **DOCTOR: A Simple Method for Detecting Misclassification Errors;** Federica Granese et al
- **Robust Models are less Over-Confident;** Julia Grabinski et al
- **Holistic Evaluation of Language Models;** Percy Liang et al
- **Confident Adaptive Language Modeling;** Tal Schuster et al
- **Adversarial Unlearning: Reducing Confidence Along Adversarial Directions;** Amrith Setlur et al
- **Bag of Tricks for In-Distribution Calibration of Pretrained Transformers;** Jaeyoung Kim et al
- **Navigating the Grey Area: Expressions of Overconfidence and Uncertainty in Language Models;** Kaitlyn Zhou et al



## Compositional Generalization
- **Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks;** Brenden Lake, Marco Baroni
- **MEASURING COMPOSITIONAL GENERALIZATION: A COMPREHENSIVE METHOD ON REALISTIC DATA;** Daniel Keysers et al
- **Improving Compositional Generalization in Classification Tasks via Structure Annotations;** Juyong Kim et al
- **MEASURING COMPOSITIONALITY IN REPRESENTATION LEARNING;** Jacob Andreas
- **Revisiting the Compositional Generalization Abilities of Neural Sequence Models;** Arkil Patel et al
- **Learning Transductions to Test Systematic Compositionality;** Josef Valvoda et al
- **CHARACTERIZING INTRINSIC COMPOSITIONALITY IN TRANSFORMERS WITH TREE PROJECTIONS;** Shikhar Murty et al
- **CREPE: Can Vision-Language Foundation Models Reason Compositionally?;** Zixian Ma et al


## Trustworthy (broad topic)
- **Systematicity, Compositionality and Transitivity of Deep NLP Models: a Metamorphic Testing Perspective;** Edoardo Manino et al
- **A Metamorphic Testing Approach for Assessing Question Answering Systems;** Kaiyi Tu et al
- **White-box Testing of NLP models with Mask Neuron Coverage;** Arshdeep Sekhon et al; Evaluation. Employ white-box information to reduce test cases. The intuition is to identify when an input’s activation of attention neurons is subsumed by that of prior test inputs. 
- **PIXMIX: Dreamlike Pictures Comprehensively Improve Safety Measures;** Dan Hendrycks et al
- **METASHIFT: A DATASET OF DATASETS FOR EVALUATING CONTEXTUAL DISTRIBUTION SHIFTS AND TRAINING CONFLICTS;** Weixin Liang, James Zou
- **TOXIGEN: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection;** Thomas Hartvigsen et al
- **NON-TRANSFERABLE LEARNING: A NEW APPROACH FOR MODEL OWNERSHIP VERIFICATION AND APPLICABILITY AUTHORIZATION;** Lixu Wang et al
- **Plex: Towards Reliability Using Pretrained Large Model Extensions;** Dustin Tran et al
- **Recovering Private Text in Federated Learning of Language Models;** Samyak Gupta et al
- **LEVERAGING UNLABELED DATA TO PREDICT OUT-OF-DISTRIBUTION PERFORMANCE;** Saurabh Garg et al
- **Formalizing Trust in Artificial Intelligence: Prerequisites, Causes and Goals of Human Trust in AI;** Alon Jacovi et al
- **Memorization in NLP Fine-tuning Methods;** Fatemehsadat Mireshghallah et al
- **SEAL: Interactive Tool for Systematic Error Analysis and Labeling;** Nazneen Rajani et al
- **DOMINO: DISCOVERING SYSTEMATIC ERRORS WITH CROSS-MODAL EMBEDDINGS;** Sabri Eyuboglu et al
- **Discover, Explanation, Improvement: Automatic Slice Detection Framework for Natural Language Processing;** Wenyue Hua et al
- **Capturing Failures of Large Language Models via Human Cognitive Biases;** Erik Jones et al
- **Foveate, Attribute, and Rationalize: Towards Safe and Trustworthy AI;** Alex Mei et al
- **Are Red Roses Red? Evaluating Consistency of Question-Answering Models;** Marco Tulio Ribeiro et al



## Noisy Label
- **Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels;** Bo Han et al; Propose a training algorithm that combat the noisy labels issue. 
- **Learning to Reweight Examples for Robust Deep Learning;** Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun
- **DIVIDEMIX: LEARNING WITH NOISY LABELS AS SEMI-SUPERVISED LEARNING;** Junnan Li, Richard Socher, Steven C.H. Hoi
- **Detecting Label Errors using Pre-Trained Language Models;** Derek Chong et al
- **Protoformer: Embedding Prototypes for Transformers;** Ashkan Farhangi et al
- **ROBUST EARLY-LEARNING: HINDERING THE MEMORIZATION OF NOISY LABELS;** Xiaobo Xia et al
- **SAMPLE SELECTION WITH UNCERTAINTY OF LOSSES FOR LEARNING WITH NOISY LABELS;** Xiaobo Xia et al





## Fairness
- **Men Also Like Shopping: Reducing Gender Bias Amplification using Corpus-level Constraints;** Jieyu Zhao et al 
- **Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings;** Tolga Bolukbasi et al
- **Easy Adaptation to Mitigate Gender Bias in Multilingual Text Classification;** Xiaolei Huang
- **Data Feedback Loops: Model-driven Amplification of Dataset Biases;** Rohan Taori, Tatsunori B. Hashimoto

## OOD/Anomaly/Openset Detection
- **Contrastive Out-of-Distribution Detection for Pretrained Transformers;** Wenxuan Zhou, Fangyu Liu, Muhao Chen
- **Types of Out-of-Distribution Texts and How to Detect Them;** Udit Arora, William Huang and He He; Analyze two types of OOD data and benchmark two popular OOD detection methods, get some interesting findings. 
- **Towards Textual Out-of-Domain Detection without In-Domain Labels;** Di Jin, Shuyang Gao, Seokhwan Kim, Yang Liu, Dilek Hakkani-Tur
- **Scaling Out-of-Distribution Detection for Real-World Settings;** Dan Hendrycks et al
- **OPEN-SET RECOGNITION: A GOOD CLOSED-SET CLASSIFIER IS ALL YOU NEED?;** Sagar Vaze et al
- **AD-NLP: A Benchmark for Anomaly Detection in Natural Language Processing;** Matei Bejan et al
- **OpenOOD: Benchmarking Generalized Out-of-Distribution Detection;** Jingkang Yang et al
- **Enhancing Out-of-Distribution Detection in Natural Language Understanding via Implicit Layer Ensemble;** Hyunsoo Cho et al
- **Are Out-of-Distribution Detection Methods Reliable?;** Vahid Reza Khazaie et al
- **ADBench: Anomaly Detection Benchmark;** Songqiao Han et al
- **Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text;** Qianhui Wu et al
- **Beyond Mahalanobis-Based Scores for Textual OOD Detection;** Pierre Colombo et al
- **Delving into Out-of-Distribution Detection with Vision-Language Representations;** Yifei Ming et al
- **Out-Of-Distribution Detection Is Not All You Need;** Joris Guerin et al
- **Rethinking Out-of-Distribution Detection From a Human-Centric Perspective;** Yao Zhu et al



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
- **Methods for Estimating and Improving Robustness of Language Models; (Generalization)** Michal Štefánik
- **A FINE-GRAINED ANALYSIS ON DISTRIBUTION SHIFT;** Olivia Wiles et al
- **The Effect of Natural Distribution Shift on Question Answering Models;** John Miller et al
- **Selective Question Answering under Domain Shift;** Amita Kamath, Robin Jia, Percy Liang
- **Towards a Theoretical Framework of Out-of-Distribution Generalization;** Haotian Ye et al
- **MULTIQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension;** Alon Talmor, Jonathan Berant
- **On the Robustness of Reading Comprehension Models to Entity Renaming;** Jun Yan et al
- **RockNER: A Simple Method to Create Adversarial Examples for Evaluating the Robustness of Named Entity Recognition Models;** Bill Yuchen Lin et al
- **Learning Stable Classifiers by Transferring Unstable Features;** Yujia Bao et al
- **Time Waits for No One! Analysis and Challenges of Temporal Misalignment;** Kelvin Luu et al
- **VL-CheckList: Evaluating Pre-trained Vision-Language Models with Objects, Attributes and Relations;** Tiancheng Zhao et al
- **DISTRIBUTIONALLY ROBUST NEURAL NETWORKS FOR GROUP SHIFTS: ON THE IMPORTANCE OF REGULARIZATION FOR WORST-CASE GENERALIZATION;** Shiori Sagawa et al; Subpopulation shift. 
- **NoiseQA: Challenge Set Evaluation for User-Centric Question Answering;** Abhilasha Ravichander et al
- **Semantically Distributed Robust Optimization for Vision-and-Language Inference;** Tejas Gokhale et al
- **Detecting and Correcting for Label Shift with Black Box Predictors;** Zachary C. Lipton et al; Label shift. 
- **LTF: A Label Transformation Framework for Correcting Target Shift;** Jiaxian Guo et al; Label shift. 
- **A Unified View of Label Shift Estimation;** Saurabh Garg et al; Label shift. 
- **Two-Stage Fine-Tuning: A Novel Strategy for Learning Class-Imbalanced Data;** Taha ValizadehAslani et al
- **SELF-SUPERVISED LEARNING IS MORE ROBUST TO DATASET IMBALANCE;** Hong Liu et al
- **SPARSITY WINNING TWICE: BETTER ROBUST GENERALIZATION FROM MORE EFFICIENT TRAINING;** Tianlong Chen et al
- **BACK-TO-BONES: REDISCOVERING THE ROLE OF BACKBONES IN DOMAIN GENERALIZATION;** Simone Angarano et al
- **In Search of Lost Domain Generalization;** Ishaan Gulrajani, David Lopez-Paz
- **Domain Adaptation for Question Answering via Question Classification;** Zhenrui Yue  et al
- **ENSEMBLES AND COCKTAILS: ROBUST FINETUNING FOR NATURAL LANGUAGE GENERATION;** John Hewitt et al
- **Robust fine-tuning of zero-shot models;** Mitchell Wortsman et al
- **Are Sample-Efficient NLP Models More Robust?;** Nelson F. Liu et al
- **On the Impact of Temporal Concept Drift on Model Explanations;** Zhixue Zhao et al
- **TestAug: A Framework for Augmenting Capability-based NLP Tests;** Guanqun Yang et al
- **Balanced Adversarial Training: Balancing Tradeoffs between Fickleness and Obstinacy in NLP Models;** Hannah Chen et al
- **Robustifying Sentiment Classification by Maximally Exploiting Few Counterfactuals;** Maarten De Raedt et al
- **Exploring The Landscape of Distributional Robustness for Question Answering Models;** Anas Awadalla et al
- **NeuroCounterfactuals: Beyond Minimal-Edit Counterfactuals for Richer Data Augmentation;** Phillip Howard et al
- **Not to Overfit or Underfit? A Study of Domain Generalization in Question Answering;** Md Arafat Sultan et al
- **Are Sample-Efficient NLP Models More Robust?;** Nelson F. Liu et al
- **Finetune like you pretrain: Improved finetuning of zero-shot vision models;** Sachin Goyal et al
- **NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation;** Kaustubh D. Dhole et al
- **Reliability Testing for Natural Language Processing Systems;** Samson Tan et al
- **If your data distribution shifts, use self-learning;** Evgenia Rusak et al
- **RobustBench: a standardized adversarial robustness benchmark;** Francesco Croce et al
- **Assaying Out-Of-Distribution Generalization in Transfer Learning;** Florian Wenzel et al
- **Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift;** Christina Baek et al
- **Benchmark for Uncertainty & Robustness in Self-Supervised Learning;** Ha Manh Bui et al
- **Effective Robustness against Natural Distribution Shifts for Models with Different Training Data;** Zhouxing Shi et al
- **Robust Question Answering against Distribution Shifts with Test-Time Adaptation: An Empirical Study;** Hai Ye et al 
- **Dynamic Benchmarking of Masked Language Models on Temporal Concept Drift with Multiple Views;** Katerina Margatina et al
- **ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing;** Xiaodan Li et al


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
- **Compositional Explanations of Neurons;** Jesse Mu, Jacob Andreas
- **NATURAL LANGUAGE DESCRIPTIONS OF DEEP VISUAL FEATURES;** Evan Hernandez et al
- **When Can Models Learn From Explanations? A Formal Framework for Understanding the Roles of Explanation Data;** Peter Hase, Mohit Bansal
- **Connecting Attributions and QA Model Behavior on Realistic Counterfactuals;** Xi Ye et al
- **CausaLM: Causal Model Explanation Through Counterfactual Language Models;** Amir Feder et al
- **CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior;** Eldar David Abraham et al
- **Interpreting Language Models with Contrastive Explanations;** Kayo Yin et al

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
- **Rethinking Data Augmentation for Robust Visual Question Answering;** Long Chen et al
- **WHAT MAKES BETTER AUGMENTATION STRATEGIES? AUGMENT DIFFICULT BUT NOT TOO DIFFERENT;** Jaehyung Kim et al



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
- **Understanding Dataset Difficulty with V-Usable Information;** Kawin Ethayarajh, Yejin Choi, Swabha Swayamdipta
- **Prioritized Training on Points that are learnable, Worth Learning, and Not Yet Learnt;** Sören Mindermann et al
- **BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning;** Anonymous
- **SELECTIVE ANNOTATION MAKES LANGUAGE MODELS BETTER FEW-SHOT LEARNERS;** Hongjin Su et al
- **Understanding Transformer Memorization Recall Through Idioms;** Adi Haviv et al
- **Multi-task Active Learning for Pre-trained Transformer-based Models;** Guy Rotman et al
- **DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning;** Chengcheng Guo et al
- **DATASET DISTILLATION;** Tongzhou Wang et al
- **DATASET CONDENSATION WITH GRADIENT MATCHING;** Bo Zhao et al
- **Dataset Distillation by Matching Training Trajectories;** George Cazenavette et al
- **Deep Learning on a Data Diet: Finding Important Examples Early in Training;** Mansheej Paul et al
- **BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning;** Mohsen Fayyaz et al
- **On Measuring the Intrinsic Few-Shot Hardness of Datasets;** Xinran Zhao et al
- **Rissanen data analysis: Examining dataset characteristics via description length;** Ethan Perez et al
- **Sensitivity as a Complexity Measure for Sequence Classification Tasks;** Michael Hahn et al
- **Data-Centric Debugging: mitigating model failures via targeted data collection;** Sahil Singla et al
- **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric;** Richard Zhang et al
- **Synthetic Data Can Also Teach: Synthesizing Effective Data for Unsupervised Visual Representation Learning;** Yawen Wu et al
- **The “Problem” of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation;** Barbara Plank
- **An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models;** Fatemehsadat Mireshghallah et al
- **Training Trajectories of Language Models Across Scales;** Mengzhou Xia et al
- **Data Selection for Language Models via Importance Resampling;** Sang Michael Xie et al
- **Retentive or Forgetful? Diving into the Knowledge Memorizing Mechanism of Language Models; (Memorization)** Boxi Cao et al


## Multi-Task & Transfer & Lifelong & Few-shot & Zero-shot Learning
- **FewNLU: Benchmarking State-of-the-Art Methods for Few-Shot Natural Language Understanding;** Yanan Zheng et al
- **FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS;** Jason Wei et al
- **MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION;** Victor Sanh et al
- **LAMOL: LANGUAGE MODELING FOR LIFELONG LANGUAGE LEARNING;** Fan-Keng Sun, Cheng-Hao Ho, Hung-Yi Lee; Use an LM to both generate previous tasks' data and solve the tasks.
- **CLUES: A Benchmark for Learning Classifiers using Natural Language Explanations;** Rakesh R Menon et al
- **Muppet: Massive Multi-task Representations with Pre-Finetuning;** Armen Aghajanyan et al
- **Cross-Task Generalization via Natural Language Crowdsourcing Instructions;** Swaroop Mishra et al
- **Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks;** Yizhong Wang et al
- **Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners;** Shashank Gupta et al
- **Exploring and Predicting Transferability across NLP Tasks;** Tu Vu et al
- **On Continual Model Refinement in Out-of-Distribution Data Streams;** Bill Yuchen Lin et al
- **Continual-T0: Progressively Instructing 50+ Tasks to Language Models Without Forgetting;** Thomas Scialom et al
- **AdapterFusion: Non-Destructive Task Composition for Transfer Learning;** Jonas Pfeiffer et al 
- **Multi-Task Pre-Training of Modular Prompt for Few-Shot Learning;** Tianxiang Sun et al
- **Eliciting and Understanding Cross-Task Skills with Task-Level Mixture-of-Experts;** Qinyuan Ye et al
- **Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer;** Sen Lin et al
- **12-in-1: Multi-Task Vision and Language Representation Learning;** Jiasen Lu et al


## Backdoor Learning
- **Rethink Stealthy Backdoor Attacks in Natural Language Processing;** Lingfeng Shen et al
- **Excess Capacity and Backdoor Poisoning;** Naren Sarayu Manoj, Avrim Blum
- **Anti-Backdoor Learning: Training Clean Models on Poisoned Data;** Yige Li et al
- **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering;** Bryant Chen et al
- **BACKDOOR DEFENSE VIA DECOUPLING THE TRAINING PROCESS;** Kunzhe Huang, Yiming Li, Baoyuan Wu, Zhan Qin, Kui Ren
- **Backdoor Attacks Against Deep Learning Systems in the Physical World;** Emily Wenger et al
- **DBIA: Data-free Backdoor Injection Attack against Transformer Networks;** Peizhuo Lv et al
- **POISONING AND BACKDOORING CONTRASTIVE LEARNING;** Nicholas Carlini, Andreas Terzis
- **BackdoorBench: A Comprehensive Benchmark of Backdoor Learning;** Baoyuan Wu et al
- **Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks;** Kang Liu et al
- **Adversarial Neuron Pruning Purifies Backdoored Deep Models;** Dongxian Wu, Yisen Wang
- **Backdoor Attacks on Self-Supervised Learning;** Aniruddha Saha et al
- **BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning;** Jinyuan Jia et al
- **BadPrompt: Backdoor Attacks on Continuous Prompts;** Xiangrui Cai et al
- **BACKDOOR OR FEATURE? A NEW PERSPECTIVE ON DATA POISONING;** Anonymous et al
- **Backdoor Vulnerabilities in Normally Trained Deep Learning Models;** Guanhong Tao et al
- **Defending Against Backdoor Attacks in Natural Language Generation;** Xiaofei Sun et al


## Prompt-based Learning
- **LFPT5: A UNIFIED FRAMEWORK FOR LIFELONG FEW-SHOT LANGUAGE LEARNING BASED ON PROMPT TUNING OF T5;** Chengwei Qin, Shafiq Joty; Life-long few-shot learning, using T5. 
- **Pre-trained Token-replaced Detection Model as Few-shot Learner;** Zicheng Li, Shoushan Li, Guodong Zhou
- **FlipDA: Effective and Robust Data Augmentation for Few-Shot Learning;** Jing Zhou et al
- **Example-based Hypernetworks for Out-of-Distribution Generalization;** Tomer Volk et al
- **Few-Shot Learning with Siamese Networks and Label Tuning;** Thomas Muller et al
- **Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks;** Yizhong Wang et al
- **On Transferability of Prompt Tuning for Natural Language Understanding;** Yusheng Su et al
- **reStructured Pre-training;** Weizhe Yuan, Pengfei Liu
- **Learning to Prompt for Vision-Language Models;** Kaiyang Zhou et al
- **Efficient Few-Shot Learning Without Prompts;** Lewis Tunstall et al
- **ASK ME ANYTHING: A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS;** Anonymous
- **Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization;** Yuxian Gu et al
- **Boosting Natural Language Generation from Instructions with Meta-Learning;** Budhaditya Deb et al
- **Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models;** Robert L. Logan IV et al
- **Prompt consistency for zero-shot task generalization;** Chunting Zhou et al
- **A Universal Discriminator for Zero-Shot Generalization;** Haike Xu et al
- **Co-training Improves Prompt-based Learning for Large Language Models;** Hunter Lang et al
- **UNISUMM: Unified Few-shot Summarization with Multi-Task Pre-Training and Prefix-Tuning;** Yulong Chen et al
- **PromptCap: Prompt-Guided Task-Aware Image Captioning;** Yushi Hu et al
- **Demystifying Prompts in Language Models via Perplexity Estimation;** Hila Gonen et al
- **Self-Prompting Large Language Models for Open-Domain QA;** Junlong Li et al
- **Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor;** Or Honovich et al
- **One Embedder, Any Task: Instruction-Finetuned Text Embeddings;** Hongjin Su et al
- **Toward Human Readable Prompt Tuning: Kubrick’s The Shining is a good movie, and a good prompt too?;** Weijia Shi et al
- **Making Pretrained Language Models Good Long-tailed Learners;** Chen Zhang et al
- **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization;** Srinivasan Iyer et al
- **Exploring the Benefits of Training Expert Language Models over Instruction Tuning;** Joel Jang et al
- **UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation;** Daixuan Cheng et al

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
- **Sparse Structure Search for Parameter-Efficient Tuning;** Shengding Hu et al
- **LiST: Lite Prompted Self-training Makes Parameter-efficient Few-shot Learners;** Yaqing Wang et al
- **Meta-Adapter: Parameter Efficient Few-Shot Learning through Meta-Learning;** Anonymous
- **TOWARDS A UNIFIED VIEW ON VISUAL PARAMETER-EFFICIENT TRANSFER LEARNING;** Bruce X.B. Yu et al
- **Performance-Efficiency Trade-Offs in Adapting Language Models to Text Classification Tasks;** Laura Aina et al
- **Attentional Mixtures of Soft Prompt Tuning for Parameter-efficient Multi-task Knowledge Sharing;** Akari Asai et al
- **Efficiently Tuned Parameters are Task Embeddings;** Wangchunshu Zhou et al
- **Different Tunes Played with Equal Skill: Exploring a Unified Optimization Subspace for Delta Tuning;** Jing Yi et al
- **Evaluating Parameter Efficient Learning for Generation;** Peng Xu et al
- **AdaMix: Mixture-of-Adaptations for Parameter-efficient Model Tuning;** Yaqing Wang et al
- **Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters;** Hongyu Zhao et al
- **HyperTuning: Toward Adapting Large Language Models without Back-propagation;** Jason Phang et al
- **On the Effectiveness of Parameter-Efficient Fine-Tuning;** Zihao Fu et al
- **LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning;** Yi-Lin Sung et al
- **Contrastive Adapters for Foundation Model Group Robustness;** Michael Zhang et al
- **Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning;** Dongze Lian et al
- **PARAMETER-EFFICIENT FINE-TUNING DESIGN SPACES;** Jiaao Chen et al
- **An Empirical Study on the Transferability of Transformer Modules in Parameter-Efficient Fine-Tuning;** Mohammad AkbarTajari et al
- **MULTITASK PROMPT TUNING ENABLES PARAMETER-EFFICIENT TRANSFER LEARNING;** Zhen Wang et al



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
- **Learning De-biased Representations with Biased Representations;** Hyojin Bahng et al
- **Unlearnable Text for Neural Classifiers;** Anonymous
- **Generative Bias for Visual Question Answering;** Jae Won Cho et al
- **Measuring Causal Effects of Data Statistics on Language Model’s `Factual’ Predictions;** Yanai Elazar et al
- **WHICH SHORTCUT CUES WILL DNNS CHOOSE? A STUDY FROM THE PARAMETER-SPACE PERSPECTIVE;** Luca Scimeca et al
- **Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing;** Vedika Agarwal et al
- **Shortcut Learning of Large Language Models in Natural Language Understanding: A Survey;** Mengnan Du et al
- **Nuisances via Negativa: Adjusting for Spurious Correlations via Data Augmentation;** Aahlad Puli et al
- **MaskTune: Mitigating Spurious Correlations by Forcing to Explore;** Saeid Asgari Taghanaki et al
- **On Feature Learning in the Presence of Spurious Correlations;** Pavel Izmailov et al
- **Finding Dataset Shortcuts with Grammar Induction;** Dan Friedman et al
- **Are All Spurious Features in Natural Language Alike? An Analysis through a Causal Lens;** Nitish Joshi et al
- **Investigating Ensemble Methods for Model Robustness Improvement of Text Classifiers;** Jieyu Zhao et al
- **XMD: An End-to-End Framework for Interactive Explanation-Based Debugging of NLP Models;** Dong-Ho Lee et al
- **“Will You Find These Shortcuts?” A Protocol for Evaluating the Faithfulness of Input Salience Methods for Text Classification;** Jasmijn Bastings et al
- **Using Focal Loss to Fight Shallow Heuristics: An Empirical Analysis of Modulated Cross-Entropy in Natural Language Inference;** Frano Rajic et al
- **Automatic Shortcut Removal for Self-Supervised Representation Learning;** Matthias Minderer et al
- **Which Shortcut Solution Do Question Answering Models Prefer to Learn?;** Kazutoshi Shinoda et al
- **On Feature Learning in the Presence of Spurious Correlations;** Pavel Izmailov et al
- **Debiasing Masks: A New Framework for Shortcut Mitigation in NLU;** Johannes Mario Meissner et al
- **SHORTCUT LEARNING THROUGH THE LENS OF EARLY TRAINING DYNAMICS;** Nihal Murali et al
- **Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations;** Polina Kirichenko et al
- **Discover and Cure: Concept-aware Mitigation of Spurious Correlation;** Shirley Wu et al


## NLP for Social Good
- **Defending Against Neural Fake News;** Rowan Zellers et al
- **Fact-Enhanced Synthetic News Generation;** Kai Shu et al
- **On Unifying Misinformation Detection;** Nayeon Lee et al 
- **Towards Few-Shot Fact-Checking via Perplexity;** Nayeon Lee et al
- **Language Models as Fact Checkers?;** Nayeon Lee et al
- **A Stylometric Inquiry into Hyperpartisan and Fake News;** Martin Potthast et al
- **Zoom Out and Observe: News Environment Perception for Fake News Detection;** Qiang Sheng et al



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
- **Visual Genome Connecting Language and Vision Using Crowdsourced Dense Image Annotations;** Ranjay Krishna et al
- **Wilds: A Benchmark of in-the-Wild Distribution Shifts;** Pang Wei Koh et al
- **Natural Adversarial Examples;** Dan Hendrycks et al; ImageNet-A & ImageNet-O.
- **The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization;** Dan Hendrycks et al; ImageNet-R.
- **Learning Robust Global Representations by Penalizing Local Predictive Power;** Haohan Wang et al; ImageNet-S.
- **ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models;** Andrei Barbu et al; ObjectNet.
- **EXTENDING THE WILDS BENCHMARK FOR UNSUPERVISED ADAPTATION;** Shiori Sagawa et al
- **MEASURING MASSIVE MULTITASK LANGUAGE UNDERSTANDING;** Dan Hendrycks et al
- **Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence;** Tal Schuster et al
- **CROSSFIT: A Few-shot Learning Challenge for Cross-task Generalization in NLP;** Qinyuan Ye, Bill Yuchen Lin, Xiang Ren
- **RAFT: A Real-World Few-Shot Text Classification Benchmark;** Neel Alex et al
- **LMentry: A Language Model Benchmark of Elementary Language Tasks;** Avia Efrat et al
- **GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective;** Linyi Yang et al
- **ClueWeb22: 10 Billion Web Documents with Rich Information;** ARNOLD OVERWIJK et al
- **What Will it Take to Fix Benchmarking in Natural Language Understanding?;** Samuel R. Bowman et al
- **Measuring Data;** Margaret Mitchell et al



## Foundation Model
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
- **The Lottery Ticket Hypothesis for Pre-trained BERT Networks;** Tianlong Chen et al
- **Learning to Win Lottery Tickets in BERT Transfer via Task-agnostic Mask Training;** Yuanxin Liu et al 
- **DEMIX Layers: Disentangling Domains for Modular Language Modeling;** Suchin Gururangan et al
- **Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?;** Yi Tay et al
- **EXPLORING THE LIMITS OF LARGE SCALE PRE-TRAINING;** Samira Abnar, Mostafa Dehghani, Behnam Neyshabur, Hanie Sedghi
- **Emergent Abilities of Large Language Models;** Jason Wei et al
- **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts;** Nan Du et al
- **RETHINKING SUPERVISED PRE-TRAINING FOR BETTER DOWNSTREAM TRANSFERRING;** Yutong Feng et al
- **HTLM: HYPER-TEXT PRE-TRAINING AND PROMPTING OF LANGUAGE MODELS;** Armen Aghajanyan et al
- **Downstream Datasets Make Surprisingly Good Pretraining Corpora;** Kundan Krishna et al
- **GUESS THE INSTRUCTION! MAKING LANGUAGE MODELS STRONGER ZERO-SHOT LEARNERS;** Seonghyeon Ye et al
- **UNDERSTANDING HTML WITH LARGE LANGUAGE MODELS;** Anonymous
- **Transcending Scaling Laws with 0.1% Extra Compute;** Yi Tay et al
- **Scaling Instruction-Finetuned Language Models;** Hyung Won Chung et al
- **LARGE LANGUAGE MODELS CAN SELF-IMPROVE;** Jiaxin Huang et al
- **Measuring Progress on Scalable Oversight for Large Language Models;** Samuel R. Bowman et al
- **Large Language Models with Controllable Working Memory;** Daliang Li et al
- **Galactica: A Large Language Model for Science;** Ross Taylor et al
- **PAL: Program-aided Language Models;** Luyu Gao et al
- **Can Offline Reinforcement Learning Help Natural Language Understanding?;** Ziqi Zhang et al
- **Training language models to follow instructions with human feedback;** Long Ouyang et al; Directly use RL to train language generation models according to the human feedback. 
- **SAFETEXT: A Benchmark for Exploring Physical Safety in Language Models;** Sharon Levy et al
- **Human or Machine? Turing Tests for Vision and Language;** Mengmi Zhang et al
- **Language Models as Agent Models;** Jacob Andreas
- **Talking About Large Language Models;** Murray Shanahan
- **Discovering Language Model Behaviors with Model-Written Evaluations;** Ethan Perez et al
- **Evaluating Human-Language Model Interaction;** Mina Lee et al
- **Toolformer: Language Models Can Teach Themselves to Use Tools;** Timo Schick et al


## Incontext Learning
- **Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?;** Sewon Min et al
- **Extrapolating to Unnatural Language Processing with GPT-3's In-context Learning: The Good, the Bad, and the Mysterious;** Frieda Rong et al
- **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning;** Haokun Liu et al
- **Learning To Retrieve Prompts for In-Context Learning;** Ohad Rubin et al
- **An Explanation of In-context Learning as Implicit Bayesian Inference;** Sang Michael Xie, Aditi Raghunathan, Percy Liang, Tengyu Ma
- **MetaICL: Learning to Learn In Context;** Sewon Min et al
- **PROMPTING GPT-3 TO BE RELIABLE;** Chenglei Si et al
- **Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm;** Laria Reynolds et al
- **Do Prompt-Based Models Really Understand the Meaning of their Prompts?;** Albert Webson et al
- **On the Relation between Sensitivity and Accuracy in In-context Learning;** Yanda Chen et al
- **Meta-learning via Language Model In-context Tuning;** Yanda Chen et al
- **Extrapolating to Unnatural Language Processing with GPT-3's In-context Learning: The Good, the Bad, and the Mysterious;** Frieda Rong
- **SELECTIVE ANNOTATION MAKES LANGUAGE MODELS BETTER FEW-SHOT LEARNERS;** Hongjin Su et al
- **Robustness of Demonstration-based Learning Under Limited Data Scenario;** Hongxin Zhang et al; Demonstration-based learning, tuning the parameters.
- **Active Example Selection for In-Context Learning;** Yiming Zhang et al
- **Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity;** Yao Lu et al
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models;** Tony Z. Zhao et al
- **DIALOGIC: Controllable Dialogue Simulation with In-Context Learning;** Zekun Li et al 
- **PRESERVING IN-CONTEXT LEARNING ABILITY IN LARGE LANGUAGE MODEL FINE-TUNING;** Yihan Wang et al
- **Teaching Algorithmic Reasoning via In-context Learning;** Hattie Zhou et al
- **On the Compositional Generalization Gap of In-Context Learning** Arian Hosseini et al
- **Transformers generalize differently from information stored in context vs weights;** Stephanie C.Y. Chan et al
- **OVERTHINKING THE TRUTH: UNDERSTANDING HOW LANGUAGE MODELS PROCESS FALSE DEMONSTRATIONS;** Anonymous
- **In-context Learning and Induction Heads;** Catherine Olsson et al
- **Complementary Explanations for Effective In-Context Learning;** Xi Ye et al
- **What is Not in the Context? Evaluation of Few-shot Learners with Informative Demonstrations;** Michal Štefánik et al
- **Robustness of Learning from Task Instructions;** Jiasheng Gu et al
- **Structured Prompting: Scaling In-Context Learning to 1,000 Examples;** Yaru Hao et al
- **Transformers learn in-context by gradient descent;** Johannes von Oswald et al
- **Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale;** Hritik Bansal et al
- **Z-ICL: Zero-Shot In-Context Learning with Pseudo-Demonstrations;** Xinxi Lyu et al
- **Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters;** Boshi Wang et al
- **Careful Data Curation Stabilizes In-context Learning;** Ting-Yun Chang et al
- **Parallel Context Windows Improve In-Context Learning of Large Language Models;** Nir Ratner et al
- **Investigating Fusion Methods for In-Context Learning;** Qinyuan Ye et al
- **Batch Prompting: Efficient Inference with Large Language Model APIs;** Zhoujun Cheng et al
- **Explanation Selection Using Unlabeled Data for In-Context Learning;** Xi Ye et al
- **Compositional Exemplars for In-context Learning;** Jiacheng Ye et al
- **Distinguishability Calibration to In-Context Learning;** Hongjing Li et al
- **How Does In-Context Learning Help Prompt Tuning?;** Simeng Sun et al
- **Guiding Large Language Models via Directional Stimulus Prompting;** Zekun Li et al
- **In-Context Instruction Learning;** Seonghyeon Ye et al
- **LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY;** Jerry Wei et al
- **kNN PROMPTING: BEYOND-CONTEXT LEARNING WITH CALIBRATION-FREE NEAREST NEIGHBOR INFERENCE;** Benfeng Xu et al
- **Learning In-context Learning for Named Entity Recognition;** Jiawei Chen et al


## Model Analysis
- **A Closer Look at How Fine-tuning Changes BERT;** Yichu Zhou, Vivek Srikumar; Analyze fine-tuning. 
- **Word Order Does Matter (And Shuffled Language Models Know It);** Vinit Ravishankar et al
- **BERT Rediscovers the Classical NLP Pipeline;** Ian Tenney, Dipanjan Das, Ellie Pavlick
- **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space;** Thomas Muller et al
- **A Primer in BERTology: What We Know About How BERT Works;** Anna Rogers, Olga Kovaleva, Anna Rumshisky 
- **Is BERT a Cross-Disciplinary Knowledge Learner? A Surprising Finding of Pre-trained Models’ Transferability;** Wei-Tsung Kao, Hung-yi Lee
- **Finding Experts in Transformer Models;** Xavier Suau et al
- **Finding Skill Neurons in Pre-trained Transformers via Prompt Tuning;** Anonymous
- **Analyzing Encoded Concepts in Transformer Language Models;** Hassan Sajjad et al
- **Language Models are General-Purpose Interfaces;** Yaru Hao et al
- **Vision Transformers with Patch Diversification;** Chengyue Gong et al
- **Attention is not all you need: pure attention loses rank doubly exponentially with depth;** Yihe Dong et al
- **BERT Loses Patience: Fast and Robust Inference with Early Exit;** Wangchunshu Zhou et al
- **Shallow-Deep Networks: Understanding and Mitigating Network Overthinking;** Yigitcan Kaya et al
- **ON THE PITFALLS OF ANALYZING INDIVIDUAL NEURONS IN LANGUAGE MODELS;** Omer Antverg, Yonatan Belinkov
- **NATURAL LANGUAGE DESCRIPTIONS OF DEEP VISUAL FEATURES;** Evan Hernandez et al
- **DISCOVERING LATENT CONCEPTS LEARNED IN BERT;** Fahim Dalvi et al
- **NO ONE REPRESENTATION TO RULE THEM ALL: OVERLAPPING FEATURES OF TRAINING METHODS;** Raphael Gontijo-Lopes, Yann Dauphin, Ekin D. Cubuk
- **Editing Factual Knowledge in Language Models;** Nicola De Cao et al
- **How Much Knowledge Can You Pack Into the Parameters of a Language Model?;** Adam Roberts et al
- **Transformer Feed-Forward Layers Are Key-Value Memories** Mor Geva et al
- **oLMpics - On what Language Model Pre-training Captures;** Alon Talmor et al
- **Calibrating Factual Knowledge in Pretrained Language Models;** Qingxiu Dong et al
- **RELATIVE REPRESENTATIONS ENABLE ZERO-SHOT LATENT SPACE COMMUNICATION;** Luca Moschella et al
- **What do Large Language Models Learn beyond Language?;** Avinash Madasu, Shashank Srivastava
- **On the Transformation of Latent Space in Fine-Tuned NLP Models;** Nadir Durrani et al
- **How Much Does Attention Actually Attend? Questioning the Importance of Attention in Pretrained Transformers;** Michael Hassid et al
- **Large Language Models Struggle to Learn Long-Tail Knowledge;** Nikhil Kandpal et al
- **Linear Interpolation In Parameter Space is Good Enough for Fine-Tuned Language Models;** Mark Rofin et al; Mode connectivity. 
- **Exploring Mode Connectivity for Pre-trained Language Models;** Yujia Qin et al; Mode connectivity. 
- **BRIDGING MODE CONNECTIVITY IN LOSS LANDSCAPES AND ADVERSARIAL ROBUSTNESS;** Pu Zhao et al; Mode connectivity. 
- **Neural Dependencies Emerging from Learning Massive Categories;** Ruili Feng et al
- **Event knowledge in large language models: the gap between the impossible and the unlikely;** Carina Kauf et al
- **EDITING MODELS WITH TASK ARITHMETIC;** Gabriel Ilharco et al
- **DISCOVERING LATENT KNOWLEDGE IN LANGUAGE MODELS WITHOUT SUPERVISION;** Collin Burns et al
- **Insights into Pre-training via Simpler Synthetic Tasks;** Yuhuai Wu et al
- **A Structural Probe for Finding Syntax in Word Representations;** John Hewitt et al
- **Why do Nearest Neighbor Language Models Work?;** Frank F. Xu et al
- **Task-Specific Skill Localization in Fine-tuned Language Models;** Abhishek Panigrahi et al
- **Knowledge is a Region in Weight Space for Fine-tuned Language Models;** Almog Gueta et al
- **Deep Learning Through the Lens of Example Difficulty;** Robert J. N. Baldock et al


## Theory
- **Reconciling modern machine learning practice and the bias-variance trade-off;** Mikhail Belkin et al
- **NEURAL COLLAPSE UNDER MSE LOSS: PROXIMITY TO AND DYNAMICS ON THE CENTRAL PATH;** X.Y. Han et al
- **ON THE ROLE OF NEURAL COLLAPSE IN TRANSFER LEARNING;** Tomer Galanti et al
- **PAIRNORM: TACKLING OVERSMOOTHING IN GNNS;** Lingxiao Zhao, Leman Akoglu
- **REVISITING OVER-SMOOTHING IN BERT FROM THE PERSPECTIVE OF GRAPH;** Han Shi et al
- **ANTI-OVERSMOOTHING IN DEEP VISION TRANSFORMERS VIA THE FOURIER DOMAIN ANALYSIS: FROM THEORY TO PRACTICE;** Peihao Wang et al
- **Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?;** Yibo Yang et al
- **Same Pre-training Loss, Better Downstream: Implicit Bias Matters for Language Models;** Hong Liu et al
- **MODELDIFF: A Framework for Comparing Learning Algorithms;** Harshay Shah et al
- **Datamodels: Predicting Predictions from Training Data;** Andrew Ilyas et al
- **Distinguishing rule- and exemplar-based generalization in learning systems;** Ishita Dasgupta et al
- **DISCOVERING AND EXPLAINING THE REPRESENTA- TION BOTTLENECK OF DNNS;** Huiqi Deng et al
- **Concept-Level Explanation for the Generalization of a DNN;** Huilin Zhou et al


## Language and Robotics
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
- **LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action;** Dhruv Shah et al
- **Inner Monologue: Embodied Reasoning through Planning with Language Models;** Wenlong Huang et al






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
- **A Unified Sequence Interface for Vision Tasks;** Ting Chen et al
- **What Makes Training Multi-modal Classification Networks Hard?;** Weiyao Wang, Du Tran, Matt Feiszli

## Scene Graph
- **Visual Distant Supervision for Scene Graph Generation;** Yuan Yao et al
- **Learning to Generate Scene Graph from Natural Language Supervision;** Yiwu Zhong et al
- **Weakly Supervised Visual Semantic Parsing;** Alireza Zareian, Svebor Karaman, Shih-Fu Chang
- **Scene Graph Prediction with Limited Labels;** Vincent S. Chen, Paroma Varma, Ranjay Krishna, Michael Bernstein, Christopher Re, Li Fei-Fei
- **Neural Motifs: Scene Graph Parsing with Global Context;** Rowan Zellers et al
- **Fine-Grained Scene Graph Generation with Data Transfer;** Ao Zhang et al
- **Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning;** Tao He et al
- **Large-Scale Visual Relationship Understanding;** Ji Zhang et al
- **Exploring Long Tail Visual Relationship Recognition with Large Vocabulary;** Sherif Abdelkarim et al


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
- **Language models show human-like content effects on reasoning;** Ishita Dasgupta et al
- **Faithful Reasoning Using Large Language Models;** Antonia Creswell et al
- **FOLIO: Natural Language Reasoning with First-Order Logic;** Simeng Han et al
- **RAINIER: Reinforced Knowledge Introspector for Commonsense Question Answering; (Commonsense)** Jiacheng Liu et al
- **MEASURING AND NARROWING THE COMPOSITIONALITY GAP IN LANGUAGE MODELS;** Ofir Press et al
- **REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS;** Shunyu Yao et al
- **Neural Theory-of-Mind? On the Limits of Social Intelligence in Large LMs;** Maarten Sap et al
- **A Systematic Investigation of Commonsense Knowledge in Large Language Models (Commonsense);** Xiang Lorraine Li et al
- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks;** Wenhu Chen et al
- **A Generative Approach for Script Event Prediction via Contrastive Fine-tuning;** Fangqi Zhu et al
- **ALERT: Adapting Language Models to Reasoning Tasks;** Ping Yu et al

## CV Reasoning
- **MERLOT: Multimodal Neural Script Knowledge Models;** Rowan Zellers et al
- **MERLOT Reserve: Neural Script Knowledge through Vision and Language and Sound;** Rowan Zellers et al
- **THE NEURO-SYMBOLIC CONCEPT LEARNER: INTERPRETING SCENES, WORDS, AND SENTENCES FROM NATURAL SUPERVISION; (Vision Reasoning)** Jiayuan Mao et al
- **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning; (Image Reasoning Benchmark)** Justin Johnson et al
- **A Corpus for Reasoning About Natural Language Grounded in Photographs; (Image & Language Reasoning)** Alane Suhr et al
- **From Recognition to Cognition: Visual Commonsense Reasoning; (Image & Language Reasoning)** Rowan Zellers et al
- **The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning;** Jack Hessel et al
- **Learning by Abstraction: The Neural State Machine;** Drew A. Hudson, Christopher D. Manning
- **VQA-LOL: Visual Question Answering under the Lens of Logic;** Tejas Gokhale et al
- **Cross-Modality Relevance for Reasoning on Language and Vision;** Chen Zheng, Quan Guo, Parisa Kordjamshidi
- **SQuINTing at VQA Models: Introspecting VQA Models with Sub-Questions;** Ramprasaath R. Selvaraju et al
- **EVENTS REALM: Event Reasoning of Entity States via Language Models;** Evangelia Spiliopoulou et al; Relevant to PiGLET & Open PI. 
- **Visually Grounded Commonsense Knowledge Acquisition;** Yuan Yao et al; Automatically build knowledge base.



## MRC Reasoning
- **NEURAL MODULE NETWORKS FOR REASONING OVER TEXT;** Nitish Gupta et al
- **NEURAL SYMBOLIC READER: SCALABLE INTEGRATION OF DISTRIBUTED AND SYMBOLIC REPRESENTATIONS FOR READING COMPREHENSION;** Xinyun Chen et al
- **Is a Question Decomposition Unit All We Need?;** Pruthvi Patel et al; Decompose a hard question into several easy ones via human-in-the-loop.
- **Successive Prompting for Decomposing Complex Questions;** Dheeru Dua et al



## Grounding
- **Climbing towards NLU: On meaning, form, and understanding in the age of data;** Emily M. Bender, Alexander Koller
- **Provable Limitations of Acquiring Meaning from Ungrounded Form: What Will Future Language Models Understand?;** William Merrill et al
- **What Does BERT with Vision Look At?;** Liunian Harold Li et al
- **Visual Grounding Strategies for Text-Only Natural Language Processing;** Damien Sileo; Discuss how multi-modal pretraining improves NLU tasks. 
- **Experience Grounds Language;** Yonatan Bisk et al
- **ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension;** Sanjay Subramanian et al
- **Do Trajectories Encode Verb Meaning?;** Dylan Ebert et al
- **Retrospectives on the Embodied AI Workshop;** Matt Deitke et al



## NLG Hallucination
- **On Faithfulness and Factuality in Abstractive Summarization;** Joshua Maynez et al
- **Entity-Based Knowledge Conflicts in Question Answering;** Shayne Longpre et al; QA Task hallucination. 
- **Evaluating the Factual Consistency of Abstractive Text Summarization;** Wojciech Krys ́cin ́ski, Bryan McCann, Caiming Xiong, Richard Socher
- **Annotating and Modeling Fine-grained Factuality in Summarization;** Tanya Goyal, Greg Durrett
- **FACTPEGASUS: Factuality-Aware Pre-training and Fine-tuning for Abstractive Summarizationl;** David Wan, Mohit Bansal
- **Evidentiality-guided Generation for Knowledge-Intensive NLP Tasks;** Akari Asai et al
- **Towards Improving Faithfulness in Abstractive Summarization;** Xiuying Chen et al
- **Just ClozE! A Fast and Simple Method for Evaluating the Factual Consistency in Abstractive Summarization;** Yiyang Li et al
- **Mutual Information Alleviates Hallucinations in Abstractive Summarization;** Liam van der Poel et al
- **Correcting Diverse Factual Errors in Abstractive Summarization via Post-Editing and Language Model Infilling;** Vidhisha Balachandran et al
- **FRSUM: Towards Faithful Abstractive Summarization via Enhancing Factual Robustness;** Wenhao Wu et al
- **Evaluating the Factual Consistency of Large Language Models Through Summarization;** Derek Tam et al
- **RARR: Researching and Revising What Language Models Say, Using Language Models;** Luyu Gao et al
- **SUMMAC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization;** Philippe Laban et al
- **Tracing and Removing Data Errors in Natural Language Generation Datasets;** Faisal Ladhak et al
- **Understanding and Detecting Hallucinations in Neural Machine Translation via Model Introspection;** Weijia Xu et al
- **Measuring Attribution in Natural Language Generation Models;** Hannah Rashkin et al
- **Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models;** Bernd Bohnet et al
- **Learning with Rejection for Abstractive Text Summarization;** Meng Cao et al
- **Factuality Enhanced Language Models for Open-Ended Text Generation;** Nayeon Lee et al





## Text Editing
- **Text Editing by Command;** Felix Faltings et al
- **CoAuthor: Designing a Human-AI Collaborative Writing Dataset for Exploring Language Model Capabilities;** Mina Lee et al
- **PEER: A Collaborative Language Model;** Timo Schick et al
- **GENERATING SEQUENCES BY LEARNING TO [SELF-]CORRECT;** Sean Welleck et al
- **Interactive Text Generation;** Felix Faltings et al



## Information Extraction
- **Connecting the Dots: Event Graph Schema Induction with Path Language Modeling;** Manling Li et al
- **Forecasting Future World Events with Neural Networks;** Andy Zou et al
- **The Future is not One-dimensional: Complex Event Schema Induction by Graph Modeling for Event Prediction;** Manling Li et al
- **Event Schema Induction with Double Graph Autoencoders;** Xiaomeng Jin, Manling Li, Heng Ji
- **TEXT2EVENT: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction;** Yaojie Lu et al
- **CLEVE: Contrastive Pre-training for Event Extraction;** Ziqi Wang et al
- **PILED: An Identify-and-Localize Framework for Few-Shot Event;** Sha Li et al
- **Liberal Event Extraction and Event Schema Induction;** Lifu Huang et al
- **Learning from Context or Names? An Empirical Study on Neural Relation Extraction;** Hao Peng et al
- **Context-aware Adversarial Training for Name Regularity Bias in Named Entity Recognition;** Abbas Ghaddar et al 
- **MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction;** Xiaozhi Wang et al
- **LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model;** Hao Fei et al
- **Unified Structure Generation for Universal Information Extraction;** Yaojie Lu et al

## Retrieval-augmented LLM
- **Unsupervised Cross-Task Generalization via Retrieval Augmentation;** Bill Yuchen Lin et al
- **REPLUG: Retrieval-Augmented Black-Box Language Models;** Weijia Shi et al
- **DEMONSTRATE–SEARCH–PREDICT: Composing retrieval and language models for knowledge-intensive NLP;** Omar Khattab et al
- **Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning;** Xiang Chen et al
- **Task-aware Retrieval with Instructions;** Akari Asai et al
- **GENERATE RATHER THAN RETRIEVE: LARGE LANGU- AGE MODELS ARE STRONG CONTEXT GENERATORS;** Wenhao Yu et al
- **In-Context Retrieval-Augmented Language Models;** Ori Ram et al
- **Semiparametric Language Models Are Scalable Continual Learners;** Guangyue Peng et al


## Code
- **BINDING LANGUAGE MODELS IN SYMBOLIC LANGUAGES;** Zhoujun Cheng et al
- **HTLM: Hyper-Text Pre-Training and Prompting of Language Models;** Armen Aghajanyan et al
- **PAL: Program-aided Language Models;** Luyu Gao et al
- **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks;** Wenhu Chen et al
- **CODE4STRUCT: Code Generation for Few-Shot Structured Prediction from Natural Language;** Xingyao Wang et al
- **Language Models of Code are Few-Shot Commonsense Learners;** Aman Madaan et al
- **CodeExp: Explanatory Code Document Generation;** Haotian Cui et al
- **Visual Programming: Compositional visual reasoning without training;** Tanmay Gupta et al
- **CM3: A Causal Masked Multimodal Model of the Internet;** Armen Aghajanyan et al
- **ProgPrompt: Generating Situated Robot Task Plans using Large Language Models;** Ishika Singh et al
- **InCoder: A Generative Model for Code Infilling and Synthesis;** Daniel Fried et al
- **Evaluating Large Language Models Trained on Code;** Mark Chen et al
- **Code as Policies: Language Model Programs for Embodied Control;** Jacky Liang et al
- **PIX2STRUCT: SCREENSHOT PARSING AS PRETRAINING FOR VISUAL LANGUAGE UNDERSTANDING;** Kenton Lee et al
- **Reasoning Like Program Executors;** Xinyu Pi et al
- **ReCode: Robustness Evaluation of Code Generation Models;** Shiqi Wang et al
- **CORRPUS: Detecting Story Inconsistencies via Codex-Bootstrapped Neurosymbolic Reasoning;** Yijiang River Dong et al
- **LEVER: Learning to Verify Language-to-Code Generation with Execution;** Ansong Ni et al
- **PLANNING WITH LARGE LANGUAGE MODELS FOR CODE GENERATION;** Shun Zhang et al


## Security of LLM
- **How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection;** Biyang Guo et al
- **A Watermark for Large Language Models;** John Kirchenbauer et al
- **DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature;** Eric Mitchell et al
- **Exploring AI Ethics of ChatGPT: A Diagnostic Analysis;** Terry Yue Zhuo et al
- **CHATGPT OR HUMAN? DETECT AND EXPLAIN. EXPLAINING DECISIONS OF MACHINE LEARNING MODEL FOR DETECTING SHORT CHATGPT-GENERATED TEXT;** Sandra Mitrovic et al
- **A Categorical Archive of ChatGPT Failures;** Ali Borji et al
- **Adversarial Prompting for Black Box Foundation Models;** Natalie Maus et al
- **The Capacity for Moral Self-Correction in Large Language Models;** Deep Ganguli et al
- **IS CHATGPT A GENERAL-PURPOSE NATURAL LANGUAGE PROCESSING TASK SOLVER?;** Chengwei Qin et al
- **A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity;** Yejin Bang et al
- **Pretraining Language Models with Human Preferences;** Tomasz Korbak et al
- **ChatGPT: Jack of all trades, master of none;** Jan Kocoń et al
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective;** Jindong Wang et al
- **How Robust is GPT-3.5 to Predecessors? A Comprehensive Study on Language Understanding Tasks;** Xuanting Chen et al
- **The Science of Detecting LLM-Generated Texts;** Ruixiang Tang et al
- **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models;** Potsawee Manakul et al
- **A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models;** Junjie Ye et al

## Interesting Topics
- **Advancing mathematics by guiding human intuition with AI;** Alex Davies et al
- **What are the best Systems? New Perspectives on NLP Benchmarking;** Pierre Colombo, Nathan Noiry, Ekhine Irurozki, Stephan Clemencon; Study how to aggregate metrics in multi-task evaluation. "benchmarks are made of datasets, metrics, and a way to aggregate performance. ... If the bulk of the NLP community efforts on this domain is about collecting new datasets and introducing new metrics, little work is concerned with the third part, namely how to aggregate various performances."
- **A Neural-Symbolic Approach to Natural Language Understanding;** Zhixuan Liu et al
- **Fast Few-shot Debugging for NLU Test Suites;** Christopher Malon et al
- **Shedding New Light on the Language of the Dark Web;** Youngjin Jin et al
- **Machine-in-the-Loop Rewriting for Creative Image Captioning;** Vishakh Padmakumar, He He
- **Describing Differences between Text Distributions with Natural Language;** Ruiqi Zhong et al
- **The Dangers of Underclaiming: Reasons for Caution When Reporting How NLP Systems Fail;** Samuel R. Bowman
- **Solving Quantitative Reasoning Problems with Language Models;** Aitor Lewkowycz et al
- **UniCausal: Unified Benchmark and Model for Causal Text Mining;** Fiona Anting Tan et al
- **Large Language models and the reverse turing test;** Terrence Sejnowski
- **Using Large Language Models to Simulate Multiple Humans;** Gati Aher et al
- **WHAT DO NLP RESEARCHERS BELIEVE? RESULTS OF THE NLP COMMUNITY METASURVEY;** Julian Michael et al
- **EXPLAINING PATTERNS IN DATA WITH LANGUAGE MODELS VIA INTERPRETABLE AUTOPROMPTING;** Chandan Singh et al
- **A fine-grained comparison of pragmatic language understanding in humans and language models;** Jennifer Hu et al


<!-- ## Theory of Mind -->


## Learning
- **Distinguishing rule- and exemplar-based generalization in learning systems;** Ishita Dasgupta et al
- **Measures of Information Reflect Memorization Patterns;** Rachit Bansal et al
- **Datamodels: Predicting Predictions from Training Data;** Andrew Ilyas et al
- **MODELDIFF: A Framework for Comparing Learning Algorithms;** Harshay Shah et al

## Interesting Fields (CV)
- **Taming Transformers for High-Resolution Image Synthesis;** Patrick Esser et al
- **Zero-Shot Text-to-Image Generation;** Aditya Ramesh et al; DALL-E. 
- **Masked Autoencoders Are Scalable Vision Learners;** Kaiming He et al
- **Procedural Image Programs for Representation Learning;** Manel Baradad et al


## Resources
- **Transfer Learning in NLP; (Tutorial)** https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit#slide=id.g5882add69e_6_646; Gives a hands-on code to conduct pre-training.
- **Robustness and Adversarial Examples in NLP; (Tutorial)** https://docs.google.com/presentation/d/1E_0qEwQkS43FJGzOEUrpee9zqi8y5lx6D-ABQl3KFas/edit#slide=id.p
- **How to write research paper?** https://students.uu.nl/sites/default/files/ge0-aw-guide-for-scientific-writing-2016.pdf
- **CausalNLP: A Practical Toolkit for Causal Inference with Text;** Arun S. Maiya
- **Challenges and Opportunities in NLP Benchmarking;** Sebastian Ruder
- **Zero- and Few-Shot NLP with Pretrained Language Models; (Tutorial)** https://github.com/allenai/acl2022-zerofewshot-tutorial
- **Dataset Shift in Machine Learning;** JOAQUIN QUIÑONERO-CANDELA, MASASHI SUGIYAMA, ANTON SCHWAIGHOFER, AND NEIL D. LAWRENCE; http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf
- **Contrastive Data and Learning for Natural Language Processing; (Tutorial)** 
- **GPT-3 Nonfiction;** https://www.gwern.net/GPT-3-nonfiction
- **Research Taste;** https://colah.github.io/notes/taste/
- **OFASYS: A Multi-Modal Multi-Task Learning System for Building Generalist Models;** Jinze Bai et al
- **Uncertainty estimation in NLP (Tutorial);** https://sites.google.com/view/uncertainty-nlp
- **A Closer Look at Large Language Models Emergent Abilities;** Yao Fu et al; Blog post.
- **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources;** Yao Fu et al; Blog post.
- **Notes on Teaching GPT-3 Adding Numbers;** Ekin Akyürek et al; Blog post. 
