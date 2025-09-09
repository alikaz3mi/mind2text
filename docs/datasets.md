# Dataset Documentation: ds004148

## Overview

The Mind2Text project uses the **ds004148** dataset from OpenNeuro - "A test-retest resting and cognitive state EEG dataset" by Wang et al. (2022). This dataset provides a rich collection of EEG recordings during various cognitive tasks, making it ideal for cognitive state classification research.

## Dataset Characteristics

### Basic Information
- **Name**: ds004148 - A test-retest resting and cognitive state EEG dataset
- **DOI**: 10.18112/openneuro.ds004148.v1.0.0
- **License**: CC0 (Public Domain)
- **Total Size**: ~50 GB
- **Format**: BIDS-compliant EEG data
| **Multi-day MI (WRBC 2019)** | 62 subjects × 3 sessions | 2-class and 3-class MI | ~30 min per session | Cross-session variability; preprocessed data and code provided [3] |
| **Large MI Cross-Session (Nature 2022)** | 87 subjects × 1 day | Left/right hand MI | ~25 min per subject | User profile metadata, test–retest reliability [4][5] |

## 2. Attention and Cognitive State

| Dataset | Subjects × Sessions | Labels | Duration | Key Features |
|---------|---------------------|--------|----------|--------------|
| **MEMA (Multi-label EEG Attention)** | 20 subjects × 12 trials | 3 multi-labels (neutral, relaxing, concentrating) + emotion/personality | 1,060 min total | Multi‐label attention states; auxiliary behavioral/personality data; GitHub repo [6] |
| **Resting & Cognitive States** | 60 subjects × 3 sessions | 5 states (eyes-open, eyes-closed, memory, music, subtraction) | ~90 min per subject | Test–retest; rich behavioral and mental health assessments [7] |
| **Cognitive Load (Arithmetic Tasks)** | – | Multiple stress/load levels | – | EEG patterns under varying mental load; detailed preprocessing and stimulus info [8] |

## 3. Clinical and Epilepsy

| Dataset | Subjects | Labels | Duration | Key Features |
|---------|----------|--------|----------|--------------|
| **TUH EEG Corpus** | 10,874 subjects | Normal vs seizure; physician reports | > 29 years cumulative | Largest heterogeneous clinical EEG; multi‐sampling rates; raw EDF + reports [9] |
| **CHB-MIT Scalp EEG** | 23 pediatric subjects | Seizure vs non-seizure | 198 h total | Long-term seizure monitoring; 23 channels; EDF format [10] |
| **Interictal Epileptiform Discharge (Nature 2025)** | 84 patients | 5 IED spatial types + wake/sleep | 28 h total | Expert‐annotated IED events, consciousness state, spatial labels [11] |
| **Auto-labeled TUH Extensions** | 15,300 recordings | Multi-label pathology via NLP | – | Automatic labels from clinical reports; extended TUH Abnormal & Epilepsy corpora [12] |

## 4. Comprehensive Repositories and Lists

- **PhysioNet EEG Topic Page**: central access to all PhysioNet EEG datasets (motor, clinical, sleep).[13]
- **EEG-Datasets GitHub**: community-maintained list of public EEG resources across paradigms.[14]
- **Meta-analysis of MI/ME Datasets**: detailed specifications and quality evaluation of 25 motor imagery/execution datasets.[15]

***

**Recommendation:**  
For **motor imagery**, start with **BCI Competition IV 2a** or the **PhysioNet MI dataset** for broad adoption and extensive labels.  
For **attention/cognitive states**, the **MEMA** and **Resting & Cognitive States** datasets provide multi-label annotations and behavioral metadata.  
For **clinical applications**, leverage the **TUH EEG Corpus** for seizure detection and the new **IED spatial distribution dataset** for expert-annotated event classification.

[1](https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a)
[2](https://physionet.org/content/eegmmidb/1.0.0/)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11930978/)
[4](https://www.nature.com/articles/s41597-023-02445-z)
[5](https://www.nature.com/articles/s41597-022-01647-1)
[6](https://arxiv.org/html/2411.09879v1)
[7](https://www.nature.com/articles/s41597-022-01607-9)
[8](https://www.sciencedirect.com/science/article/pii/S2352340925002094)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC4865520/)
[10](https://physionet.org/content/chbmit/1.0.0/)
[11](https://www.nature.com/articles/s41597-025-04572-1)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC10432245/)
[13](https://physionet.org/content/?topic=eeg)
[14](https://github.com/meagmohit/EEG-Datasets)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC10101208/)
[16](https://arxiv.org/html/2506.11830v1)
[17](https://www.physionet.org/physiobank/)
[18](https://www.sciexplor.com/articles/jbde.2025.0005)
[19](https://openneuro.org/datasets/ds004504/versions/1.0.8)
[20](https://www.sciencedirect.com/science/article/pii/S2352340924001525)