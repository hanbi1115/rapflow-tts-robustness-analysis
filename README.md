# RapFlow-TTS Few-Step Robustness Analysis

## 1. Project Overview

This project studies how robust **few-step inference** is in RapFlow-TTS across different types of text inputs.

RapFlow-TTS is designed to generate high-quality speech with a small number of sampling steps. Instead of simply reproducing the original paper's average performance results, this project asks a more specific question:

> **Does few-step TTS remain stable across different input conditions, including difficult or unusual sentences?**

The experiment compares sampling steps (`NFE = 1, 2, 4, 8, 16`) across both ordinary and stress-test inputs.

---

## 2. Research Questions

1. How does increasing NFE affect inference speed?
2. Does intelligibility improve consistently as NFE increases?
3. Are some input types more sensitive to low-step generation than others?
4. Do ASR-based metrics such as WER agree with human listening judgments?

---

## 3. Experimental Setup

### Model
- RapFlow-TTS
- Pretrained LJSpeech checkpoint
- HiFi-GAN vocoder

### Sampling conditions
- NFE: `1, 2, 4, 8, 16`

### Baseline experiment
Three ordinary English sentences were synthesized at each NFE.

### Stress-test experiment
40 sentences were divided into 8 categories:

- short
- long
- numbers
- abbreviations
- rare words
- repetition
- prosody
- tongue twisters

Each sentence was synthesized at 5 NFE settings, producing **200 stress-test audio samples**.

---

## 4. Metrics

### Speed
- Model inference time
- Total synthesis time
- Real-Time Factor (RTF)

\[
RTF = \frac{\text{synthesis time}}{\text{audio duration}}
\]

Lower RTF means faster synthesis.

### Automatic intelligibility
- Whisper-based Word Error Rate (WER)

Lower WER means the generated speech was transcribed more accurately.

### Acoustic descriptors
- F0
- RMS energy
- voiced ratio
- spectral centroid

### Human evaluation
Blind listening tests measured:
- word content
- pronunciation
- naturalness
- obvious audible errors

---

## 5. Main Results

### 5.1 Speed increases with NFE

Across the 40-sentence stress test:

| NFE | Mean WER | Mean RTF |
|---:|---:|---:|
| 1 | 0.0851 | 0.0882 |
| 2 | 0.0809 | 0.0941 |
| 4 | 0.0890 | 0.1058 |
| 8 | 0.1078 | 0.1200 |
| 16 | 0.0840 | 0.1575 |

NFE 1 was the fastest setting. Increasing NFE consistently increased computational cost.

However, WER did **not** improve monotonically as NFE increased.

---

## 6. Robustness Across Input Categories

The most difficult categories by average Whisper WER included:

- tongue twisters
- rare words
- abbreviations
- numbers

Meanwhile, the long-sentence category achieved an average WER of 0 in this experiment.

Importantly, categories such as abbreviations showed nearly identical WER across NFE settings. This suggests that some errors may be related to text pronunciation or ASR behavior rather than insufficient sampling steps.

---

## 7. NFE-Sensitive Cases

Several sentences showed relatively large WER changes across NFE settings:

- `Go go go go go.`
- `She sells seashells by the seashore.`
- `Red lorry, yellow lorry, red lorry, yellow lorry.`
- `The mathematician discussed eigenvectors and diffeomorphisms.`
- `The test was hard, hard, hard, but fair.`

For example:

`She sells seashells by the seashore.`

showed a higher WER at NFE 1, while higher-NFE versions were transcribed correctly.

However, automatic WER alone was not enough to determine whether these differences represented genuine perceptual errors.

---

## 8. Blind Listening Validation

The five most NFE-sensitive cases were evaluated in a blind listening test.

Average ratings by NFE:

| NFE | Word Content | Pronunciation | Naturalness |
|---:|---:|---:|---:|
| 1 | 5.0 | 2.8 | 2.8 |
| 2 | 5.0 | 2.6 | 2.6 |
| 4 | 5.0 | 2.6 | 2.6 |
| 8 | 5.0 | 2.6 | 2.6 |
| 16 | 5.0 | 2.6 | 2.6 |

The intended word content was preserved across all NFE settings in the evaluated cases.

The large WER differences observed in some samples were therefore not clearly reproduced as audible content errors in the blind listening test.

This suggests that **Whisper-based WER can be sensitive to ASR recognition behavior and may not perfectly reflect perceptual TTS robustness**.

---

## 9. Interpretation

The experiments suggest three main findings.

### Finding 1
Few-step inference remained surprisingly stable across a wide range of inputs.

### Finding 2
Increasing NFE increased inference cost, but did not consistently improve intelligibility.

### Finding 3
Automatic WER and human perception did not always agree.

Some inputs produced large differences in Whisper WER despite sounding correct to a human listener.

Therefore, robustness evaluation for few-step TTS should not rely on ASR-based WER alone.

---

## 10. Limitations

This project has several limitations:

- Only one pretrained RapFlow-TTS checkpoint was tested.
- The model was trained on LJSpeech and the experiment focused on English.
- The stress-test set contained only 40 sentences.
- Human evaluation was conducted by a single listener.
- WER was computed using one Whisper model.
- No formal perceptual metric such as MOS was collected from multiple participants.
- Acoustic descriptors such as spectral centroid are not direct measures of perceptual speech quality.

---

## 11. Future Work

Possible extensions include:

- increasing the number of stress-test sentences
- using multiple human listeners
- comparing different ASR models
- evaluating multiple TTS models
- testing multilingual speech
- examining speaker-dependent effects
- analyzing prosody and phoneme-level failures in more detail

---

## 12. Project Takeaway

This project began as a reproduction of RapFlow-TTS few-step inference and was extended into a robustness analysis.

The main takeaway is:

> **RapFlow-TTS preserved intelligible speech even at very low sampling steps for most tested inputs, while higher NFE increased computational cost without producing consistent improvements in intelligibility. In addition, ASR-based WER did not always match human perception, highlighting the importance of combining automatic and perceptual evaluation.**

