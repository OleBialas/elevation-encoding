---
title:
- Decoding of Evoked Responses Reveals Linear Encoding of Sound Source Elevation in Human Cortex
author:
- Ole Bialas
- Burkhard Maess
- Marc Schoenwiesner
bibliography: 
- bibliography.bib
---

# Abstract (250 words maximum, including citations)
Within the scope of this thesis, we conducted two
experiments to investigate the cortical encoding of sound source elevation
using electroencephalography (EEG) while subjects were engaged in a local-
ization task. The results indicate that the human cortex represents sound
source elevation in a linear population rate code and that this neural repre-
sentation is predictive of localization accuracy. This is, to our knowledge, the

- concise summary of objectives, methodology including species and whether both sexes were included (really?)
<!-- Do we really need to make it explicit that both sexes where studies? Seems kinda irrelevant -->
<!-- Marc: It's the Zeitgeist; just state why it is not relevant in this case or do a statistical test for any sex differences; see commentary in https://www.nature.com/articles/d41586-022-02919-x -->
- written in complete sentences without subheadings

# Significance Statement (120 words maximum)
- convey importance to experts and non-experts

# Introduction (650 words maximum, including citations)

## Motivation
Unlike in the visual system, where retinal ganglion cells provide a topographic representation of sensory input, hair cells in the cochlea provide a tonotopic representation of sound frequency. Since sound location is not represented on the sensory epithelium, it must be computed by the central auditory system. This rapid pre-attentive computation makes sound localization an interesting model of information processing and integration in the brain.

## Psychophysics of sound localization
When sound impinges on the ear from one side, the head shadows the averted ear from the sound. The acoustic shadow dampens the sound, resulting in an interaural level difference (ILD). Additionally, the variation in traveling distance creates an interaural time difference (ITD) between the sounds
arriving at both ears. These interaural cues uniquely determine a sound's the azimuth within the frontal field. A sound's elevation on the other hand, is inferred from the spectral pattern that results from direction-dependent filtering trough the head and pinnae. Consequently, the auditory system must learn a representation of one's head-related transfer transfer function (HRTF) to map the filtered spectrum to an elevation. 



### Elevation is where our understanding is lacking the most (Marc: too strong (f.i. distance is much less understood...))
The acoustic cues that convey the location of a sound source in the horizontal (azimuth) and
vertical planes (elevation) are well understood. The neural mechanisms that help to extract
interaural cues for horizontal sound localization, implemented in the brainstem, and the
representation of these cues at different stages of the ascending auditory system are also
understood in some detail. The encoding of sound source elevation and its integration with
source azimuth is less well understood.

### Encoding of elevation has previously not been shown in EEG
While several studies investigated the ﬁring patterns of single neurons in response to
changes in sound source elevation, they do not provide an explanation for the code that
represents elevation on a population level. A recent study used functional magnetic resonance
imaging (fMRI) to observe the cortical encoding of sound source elevation. However, fMRI does
not measure neural activity but associated changes in blood oxygenation, and it requires
virtual auditory displays.


- objectives, background and motivation

- what hypotheses were tested

# Materials and Methods
- sex of subjects should be stated
- all companies from which materials were obtained should be listed

# Experimental Design and Statistical Analyses
- subsection of material an methods
- details of experimental design for each experiment
- within- and between-subject factors and critical variables (e.g. number of trials)
- justification for sample style, power analysis
<!-- We didn't do a power analysis during planning, should be do one now? -->
- statistical tests, how do we control for multiple comparisons?
- encouraged to report all data in histograms, scatter plots etc.
- where can the data be accessed?

# Data and Code Availability
The raw data as well as the analysis code can be obtained from an online repository (github.com/OleBialas/elevation-encoding.git).

# Results

## Behavior
We quantified each subject’s localization accuracy as elevation gain (EG), which is the slope of the regression between the actual and perceived sound source elevation [@hofman1998]. An EG of 1 indicates perfect localization (discounting symmetric deviations around the mean), while an EG of 0
indicates that the subject responded randomly. The initial localization tests showed that subjects could determine the sounds' elevation accurately, indicated by an EG of $0.68 (SD=0.25)$ in the test preceding experiment I and an EG of $0.78 (SD=0.13)$ in the test preceding experiment II (note that subject responses where obtained with different methods). While the EGs we observed are lower compared to previous reports on free field sound localization [@makous1990, @hofman1998], this is likely due to the much shorter training that subjects received in our study.

The average EG for the behavioral task in experiment II was $0.45 (SD=0.26$, significantly lower than the EG measured during the prior localization test (paired-samples t-test: $t(30)=6.05, p<0.001$). We expected a decrease in performance because the continuous display of sounds and uncertainty
about which stimuli were targets made the task much more difficult. Despite the decrease in accuracy, most subject's response was clearly modulated by the target's position, indicating that they were able to perceive elevation.

![Linear regression between the target's elevation and the subject's response. (A) and (B): data from the localization tests preceding each experiment. (C) data from the behavioral task during the second experiment. Each gray line represents one subject's responses and the red line indicates the
group average.](figures/eg.png)

## Evoked Responses


## Decoding


# Discussion (1,500 words maximum, including citations)
- brief statement of the principal findings
- validity and significance in the light of other published work
- extensive discussion of the litertaure is discouraged

