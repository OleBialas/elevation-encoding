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

## Subjects
Twenty-three subjects (15 female) participated in the first and thirty subjects (15 female) in the second experiment. They were between 20 and 33 years old and had no history of neurological or hearing disorders. All subjects gave informed consent and were monetarily compensated at an hourly rate. All procedures were approved by the ethics commit-
tee of the University of Leipzig’s medical faculty.

## Apparatus
Subjects sat in the center of a custom-built spherical array of loudspeakers (model Mod1, Sherman Oaks, CA, USA) inside a 40m² hemi-anechoic chamber (Industrial Acoustics Company, Niederkrüchten, Germany). In experiment II, two additional loudspeakers were mounted next to the subjects ears and
served as headphones. Because of their proximity to the ears, sounds from those loudspeakers were unaffected by the listeners directional transfer function [@moller1995]. This allowed us to simultaneously present sounds from localized loudspeakers as well as non-spatial headphone sounds. We used two
digital signal processors with a 50 kHz sampling rate and six 8-channel amplifiers (models RX8.1 and SA8, Tucker-Davis Technologies, Alachua, FL, USA) to drive the loudspeakers. The processors' digital ports were used to control the LEDs attached to each loudspeaker, obtain responses from the
custom-built button box used in experiment I and send event-triggers to the EEG-system. We used a 64-channel system (model BrainAmpMR, Brain Products, Gilching Germany) to record EEG at a sampling rate of 500 Hz. The silver chloride electrodes were ﬁxated on the subject’s head with an elastic cap
(Easycap, Germany) according to the international 10/20 system. The cap’s size was chosen based on the subject’s head circumference. After positioning, the electrodes were prepared with an electrolyte gel so that their impedance was below 2 kΩ. The data was recorded with FCz as reference using the
manufacturers proprietarty software. In experiment II, we obtained the subjects' responses by estiamting their ehad pose from images acquired by two cameras (model Firefly S, Teledyne FLIR, OR, USA) positioned between the target loudspeakers. 

## Software


## Preprocessing
We applied a de-noising algorithm which combines filtering and source separation to remove power line artifacts while minimizing temporal distortions due to filtering [@de2020]. Next, to remove slow channel drifts and offsets, we applied a causal, minimum- phase, highpass ﬁlter with a hamming window and a 1 Hz cutoff frequency. Subsequently, we re-referenced the data to a robust average after interpolating all channels that could not be predicted accurately from their neighbors [@bigdely2015]. Afterwards, we removed eye-blink artifacts using independent component analysis [@plochl2012]. Components reflecting eye-artifacts were selected automatically by correlating their topography to a previously selected reference [@viola2009]. Finally, we repaired or rejected bad epochs using an algorithm that finds the optimal parameters using cross-validation and baysian optimization [@jas2017]. Except for the initial selection of a reference for blink removal, the entire preprocessing pipeline was automated.

## Experiment I
Subjects sat comfortably on a hight-adjustable chair in the anechoic chamber. The target loudspeakers where located at elevations of 37.5°, 12.5°, −12.5° and −37.5°, where 0° represents eye-level in 3.20 meter distance to the subjects. Because the perception of sound source elevation is slightly more accurate for lateral targets (Makous and Middlebrooks, 1990), all target speakers were positioned at an azimuth of 10° to the subject’s right. All subjects completed an initial localization test which probed their ability to localize sounds and familiarized them with the setup. To avoid familiarization with the transfer functions of the targets, we used a different set of loudspeakers located at elevations of +50°, +25°, 0° and -25°. Subjects heard 150 ms bursts of noise with a 5 ms on- and offset ramp from single loudspeakers in randomized order without direct repetition of the same speaker. They were instructed to keep their head and gaze aligned with the fixation cross at 0°. Subjects localized each sound by pressing one of four buttons - each corresponding to one speaker location. There was no time limit for responding, and the subsequent trial started automatically after the subject had responded to the previous stimulus. After completing 200 training trials, the subject took a break, during which we prepared the EEG electrodes. During recordings, each 150 ms stimulus (probe) was preceded by 600 ms of noise played the loudspeakers at 37.5° or −37.5°. The probe and adapter were never played from the same speaker, and they were cross-faded so that the sound intensity remained constant during the transition. The adapter’s initial position was chosen randomly and then varied every 30 trials. Every probe was followed by a 350 ms silent inter-stimulus interval, after which the subsequent trial started automatically. In five percent of all trials, the probe did not come from one of the target positions but from a random speaker within the frontal field. Subjects had to respond to these deviant trials by pressing a random button as fast as possible. If they managed to respond within one second after sound onset, the trial was considered a success. After one second, the trial was considered failed, and stimulation resumed. The task ensured that subjects were paying attention to the sounds' location. The experiment was divided into four blocks, each of which consisted of 504 trials (480 standard trials plus 24 deviants) and lasted 8.4 min. Subjects were instructed to keep their head and gaze aligned with the fixation cross at 0° throughout the recording.

## Experiment II
The second experiment was conducted in the same anechoic chamber using the same array of loudspeakers. 
Again, subjects completed an initial test where they had to localize sounds coming from loudspeakers at +50°, +25°, 0°, -25° and -50°. 
The second experiment started with a localization test as well. During the first 15 trials, each 100 ms stimulus was accompanied by a visual cues (a flashing LED at the loudspeakers' location).


## Statistical Analysis


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
indicates that the subject responded randomly. The initial localization tests showed that subjects could determine the sounds' elevation accurately, indicated by an EG of $0.68 (SD=0.25)$ in the test preceding experiment I and an EG of $0.78 (SD=0.13)$ in the test preceding experiment II
(Fig.\ref{eg} A and B). While the EGs we observed are lower compared to previous reports on free field sound localization [@makous1990, @hofman1998], this is likely due to the much shorter training that subjects received in our study.

The average EG for the behavioral task in experiment II (Fig.\ref{eg}) was $0.45 (SD=0.26$, significantly lower than the EG measured during the prior localization test (paired-samples t-test: $t(30)=6.05, p<0.001$). We expected a decrease in performance because the continuous display of sounds and uncertainty
about which stimuli were targets made the task much more difficult. Despite the decrease in accuracy, most subject's response was clearly modulated by the target's position, indicating that they were able to perceive elevation.

![Linear regression between the target's elevation and the subject's response. (A) and (B): data from the localization tests preceding each experiment. (C) data from the behavioral task during the second experiment. Each gray line represents one subject's responses and the red line indicates the group average.\label{eg}](figures/eg.png)

## Evoked Responses
![Average evoked response from experiment II. (A)-(D): distribution of voltage across the scalp at the time points marked by dashed lines in the ERP. (E): Average voltage across time at each channel with Fz highlighted. The fist red line marks the onset of the adapter, the second the transition to the probe. (F): resuts from the perumutation test where color encodes the number of subjects for whom aparticular sample was part of a significant cluster.\label{erp}](figures/erp.png)



## Decoding


# Discussion (1,500 words maximum, including citations)
- brief statement of the principal findings
- validity and significance in the light of other published work
- extensive discussion of the litertaure is discouraged

