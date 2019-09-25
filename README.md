# SEISMOGRAM SIMILARITY

---


This tutorial presents a novel measure of seismogram similarity that is explicitly invariant to path. The work is based on the paper "Beyond Correlation: A Path-Invariant Measure for Seismogram Similarity" by Joshua Dickey, Brett Borghetti, William Junek and Richard Martin, which can be viewed in full on the ArXiv: https://arxiv.org/pdf/1904.07936.pdf.

The tutorial consists of 5 parts:

1) Background

2) Data Exploration

3) Similarity Model

4) Pairwise Association

5) Template-based Discrimination

---
### BACKGROUND

#### Path-Dominant Similarity:

Similarity search is a popular technique for seismic signal processing, including both signal detection and source discrimination. Traditionally, these techniques rely on the cross-correlation function as the basis for measuring similarity. Unfortunately, seismogram correlation is dominated by path effects, as shown in the figure below:

<img src="images/Path_dominant_similarity.png" width="400px">


The figure shows three Seismograms, each depicting an explosion at a coal mine near Thunder Basin, WY. Seismograms a) and b) depict a common source event (600221452), recorded at two separate seismic stations, ISCO and K22A respectively. Seismogram c) depicts a nearby event (600221802), also recorded at K22A. The correlation and visual similarity between the path-similar waveforms b) and c) is obvious. This path-dominant similarity can be desirable when detecting aftershock sequences from a particular fault, or mining blasts from within a small quarry. For general detection and discrimination, however, path-dominant similarity is problematic, as path differences of even just a quarter wavelength can significantly degrade the correlation of two seismograms.


#### Path-Invariant Similarity:

We now envision a new measure of seismogram similarity, that is path-independant. The notional diagram below illustrates an embedding function, $f(\cdot)$, which is a non-linear transformation that maps time-series seismograms to low-dimensional embeddings. The mappings are desired to be path-invariant and source-specific, such that regardless of the recording station, all seismograms associated with a particular event are mapped closely in the embedding space, and seismograms not associated with that event have more distant embeddings. We propose to learn such an embedding function explicitly, using a specialized convolutional neural network architecture, called a triplet network.

<img src="images/STA_dominant_similarity.png" width="400px">

The Triplet Network is trained on batches of $m$ triples, where each triple is comprised of an anchor object, $X_A^{(i)}$, a positive object, $X_P^{(i)}$, and a negative object, $X_N^{(i)}$. The triplet loss function computes the relative embedding distance between the matched pair and non-matched pair, and loss is accrued whenever the matched pair distance is not smaller than the non-matched distance by some margin, $\alpha$, as given in the Equation below:

$\mathcal{J} =  \sum^{m}_{i=1} [ \langle f(X_A^{(i)}) , f(X_P^{(i)}) \rangle - \langle f(X_A^{(i)}) , f(X_N^{(i)}) \rangle + \alpha ] \small_+ $ 

To learn path-invariant embeddings, we simply pick our triples such that the anchor and positive objects are seismograms sharing the same source event, but recorded at different stations. In this way, the network learns a transformation that is invariant to path, calibration function, recording equipment and station.


